"""Loopback-only PresentCoach application server."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
import secrets
import stat
import sys
import tempfile
import threading
import time

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    request,
    send_from_directory,
    session,
    stream_with_context,
)
from itsdangerous import BadSignature, URLSafeTimedSerializer
from waitress import serve
from werkzeug.exceptions import RequestEntityTooLarge

from .local_camera import LocalCamera
from .presentation_ai import LocalPresentationAIError, OllamaPresentationLLM
from .presentation_audio import AudioCaptureError, TranscriptionError, WhisperCppTranscriber
from .presentation_calibration import calibration_status, prepare_feedback_metrics
from .presentation_camera import CameraBusyError, CameraSessionError, PresentationCameraService
from .presentation_core import compute_metrics, generate_feedback
from .presentation_recording import (
    RecordedPresentation,
    PresentationRecordingError,
    PresentationRecordingService,
)
from .presentation_review import build_review_cues
from .presentation_store import (
    ProfileNotFoundError,
    SessionMediaNotFoundError,
    PresentationStorageError,
    PresentationStore,
    StoredPresentation,
)
from .presentation_test_lab import (
    TEST_MEDIA_BY_ID,
    resolve_tracking_test_media,
    test_lab_payload,
)
from .presentation_video import (
    LocalVideoAnalyzer,
    MAX_UPLOAD_BYTES,
    SUPPORTED_VIDEO_SUFFIXES,
    VideoImportError,
)


LOGGER = logging.getLogger("presentcoach")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MEDIA_PLAYBACK_TOKEN_SECONDS = 60 * 60
MEDIA_PLAYBACK_RANGE_BYTES = 4 * 1024 * 1024
ACCESS_CAPABILITY_BYTES = 48
AUTHORIZATION_NONCE_BYTES = 32
AUTHORIZATION_SESSION_KEY = "presentation_authorization_nonce"
ACCESS_FILE_NAME = "access-url"


class _StoredFeedbackReplay:
    """Replay stored JSON through the current deterministic feedback verifier."""

    def __init__(self, document: dict[str, object]) -> None:
        self.document = document

    def complete_json(self, **_kwargs) -> dict[str, object]:
        return self.document


def _requested_byte_range(
    header: str | None, size: int
) -> tuple[int, int, bool]:
    """Parse one RFC 7233 byte range as a half-open interval."""

    if size <= 0:
        raise ValueError("Media size must be positive")
    if not header:
        return 0, size, False
    value = header.strip()
    if not value.startswith("bytes=") or "," in value:
        raise ValueError("Only one byte range is supported")
    specification = value[6:]
    if specification.count("-") != 1:
        raise ValueError("The byte range is malformed")
    raw_start, raw_end = specification.split("-", 1)
    if not raw_start:
        if not raw_end.isdigit() or int(raw_end) <= 0:
            raise ValueError("The suffix byte range is malformed")
        length = min(size, int(raw_end), MEDIA_PLAYBACK_RANGE_BYTES)
        return size - length, size, True
    if not raw_start.isdigit():
        raise ValueError("The byte range start is malformed")
    start = int(raw_start)
    if start >= size:
        raise ValueError("The byte range starts beyond the media")
    if raw_end:
        if not raw_end.isdigit():
            raise ValueError("The byte range end is malformed")
        inclusive_end = min(size - 1, int(raw_end))
        if inclusive_end < start:
            raise ValueError("The byte range is reversed")
        end = inclusive_end + 1
    else:
        end = size
    return start, min(end, start + MEDIA_PLAYBACK_RANGE_BYTES), True

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _authorization_nonce() -> str | None:
    value = session.get(AUTHORIZATION_SESSION_KEY)
    return value if isinstance(value, str) and len(value) >= 32 else None


def _locked_page() -> Response:
    return Response(
        (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>PresentCoach locked</title></head><body><main>"
            "<h1>PresentCoach is locked</h1>"
            "<p>Open the private launch URL saved by PresentCoach in the Codex "
            "browser. From Terminal, run:</p>"
            "<pre>cat &quot;$HOME/Library/Application Support/PresentCoach/access-url&quot;</pre>"
            "</main></body></html>"
        ),
        status=401,
        mimetype="text/html",
    )


def _write_access_url(path: Path, url: str) -> None:
    """Atomically publish the private launch URL in a user-only directory."""

    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    parent_metadata = path.parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
        parent_metadata.st_mode
    ):
        raise RuntimeError("PresentCoach access URL directory is unsafe")
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".presentcoach-access-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            descriptor = -1
            handle.write(url + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _remove_access_url(path: Path, url: str) -> None:
    """Remove only the launch file written by this server process."""

    try:
        if path.read_text(encoding="utf-8").strip() == url:
            path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        LOGGER.warning("PresentCoach could not remove its private access URL")


def _serialize_archive(archive) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    calibration = calibration_status(archive)
    for item in archive.sessions:
        document = item.to_dict()
        # Older encrypted sessions remain readable while gaining newly added
        # derived rates/events. Raw transcript and vision samples are the
        # source of truth, so this migration is deterministic and read-only.
        document["metrics"] = compute_metrics(item.session)
        document["session"]["quality_flags"] = document["metrics"]["quality_flags"]
        feedback = document.get("feedback")
        quality = document["metrics"]["quality_flags"]
        prepared = prepare_feedback_metrics(document["metrics"], calibration)
        if isinstance(feedback, dict) and feedback.get("status") == "ready":
            try:
                current_feedback = generate_feedback(
                    prepared, _StoredFeedbackReplay(feedback)
                ).to_dict()
            except (LocalPresentationAIError, TypeError, ValueError):
                current_feedback = None
            if current_feedback != feedback:
                document["feedback"] = {
                    "status": "calibration_required",
                    "strengths": [], "improvements": [],
                    "insufficient_data": [
                        name for name, state in quality.items() if state != "good"
                    ],
                    "message": "Saved feedback no longer matches the current verified measurements.",
                    "source": "guardrail",
                }
        document["coaching_cues"] = build_review_cues(
            prepared, document.get("feedback")
        )
        documents.append(document)
    return documents


def create_app(
    *,
    store: PresentationStore | None = None,
    llm: OllamaPresentationLLM | None = None,
    recorder: PresentationRecordingService | None = None,
    video_analyzer: LocalVideoAnalyzer | None = None,
    test_media_dir: Path | None = None,
    reports_dir: Path | None = None,
    access_capability: str | None = None,
    testing: bool = False,
) -> Flask:
    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask(__name__, static_folder=None)
    app.config.update(
        SECRET_KEY=secrets.token_hex(32),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES + 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_SECURE=False,
        SESSION_COOKIE_NAME="presentcoach_session",
        SESSION_REFRESH_EACH_REQUEST=False,
        TESTING=testing,
    )
    app.extensions["presentation_store"] = store or PresentationStore()
    app.extensions["presentation_llm"] = llm or OllamaPresentationLLM()
    if recorder is None:
        root = _project_root()
        camera = PresentationCameraService(
            width=960, height=540, timeout_seconds=30 * 60 + 30
        )
        transcriber = WhisperCppTranscriber(
            model_path=root / "models" / "whisper" / "ggml-base.en-q5_1.bin"
        )
        recorder = PresentationRecordingService(
            camera=camera,
            model_path=root / "models" / "face_landmarker.task",
            transcriber=transcriber,
        )
        if video_analyzer is None:
            video_analyzer = LocalVideoAnalyzer(
                model_path=root / "models" / "face_landmarker.task",
                transcriber=transcriber,
            )
    app.extensions["presentation_recorder"] = recorder
    app.extensions["presentation_video_analyzer"] = video_analyzer
    root = _project_root()
    app.extensions["presentation_test_media_dir"] = (test_media_dir or root / "test_media").resolve()
    app.extensions["presentation_reports_dir"] = (reports_dir or root / "reports").resolve()
    app.extensions["presentation_test_media_hash_cache"] = {}
    app.extensions["presentation_recording_transition_lock"] = threading.Lock()
    capability = (
        secrets.token_urlsafe(ACCESS_CAPABILITY_BYTES)
        if access_capability is None
        else access_capability
    )
    if not isinstance(capability, str) or len(capability) < 32:
        raise ValueError("PresentCoach access capability must be at least 32 characters")
    app.extensions["presentation_access_capability"] = capability
    app.extensions["presentation_media_signer"] = URLSafeTimedSerializer(
        app.config["SECRET_KEY"], salt="presentcoach-session-media-v1"
    )

    @app.before_request
    def local_only():
        host = request.host.partition(":")[0].strip("[]").lower()
        if host not in {"127.0.0.1", "localhost"}:
            return jsonify(error="PresentCoach accepts local requests only"), 400
        public_api = request.path == "/api/health" or (
            request.method in {"GET", "HEAD"}
            and request.path.startswith("/api/test-media/")
        )
        if (
            request.path.startswith("/api/")
            and not public_api
            and _authorization_nonce() is None
        ):
            return jsonify(
                error="Open PresentCoach with its private local access URL",
                code="local_access_required",
            ), 401
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            expected = session.get("csrf_token")
            supplied = request.headers.get("X-CSRF-Token", "")
            if not expected or not secrets.compare_digest(str(expected), supplied):
                return jsonify(error="The local session token is missing or expired"), 403
            origin = request.headers.get("Origin")
            port = request.host.partition(":")[2] or str(DEFAULT_PORT)
            if origin and origin not in {
                f"http://127.0.0.1:{port}", f"http://localhost:{port}"
            }:
                return jsonify(error="Cross-origin requests are not allowed"), 403
        return None

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; form-action 'self'; "
            "frame-ancestors 'none'; object-src 'none'; img-src 'self' data:; "
            "media-src 'self' blob:; connect-src 'self'; script-src 'self'; "
            "style-src 'self'; style-src-attr 'unsafe-inline'; font-src 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
        response.headers["Cache-Control"] = "no-store"
        return response

    def current_store() -> PresentationStore:
        return app.extensions["presentation_store"]

    def current_recorder() -> PresentationRecordingService:
        return app.extensions["presentation_recorder"]

    def current_video_analyzer() -> LocalVideoAnalyzer:
        analyzer = app.extensions.get("presentation_video_analyzer")
        if analyzer is None:
            root = _project_root()
            analyzer = LocalVideoAnalyzer(
                model_path=root / "models" / "face_landmarker.task",
                transcriber=WhisperCppTranscriber(
                    model_path=root / "models" / "whisper" / "ggml-base.en-q5_1.bin"
                ),
            )
            app.extensions["presentation_video_analyzer"] = analyzer
        return analyzer

    def media_payload(profile_id: str, session_id: str) -> dict[str, object]:
        try:
            media = current_store().session_media(profile_id, session_id)
        except SessionMediaNotFoundError:
            return {"available": False}
        except PresentationStorageError as error:
            LOGGER.warning("Encrypted session media is unavailable: %s", error)
            return {"available": False}
        signer = app.extensions["presentation_media_signer"]
        authorization_nonce = _authorization_nonce()
        if authorization_nonce is None:
            raise PresentationStorageError(
                "The local browser authorization is unavailable"
            )
        token = signer.dumps({
            "profile_id": profile_id,
            "session_id": session_id,
            "authorization_nonce": authorization_nonce,
        })
        return {
            "available": True,
            "playback_url": (
                f"/api/profiles/{profile_id}/sessions/{session_id}/video?token={token}"
            ),
            "mime_type": media.mime_type,
            "source": media.source,
            "token_expires_in_seconds": MEDIA_PLAYBACK_TOKEN_SECONDS,
        }

    def analyze_and_store(
        profile_id: str,
        presentation,
        *,
        media_path: Path | None = None,
        media_mime_type: str | None = None,
        media_source: str | None = None,
        media_error: str | None = None,
    ):
        metrics = compute_metrics(presentation)
        archive = current_store().load_profile(profile_id)
        calibration = calibration_status(archive)
        prepared = prepare_feedback_metrics(metrics, calibration)
        feedback = {
            "status": "feedback_pending",
            "strengths": [],
            "improvements": [],
            "insufficient_data": list(metrics.get("insufficient_metrics", [])),
            "message": (
                "The measurements and any available replay were saved before "
                "local feedback generation. Generate feedback to retry."
            ),
            "source": "guardrail",
        }
        # Commit the deterministic measurement record before optional Ollama
        # work. A stalled or interrupted local model must never erase a
        # completed camera/Whisper session.
        current_store().append(
            profile_id,
            StoredPresentation(presentation, metrics, feedback),
        )
        if media_path is not None and media_mime_type and media_source:
            try:
                current_store().put_session_media(
                    profile_id,
                    presentation.session_id,
                    media_path,
                    mime_type=media_mime_type,
                    source=media_source,
                )
            except PresentationStorageError as error:
                LOGGER.warning("Session measurements saved without video: %s", error)
                media_error = (
                    "The measurements were saved, but the local video could not "
                    "be retained safely."
                )
        if presentation.session_kind == "baseline":
            current_store().update_calibration(profile_id, {
                "baseline_session_id": presentation.session_id,
                "baseline_confirmed": False,
            })
        try:
            feedback = generate_feedback(
                prepared, app.extensions["presentation_llm"]
            ).to_dict()
        except (LocalPresentationAIError, ValueError) as error:
            LOGGER.warning("Local feedback failed closed: %s", error)
            feedback = {
                "status": "local_ai_unavailable",
                "strengths": [],
                "improvements": [],
                "insufficient_data": list(metrics.get("insufficient_metrics", [])),
                "message": "The measurements were saved, but local feedback failed verification.",
                "source": "guardrail",
            }
        current_store().replace_feedback(
            profile_id, presentation.session_id, feedback
        )
        updated = current_store().load_profile(profile_id)
        retained_media = media_payload(profile_id, presentation.session_id)
        if not retained_media["available"] and media_error:
            retained_media["message"] = (
                "The measurements were saved, but the local video is unavailable."
            )
        return {
            "session": presentation.to_dict(),
            "metrics": metrics,
            "feedback": feedback,
            "coaching_cues": build_review_cues(metrics, feedback),
            "calibration": calibration_status(updated),
            "media": retained_media,
        }

    def recording_client_id(value) -> str:
        candidate = str(value or "")
        if (
            not 16 <= len(candidate) <= 128
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in candidate)
        ):
            raise PresentationRecordingError("The recording tab identity is invalid")
        return candidate

    def owner() -> str:
        value = session.get("presentation_owner")
        if not isinstance(value, str):
            raise PresentationRecordingError("Start a presentation recording first")
        expected_client = session.get("presentation_client_id")
        if request.method in {"GET", "HEAD"}:
            supplied_client = request.args.get("client_id")
        else:
            document = request.get_json(silent=True)
            supplied_client = (
                document.get("client_id") if isinstance(document, dict) else None
            )
        candidate = recording_client_id(supplied_client)
        if not isinstance(expected_client, str) or not secrets.compare_digest(
            expected_client, candidate
        ):
            raise PresentationRecordingError(
                "This browser tab does not own the active recording"
            )
        return value

    @app.get("/api/health")
    def health():
        return jsonify(status="ready", local_only=True)

    @app.get("/api/bootstrap")
    def bootstrap():
        csrf = session.get("csrf_token")
        if not csrf:
            csrf = secrets.token_urlsafe(32)
            session["csrf_token"] = csrf
        profiles = current_store().list_profiles()
        requested = request.args.get("profile", "")
        profile = None
        sessions: list[dict[str, object]] = []
        calibration = {
            "stage": "create_profile", "ready": False,
            "message": "Create a local profile to begin.",
        }
        if profiles:
            allowed = {item["id"] for item in profiles}
            selected = requested if requested in allowed else profiles[0]["id"]
            archive = current_store().load_profile(selected)
            profile = {"id": archive.profile_id, "name": archive.name}
            sessions = _serialize_archive(archive)
            for document in sessions:
                raw_session = document.get("session")
                if isinstance(raw_session, dict):
                    session_id = raw_session.get("session_id")
                    if isinstance(session_id, str):
                        document["media"] = media_payload(selected, session_id)
            calibration = calibration_status(archive)
        return jsonify(
            csrf_token=csrf,
            profiles=profiles,
            profile=profile,
            sessions=sessions,
            calibration=calibration,
            local_model=app.extensions["presentation_llm"].status(),
            whisper={"available": True, "engine": "whisper.cpp", "model": "base.en-q5_1"},
            test_lab=test_lab_payload(
                media_dir=app.extensions["presentation_test_media_dir"],
                reports_dir=app.extensions["presentation_reports_dir"],
            ),
            privacy="Camera, microphone, transcript, metrics, and AI stay on this Mac.",
        )

    @app.get("/api/test-media/<media_id>/video")
    def test_media_video(media_id: str):
        if request.headers.get("Sec-Fetch-Site", "same-origin") not in {"same-origin", "none"}:
            return jsonify(error="Cross-site test media requests are not allowed"), 403
        item = TEST_MEDIA_BY_ID.get(media_id)
        directory = app.extensions["presentation_test_media_dir"]
        if item is not None:
            path = directory / item["filename"]
            expected_sha256 = item.get("sha256")
        else:
            contract = resolve_tracking_test_media(
                media_dir=directory, media_id=media_id
            )
            if contract is None:
                return jsonify(error="Test clip not found"), 404
            path = contract["path"]
            expected_sha256 = contract["sha256"]
        if not isinstance(path, Path) or not isinstance(expected_sha256, str):
            return jsonify(error="Test clip not found"), 404
        try:
            stat = path.stat()
        except OSError:
            return jsonify(error="Test clip not found"), 404
        fingerprint = (
            str(path), stat.st_size, stat.st_mtime_ns, expected_sha256,
        )
        digest_cache = app.extensions["presentation_test_media_hash_cache"]
        verified = digest_cache.get(media_id) == fingerprint
        if not verified:
            digest = hashlib.sha256()
            try:
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError:
                return jsonify(error="Test clip not found"), 404
            verified = secrets.compare_digest(digest.hexdigest(), expected_sha256)
            if verified:
                digest_cache[media_id] = fingerprint
            else:
                digest_cache.pop(media_id, None)
        if not verified:
            return jsonify(error="Test clip not found"), 404
        return send_from_directory(
            directory,
            path.name,
            mimetype="video/webm",
            conditional=True,
        )

    @app.post("/api/profiles")
    def create_profile():
        document = request.get_json(silent=True)
        if not isinstance(document, dict):
            return jsonify(error="A JSON profile request is required"), 400
        return jsonify(profile=current_store().create_profile(document.get("name"))), 201

    @app.post("/api/profiles/<profile_id>/calibration/confirm")
    def confirm_baseline(profile_id: str):
        archive = current_store().load_profile(profile_id)
        candidate_id = archive.calibration.get("baseline_session_id")
        candidate = next(
            (item for item in archive.sessions if item.session.session_id == candidate_id),
            None,
        )
        if candidate is None:
            candidate = next(
                (item for item in archive.sessions if item.session.session_kind == "baseline"),
                None,
            )
        if candidate is None:
            return jsonify(error="Record a baseline before confirming it"), 409
        if candidate.session.duration_seconds < 30:
            return jsonify(error="The baseline must be at least 30 seconds"), 422
        current_quality = compute_metrics(candidate.session)["quality_flags"]
        if any(state != "good" for state in current_quality.values()):
            return jsonify(error="The baseline has insufficient face or audio data; record it again"), 422
        current_store().update_calibration(profile_id, {
            "baseline_session_id": candidate.session.session_id,
            "baseline_confirmed": True,
        })
        return jsonify(calibration=calibration_status(current_store().load_profile(profile_id)))

    @app.post("/api/profiles/<profile_id>/sessions/<session_id>/feedback")
    def regenerate_session_feedback(profile_id: str, session_id: str):
        if current_recorder().is_active() or current_video_analyzer().is_active():
            return jsonify(error="Wait for the active recording or video analysis to finish"), 409
        archive = current_store().load_profile(profile_id)
        stored = next(
            (item for item in archive.sessions if item.session.session_id == session_id),
            None,
        )
        if stored is None:
            return jsonify(error="The presentation session was not found"), 404
        metrics = compute_metrics(stored.session)
        prepared = prepare_feedback_metrics(metrics, calibration_status(archive))
        feedback = generate_feedback(
            prepared, app.extensions["presentation_llm"]
        ).to_dict()
        current_store().replace_feedback(profile_id, session_id, feedback)
        return jsonify(
            feedback=feedback,
            coaching_cues=build_review_cues(metrics, feedback),
        )

    @app.get("/api/profiles/<profile_id>/sessions/<session_id>/video")
    def session_video(profile_id: str, session_id: str):
        if request.headers.get("Sec-Fetch-Site", "same-origin") not in {
            "same-origin", "none"
        }:
            return jsonify(
                error="Cross-site session video requests are not allowed",
                code="media_access_denied",
            ), 403
        token = request.args.get("token", "")
        signer = app.extensions["presentation_media_signer"]
        try:
            grant = signer.loads(token, max_age=MEDIA_PLAYBACK_TOKEN_SECONDS)
        except BadSignature:
            return jsonify(
                error="The local video access token is missing or invalid",
                code="media_access_denied",
            ), 403
        if (
            not isinstance(grant, dict)
            or not secrets.compare_digest(str(grant.get("profile_id", "")), profile_id)
            or not secrets.compare_digest(str(grant.get("session_id", "")), session_id)
            or not secrets.compare_digest(
                str(grant.get("authorization_nonce", "")),
                _authorization_nonce() or "",
            )
        ):
            return jsonify(
                error="The local video access token does not match this session",
                code="media_access_denied",
            ), 403
        try:
            reader = current_store().open_session_media(profile_id, session_id)
        except (SessionMediaNotFoundError, PresentationStorageError) as error:
            LOGGER.warning("Session video playback failed closed: %s", error)
            return jsonify(
                error="The encrypted local video is unavailable",
                code="media_unavailable",
            ), 404
        try:
            start, end, partial = _requested_byte_range(
                request.headers.get("Range"), reader.metadata.plaintext_bytes
            )
        except ValueError:
            size = reader.metadata.plaintext_bytes
            reader.close()
            response = jsonify(
                error="The requested video byte range is invalid",
                code="media_range_invalid",
            )
            response.status_code = 416
            response.headers["Content-Range"] = f"bytes */{size}"
            response.headers["Accept-Ranges"] = "bytes"
            return response

        try:
            reader.authenticate_range(start, end)
        except PresentationStorageError as error:
            reader.close()
            LOGGER.warning(
                "Session video requested range failed authentication: %s", error
            )
            return jsonify(
                error="The encrypted local video is unavailable",
                code="media_unavailable",
            ), 404

        status = 206 if partial else 200
        length = end - start
        if request.method == "HEAD":
            reader.close()
            response = Response(status=status, mimetype=reader.metadata.mime_type)
        else:
            chunks = reader.iter_range(start, end)
            try:
                first_chunk = next(chunks)
            except (StopIteration, PresentationStorageError) as error:
                reader.close()
                LOGGER.warning("Session video chunk failed authentication: %s", error)
                return jsonify(
                    error="The encrypted local video is unavailable",
                    code="media_unavailable",
                ), 404

            def encrypted_media_stream():
                try:
                    yield first_chunk
                    yield from chunks
                finally:
                    reader.close()

            response = Response(
                encrypted_media_stream(),
                status=status,
                mimetype=reader.metadata.mime_type,
                direct_passthrough=True,
            )
            response.call_on_close(reader.close)
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["Content-Length"] = str(length)
        if partial:
            response.headers["Content-Range"] = (
                f"bytes {start}-{end - 1}/{reader.metadata.plaintext_bytes}"
            )
        response.headers["Content-Disposition"] = "inline"
        return response

    @app.delete("/api/profiles/<profile_id>/sessions/<session_id>/video")
    def delete_session_video(profile_id: str, session_id: str):
        current_store().delete_session_media(profile_id, session_id)
        return jsonify(deleted=True, media={"available": False})

    @app.post("/api/recordings/start")
    def start_recording():
        transition_lock = app.extensions["presentation_recording_transition_lock"]
        if not transition_lock.acquire(blocking=False):
            return jsonify(
                error="Wait for the previous recording to finish saving"
            ), 409
        try:
            return _start_recording_locked()
        finally:
            transition_lock.release()

    def _start_recording_locked():
        document = request.get_json(silent=True)
        if not isinstance(document, dict):
            return jsonify(error="A JSON recording request is required"), 400
        profile_id = str(document.get("profile_id", ""))
        client_id = recording_client_id(document.get("client_id"))
        archive = current_store().load_profile(profile_id)
        analyzer = current_video_analyzer()
        if analyzer.is_active():
            return jsonify(error="Wait for the imported video analysis to finish"), 409
        if current_recorder().is_active():
            return jsonify(error="A live recording is already active in another tab"), 409
        calibration = calibration_status(archive)
        stage = calibration["stage"]
        if stage == "review_baseline":
            return jsonify(error="Review and confirm your baseline before recording repeats"), 409
        kind = "practice"
        if stage == "record_baseline":
            kind = "baseline"
        elif stage in {"record_repeats", "repeatability_failed"}:
            kind = "repeat"
        recording_owner = current_recorder().start(
            session_kind=kind,
            note=str(document.get("note", ""))[:1000],
        )
        session["presentation_owner"] = recording_owner
        session["presentation_profile"] = profile_id
        session["presentation_client_id"] = client_id
        return jsonify(
            started=True,
            session_kind=kind,
            preview_url=f"/api/recordings/stream.mjpg?client_id={client_id}",
        )

    @app.get("/api/recordings/status")
    def recording_status():
        return jsonify(current_recorder().status(owner()))

    @app.get("/api/recordings/stream.mjpg")
    def recording_stream():
        if request.headers.get("Sec-Fetch-Site", "same-origin") not in {"same-origin", "none"}:
            return jsonify(error="Cross-site previews are not allowed"), 403
        recording_owner = owner()

        @stream_with_context
        def frames():
            while True:
                started = time.monotonic()
                try:
                    jpeg = current_recorder().preview_jpeg(recording_owner)
                except (PresentationRecordingError, CameraSessionError):
                    break
                yield (
                    b"--presentcoach-frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg + b"\r\n"
                )
                time.sleep(max(0.0, 0.125 - (time.monotonic() - started)))

        return app.response_class(
            frames(),
            mimetype="multipart/x-mixed-replace; boundary=presentcoach-frame",
            direct_passthrough=True,
        )

    def transition_error_response(error: Exception, transition_lock):
        """Render an error while retaining transition ownership through send."""

        try:
            response = app.make_response(app.handle_user_exception(error))
        except Exception:
            LOGGER.exception("Unhandled local recording transition failure")
            response = jsonify(
                error="The local recording transition failed safely"
            )
            response.status_code = 500
        response.call_on_close(transition_lock.release)
        return response

    def clear_recording_ownership() -> None:
        session.pop("presentation_owner", None)
        session.pop("presentation_profile", None)
        session.pop("presentation_client_id", None)

    def recorder_confirmed_inactive(recorder) -> bool:
        try:
            return not recorder.is_active()
        except Exception:
            # Preserve the only retry capability when native state cannot be
            # confirmed. A later authenticated status/cancel can recover it.
            return False

    @app.post("/api/recordings/stop")
    def stop_recording():
        transition_lock = app.extensions["presentation_recording_transition_lock"]
        transition_lock.acquire()
        release_with_response = False
        try:
            recording_owner = owner()
            profile_id = session.get("presentation_profile")
            if not isinstance(profile_id, str):
                raise PresentationRecordingError("The recording profile expired")
            recorder = current_recorder()
            try:
                completed = recorder.stop(recording_owner)
            except Exception:
                if recorder_confirmed_inactive(recorder):
                    clear_recording_ownership()
                raise
            else:
                clear_recording_ownership()
            if isinstance(completed, RecordedPresentation):
                try:
                    document = analyze_and_store(
                        profile_id,
                        completed.session,
                        media_path=(completed.media.path if completed.media else None),
                        media_mime_type=(completed.media.mime_type if completed.media else None),
                        media_source=(completed.media.source if completed.media else None),
                        media_error=completed.media_error,
                    )
                finally:
                    completed.close()
            else:
                document = analyze_and_store(profile_id, completed)
            response = jsonify(document)
            response.status_code = 201
            response.call_on_close(transition_lock.release)
            release_with_response = True
            return response
        except Exception as error:
            # Build handled terminal failures before releasing the transition
            # lock. Their session-clearing Set-Cookie must be fully emitted
            # before another Start is allowed to publish a new owner tuple.
            response = transition_error_response(error, transition_lock)
            release_with_response = True
            return response
        finally:
            if not release_with_response:
                transition_lock.release()

    @app.post("/api/videos/analyze")
    def analyze_uploaded_video():
        transition_lock = app.extensions["presentation_recording_transition_lock"]
        if not transition_lock.acquire(blocking=False):
            return jsonify(
                error="Wait for the current recording or video analysis to finish"
            ), 409
        release_with_response = False
        try:
            if current_recorder().is_active():
                return jsonify(error="Stop the live recording before importing a video"), 409
            profile_id = str(request.form.get("profile_id", ""))
            current_store().load_profile(profile_id)
            upload = request.files.get("video")
            if upload is None or not upload.filename:
                return jsonify(error="Choose a local video file first"), 400
            suffix = Path(upload.filename).suffix.casefold()
            if suffix not in SUPPORTED_VIDEO_SUFFIXES:
                return jsonify(error="Use an MP4, MOV, M4V, or WebM video"), 415
            note = str(request.form.get("note", ""))[:1000]
            with tempfile.TemporaryDirectory(prefix="presentcoach-video-") as temporary:
                workspace = Path(temporary)
                local_path = workspace / f"import{suffix}"
                upload.save(local_path)
                imported = current_video_analyzer().analyze_with_playback(
                    local_path,
                    workspace / "playback.mp4",
                    note=note,
                )
                result = analyze_and_store(
                    profile_id,
                    imported.session,
                    media_path=(imported.playback.path if imported.playback else None),
                    media_mime_type=(
                        imported.playback.mime_type if imported.playback else None
                    ),
                    media_source="upload",
                    media_error=imported.media_error,
                )
            response = jsonify(result)
            response.status_code = 201
            response.call_on_close(transition_lock.release)
            release_with_response = True
            return response
        except Exception as error:
            response = transition_error_response(error, transition_lock)
            release_with_response = True
            return response
        finally:
            if not release_with_response:
                transition_lock.release()

    @app.post("/api/recordings/cancel")
    def cancel_recording():
        transition_lock = app.extensions["presentation_recording_transition_lock"]
        transition_lock.acquire()
        release_with_response = False
        try:
            recording_owner = owner()
            recorder = current_recorder()
            try:
                recorder.cancel(recording_owner)
            except Exception:
                if recorder_confirmed_inactive(recorder):
                    clear_recording_ownership()
                raise
            else:
                clear_recording_ownership()
            response = jsonify(cancelled=True)
            response.call_on_close(transition_lock.release)
            release_with_response = True
            return response
        except Exception as error:
            response = transition_error_response(error, transition_lock)
            release_with_response = True
            return response
        finally:
            if not release_with_response:
                transition_lock.release()

    def recording_error(error):
        return jsonify(error=str(error)), 409

    app.register_error_handler(PresentationRecordingError, recording_error)
    app.register_error_handler(CameraSessionError, recording_error)

    @app.errorhandler(CameraBusyError)
    def busy_error(error):
        return jsonify(error=str(error)), 409

    def audio_error(error):
        LOGGER.warning("Local audio operation failed: %s", error)
        return jsonify(error=str(error)), 503

    app.register_error_handler(AudioCaptureError, audio_error)
    app.register_error_handler(TranscriptionError, audio_error)

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return jsonify(error="Videos must be 512 MB or smaller"), 413

    @app.errorhandler(VideoImportError)
    def video_error(error):
        LOGGER.warning("Local video import rejected: %s", error)
        return jsonify(error=str(error)), 422

    def storage_error(_error):
        LOGGER.exception("Encrypted PresentCoach storage failed")
        return jsonify(error="The encrypted local session history could not be opened safely"), 500

    app.register_error_handler(PresentationStorageError, storage_error)

    @app.errorhandler(ProfileNotFoundError)
    def missing_profile(_error):
        return jsonify(error="The local profile was not found"), 404

    def validation_error(error):
        LOGGER.warning("PresentCoach operation rejected: %s", error)
        return jsonify(error=str(error)), 422

    app.register_error_handler(ValueError, validation_error)
    app.register_error_handler(LocalPresentationAIError, validation_error)

    @app.get("/")
    def index():
        supplied = request.args.get("access")
        if supplied is not None:
            capability = app.extensions["presentation_access_capability"]
            if not secrets.compare_digest(str(supplied), str(capability)):
                return _locked_page()
            # Reopening the current process's private URL in an already
            # authorized browser must not orphan an active recording by
            # clearing its owner/client tuple. Initial authorization still
            # clears all preexisting state to prevent session fixation.
            if _authorization_nonce() is not None:
                return redirect("/", code=303)
            session.clear()
            session[AUTHORIZATION_SESSION_KEY] = secrets.token_urlsafe(
                AUTHORIZATION_NONCE_BYTES
            )
            return redirect("/", code=303)
        if _authorization_nonce() is None:
            return _locked_page()
        return send_from_directory(static_dir, "index.html")

    @app.get("/<path:asset_path>")
    def assets(asset_path: str):
        if asset_path.startswith("api/"):
            return jsonify(error="API route not found"), 404
        target = static_dir / asset_path
        if target.is_file() and static_dir in target.resolve().parents:
            return send_from_directory(static_dir, asset_path)
        if _authorization_nonce() is None:
            return _locked_page()
        return send_from_directory(static_dir, "index.html")

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local PresentCoach app")
    parser.add_argument("--host", default=DEFAULT_HOST, choices=[DEFAULT_HOST])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if sys.platform == "darwin":
        camera = LocalCamera(0, width=640, height=480)
        try:
            camera.open()
            camera.read()
            LOGGER.info("Local camera permission check passed")
        except RuntimeError as error:
            LOGGER.warning("Local camera permission check failed: %s", error)
        finally:
            camera.close()
    access_capability = secrets.token_urlsafe(ACCESS_CAPABILITY_BYTES)
    application = create_app(access_capability=access_capability)
    access_url = (
        f"http://{args.host}:{args.port}/?access={access_capability}"
    )
    access_file = (
        application.extensions["presentation_store"].data_dir / ACCESS_FILE_NAME
    )
    _write_access_url(access_file, access_url)
    LOGGER.info("PresentCoach is running locally at http://%s:%s", args.host, args.port)
    LOGGER.info("Private Codex-browser URL saved to %s", access_file)
    try:
        serve(
            application,
            host=args.host,
            port=args.port,
            threads=6,
            connection_limit=24,
            channel_timeout=180,
            url_scheme="http",
            ident="PresentCoach",
        )
    finally:
        _remove_access_url(access_file, access_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
