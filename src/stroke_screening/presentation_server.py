"""Loopback-only PresentCoach application server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import secrets
import sys
import tempfile
import time

from flask import Flask, jsonify, request, send_from_directory, session, stream_with_context
from waitress import serve
from werkzeug.exceptions import RequestEntityTooLarge

from .local_camera import LocalCamera
from .presentation_ai import LocalPresentationAIError, OllamaPresentationLLM
from .presentation_audio import AudioCaptureError, TranscriptionError, WhisperCppTranscriber
from .presentation_calibration import calibration_status, prepare_feedback_metrics
from .presentation_camera import CameraBusyError, CameraSessionError, PresentationCameraService
from .presentation_core import compute_metrics, generate_feedback
from .presentation_recording import PresentationRecordingError, PresentationRecordingService
from .presentation_store import (
    ProfileNotFoundError,
    PresentationStorageError,
    PresentationStore,
    StoredPresentation,
)
from .presentation_test_lab import TEST_MEDIA_BY_ID, test_lab_payload
from .presentation_video import (
    LocalVideoAnalyzer,
    MAX_UPLOAD_BYTES,
    SUPPORTED_VIDEO_SUFFIXES,
    VideoImportError,
)


LOGGER = logging.getLogger("presentcoach")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _serialize_archive(archive) -> list[dict[str, object]]:
    return [item.to_dict() for item in archive.sessions]


def create_app(
    *,
    store: PresentationStore | None = None,
    llm: OllamaPresentationLLM | None = None,
    recorder: PresentationRecordingService | None = None,
    video_analyzer: LocalVideoAnalyzer | None = None,
    test_media_dir: Path | None = None,
    reports_dir: Path | None = None,
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
        PERMANENT_SESSION_LIFETIME=3600,
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

    @app.before_request
    def local_only():
        host = request.host.partition(":")[0].strip("[]").lower()
        if host not in {"127.0.0.1", "localhost"}:
            return jsonify(error="PresentCoach accepts local requests only"), 400
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
            "style-src 'self'; font-src 'self'"
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

    def analyze_and_store(profile_id: str, presentation):
        metrics = compute_metrics(presentation)
        archive = current_store().load_profile(profile_id)
        calibration = calibration_status(archive)
        prepared = prepare_feedback_metrics(metrics, calibration)
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
        current_store().append(
            profile_id,
            StoredPresentation(presentation, metrics, feedback),
        )
        if presentation.session_kind == "baseline":
            current_store().update_calibration(profile_id, {
                "baseline_session_id": presentation.session_id,
                "baseline_confirmed": False,
            })
        updated = current_store().load_profile(profile_id)
        return {
            "session": presentation.to_dict(),
            "metrics": metrics,
            "feedback": feedback,
            "calibration": calibration_status(updated),
        }

    def owner() -> str:
        value = session.get("presentation_owner")
        if not isinstance(value, str):
            raise PresentationRecordingError("Start a presentation recording first")
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
        if item is None:
            return jsonify(error="Test clip not found"), 404
        directory = app.extensions["presentation_test_media_dir"]
        path = directory / item["filename"]
        if not path.is_file():
            return jsonify(error="Test clip is not installed locally"), 404
        return send_from_directory(
            directory,
            item["filename"],
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
        if any(state != "good" for state in candidate.session.quality_flags.values()):
            return jsonify(error="The baseline has insufficient face or audio data; record it again"), 422
        current_store().update_calibration(profile_id, {
            "baseline_session_id": candidate.session.session_id,
            "baseline_confirmed": True,
        })
        return jsonify(calibration=calibration_status(current_store().load_profile(profile_id)))

    @app.post("/api/recordings/start")
    def start_recording():
        document = request.get_json(silent=True)
        if not isinstance(document, dict):
            return jsonify(error="A JSON recording request is required"), 400
        profile_id = str(document.get("profile_id", ""))
        archive = current_store().load_profile(profile_id)
        analyzer = current_video_analyzer()
        if analyzer.is_active():
            return jsonify(error="Wait for the imported video analysis to finish"), 409
        calibration = calibration_status(archive)
        stage = calibration["stage"]
        if stage == "review_baseline":
            return jsonify(error="Review and confirm your baseline before recording repeats"), 409
        kind = "practice"
        if stage == "record_baseline":
            kind = "baseline"
        elif stage in {"record_repeats", "repeatability_failed"}:
            kind = "repeat"
        previous = session.pop("presentation_owner", None)
        if isinstance(previous, str):
            try:
                current_recorder().cancel(previous)
            except PresentationRecordingError:
                pass
        recording_owner = current_recorder().start(
            session_kind=kind,
            note=str(document.get("note", ""))[:1000],
        )
        session["presentation_owner"] = recording_owner
        session["presentation_profile"] = profile_id
        return jsonify(
            started=True,
            session_kind=kind,
            preview_url="/api/recordings/stream.mjpg",
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

    @app.post("/api/recordings/stop")
    def stop_recording():
        recording_owner = owner()
        profile_id = session.get("presentation_profile")
        if not isinstance(profile_id, str):
            raise PresentationRecordingError("The recording profile expired")
        presentation = current_recorder().stop(recording_owner)
        session.pop("presentation_owner", None)
        session.pop("presentation_profile", None)
        return jsonify(analyze_and_store(profile_id, presentation)), 201

    @app.post("/api/videos/analyze")
    def analyze_uploaded_video():
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
            local_path = Path(temporary) / f"import{suffix}"
            upload.save(local_path)
            presentation = current_video_analyzer().analyze(local_path, note=note)
        return jsonify(analyze_and_store(profile_id, presentation)), 201

    @app.post("/api/recordings/cancel")
    def cancel_recording():
        recording_owner = session.pop("presentation_owner", None)
        session.pop("presentation_profile", None)
        if isinstance(recording_owner, str):
            try:
                current_recorder().cancel(recording_owner)
            except PresentationRecordingError:
                pass
        return jsonify(cancelled=True)

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
        return send_from_directory(static_dir, "index.html")

    @app.get("/<path:asset_path>")
    def assets(asset_path: str):
        if asset_path.startswith("api/"):
            return jsonify(error="API route not found"), 404
        target = static_dir / asset_path
        if target.is_file() and static_dir in target.resolve().parents:
            return send_from_directory(static_dir, asset_path)
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
    application = create_app()
    LOGGER.info("PresentCoach is running locally at http://%s:%s", args.host, args.port)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
