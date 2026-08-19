"""Authenticated, encrypted local history for PresentCoach."""

from __future__ import annotations

import base64
import binascii
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .presentation_core import (
    AudioMetrics,
    FillerOccurrence,
    PaceWindow,
    PresentationSession,
    TranscriptWord,
    VisionSample,
)
KEYCHAIN_SERVICE = "com.tanav.presentcoach.profile-key.v1"
SCHEMA = "presentcoach-profile-v1"
ENVELOPE_SCHEMA = "presentcoach-envelope-v1"
MAX_BYTES = 32 * 1024 * 1024
PROFILE_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class PresentationStorageError(RuntimeError):
    """Raised when encrypted presentation history cannot be trusted."""


class ProfileNotFoundError(PresentationStorageError):
    """Raised when a profile key or encrypted archive is missing."""


class KeyStore(Protocol):
    def save(self, profile: str, key: bytes) -> None: ...
    def load(self, profile: str) -> bytes: ...
    def delete(self, profile: str) -> None: ...


def validate_profile_id(profile_id: str) -> str:
    if not isinstance(profile_id, str) or not PROFILE_ID_PATTERN.fullmatch(profile_id):
        raise ValueError("Profile id must be a 32-character lowercase hex value")
    return profile_id


def validate_display_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("Profile name must be text")
    value = " ".join(name.split())
    if not 1 <= len(value) <= 80:
        raise ValueError("Profile name must contain 1-80 characters")
    return value


def _encode_key(key: bytes) -> str:
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("Encryption keys must contain 32 bytes")
    return base64.b64encode(key).decode("ascii")


def _decode_key(value: str) -> bytes:
    try:
        key = base64.b64decode(value.strip(), validate=True)
    except (ValueError, binascii.Error) as error:
        raise PresentationStorageError("The Keychain encryption key is corrupted") from error
    if len(key) != 32:
        raise PresentationStorageError("The Keychain encryption key has an invalid length")
    return key


@dataclass(frozen=True)
class MacOSKeychainKeyStore:
    """Keep one random profile key in macOS Keychain, never in argv."""

    security_path: str = "/usr/bin/security"
    service: str = KEYCHAIN_SERVICE

    def _run(
        self, arguments: list[str], *, secret_input: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [self.security_path, *arguments],
                input=secret_input,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
                start_new_session=True,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PresentationStorageError("macOS Keychain could not be reached") from error

    def save(self, profile: str, key: bytes) -> None:
        safe_profile = validate_profile_id(profile)
        encoded = _encode_key(key)
        result = self._run(
            [
                "add-generic-password", "-a", safe_profile, "-s", self.service,
                "-D", "PresentCoach encryption key", "-l", f"PresentCoach: {safe_profile}",
                "-j", "AES key for one local PresentCoach profile", "-U", "-T", "", "-w",
            ],
            secret_input=encoded + "\n" + encoded + "\n",
        )
        if result.returncode != 0:
            raise PresentationStorageError("The profile key was not saved to Keychain")

    def load(self, profile: str) -> bytes:
        safe_profile = validate_profile_id(profile)
        result = self._run(
            ["find-generic-password", "-a", safe_profile, "-s", self.service, "-w"]
        )
        if result.returncode == 44:
            raise ProfileNotFoundError("The PresentCoach profile key is missing")
        if result.returncode != 0:
            raise PresentationStorageError("The profile key could not be read from Keychain")
        return _decode_key(result.stdout)

    def delete(self, profile: str) -> None:
        safe_profile = validate_profile_id(profile)
        result = self._run(
            ["delete-generic-password", "-a", safe_profile, "-s", self.service]
        )
        if result.returncode not in {0, 44}:
            raise PresentationStorageError("The profile key could not be deleted")


@dataclass(frozen=True)
class StoredPresentation:
    session: PresentationSession
    metrics: dict[str, object]
    feedback: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "session": self.session.to_dict(),
            "metrics": self.metrics,
            "feedback": self.feedback,
        }


def _session_from_dict(raw: object) -> PresentationSession:
    if not isinstance(raw, dict):
        raise ValueError("Stored session must be an object")
    transcript = tuple(TranscriptWord(**item) for item in raw.get("transcript", ()))
    vision = tuple(VisionSample(**item) for item in raw.get("vision_metrics", ()))
    audio_raw = raw.get("audio_metrics")
    if not isinstance(audio_raw, dict):
        raise ValueError("Stored audio metrics must be an object")
    audio = AudioMetrics(
        fillers=tuple(FillerOccurrence(**item) for item in audio_raw.get("fillers", ())),
        filler_counts={str(key): int(value) for key, value in audio_raw.get("filler_counts", {}).items()},
        pace_windows=tuple(PaceWindow(**item) for item in audio_raw.get("pace_windows", ())),
        pauses_seconds=tuple(float(value) for value in audio_raw.get("pauses_seconds", ())),
        pauses_over_2_seconds=int(audio_raw.get("pauses_over_2_seconds", 0)),
        total_duration_seconds=float(audio_raw.get("total_duration_seconds", 0)),
        overall_words_per_minute=float(audio_raw.get("overall_words_per_minute", 0)),
        waveform_rms=float(audio_raw.get("waveform_rms", 0)),
    )
    session = PresentationSession(
        session_id=validate_profile_id(str(raw.get("session_id", ""))),
        start_time=str(raw.get("start_time", "")),
        duration_seconds=float(raw.get("duration_seconds", 0)),
        transcript_text=str(raw.get("transcript_text", ""))[:250_000],
        transcript=transcript,
        vision_metrics=vision,
        audio_metrics=audio,
        quality_flags={str(key): str(value) for key, value in raw.get("quality_flags", {}).items()},
        session_kind=str(raw.get("session_kind", "practice")),
        note=(str(raw["note"])[:1000] if raw.get("note") else None),
    )
    if session.duration_seconds < 0 or session.duration_seconds > 30 * 60 + 5:
        raise ValueError("Stored session duration is outside the limit")
    if session.session_kind not in {"baseline", "repeat", "practice", "imported"}:
        raise ValueError("Stored session kind is invalid")
    return session


@dataclass(frozen=True)
class PresentationArchive:
    profile_id: str
    name: str
    sessions: tuple[StoredPresentation, ...] = ()
    calibration: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PresentationStore:
    data_dir: Path = Path.home() / "Library" / "Application Support" / "PresentCoach"
    key_store: KeyStore = field(
        default_factory=lambda: MacOSKeychainKeyStore(service=KEYCHAIN_SERVICE)
    )

    def _directory(self) -> None:
        try:
            metadata = self.data_dir.lstat()
        except FileNotFoundError:
            self.data_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
            metadata = self.data_dir.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise PresentationStorageError("PresentCoach storage must be a real directory")
        os.chmod(self.data_dir, 0o700)

    def _path(self, profile_id: str) -> Path:
        return self.data_dir / f"{validate_profile_id(profile_id)}.presentcoach"

    @staticmethod
    def _aad(profile_id: str) -> bytes:
        return f"{ENVELOPE_SCHEMA}:{validate_profile_id(profile_id)}".encode()

    def _serialize(self, archive: PresentationArchive) -> bytes:
        document = {
            "schema": SCHEMA,
            "profile_id": validate_profile_id(archive.profile_id),
            "name": validate_display_name(archive.name),
            "sessions": [stored.to_dict() for stored in archive.sessions],
            "calibration": archive.calibration,
        }
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        if len(payload) > MAX_BYTES:
            raise PresentationStorageError("The encrypted session history is full")
        return payload

    def _deserialize(self, profile_id: str, payload: bytes) -> PresentationArchive:
        try:
            document = json.loads(payload.decode())
            if document.get("schema") != SCHEMA or document.get("profile_id") != profile_id:
                raise ValueError("Stored profile identity is invalid")
            sessions = []
            for raw in document.get("sessions", ()):
                if not isinstance(raw, dict) or not isinstance(raw.get("metrics"), dict):
                    raise ValueError("Stored presentation is invalid")
                feedback = raw.get("feedback")
                if feedback is not None and not isinstance(feedback, dict):
                    raise ValueError("Stored feedback is invalid")
                sessions.append(StoredPresentation(
                    session=_session_from_dict(raw.get("session")),
                    metrics=raw["metrics"],
                    feedback=feedback,
                ))
            archive = PresentationArchive(
                profile_id=profile_id,
                name=validate_display_name(document.get("name")),
                sessions=tuple(sessions),
                calibration=(
                    document.get("calibration", {})
                    if isinstance(document.get("calibration", {}), dict)
                    else {}
                ),
            )
            self._serialize(archive)
            return archive
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as error:
            raise PresentationStorageError("The decrypted presentation history is invalid") from error

    def _write(self, archive: PresentationArchive, key: bytes) -> None:
        self._directory()
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(key).encrypt(nonce, self._serialize(archive), self._aad(archive.profile_id))
        envelope = json.dumps({
            "schema": ENVELOPE_SCHEMA,
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex(),
        }, separators=(",", ":")).encode()
        if len(envelope) > MAX_BYTES:
            raise PresentationStorageError("The encrypted session history is full")
        descriptor, temporary = tempfile.mkstemp(prefix=".presentcoach-", dir=self.data_dir)
        temporary_path = Path(temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(envelope)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path(archive.profile_id))
            directory_descriptor = os.open(self.data_dir, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_path.exists():
                temporary_path.unlink()

    def load_profile(self, profile_id: str) -> PresentationArchive:
        profile_id = validate_profile_id(profile_id)
        path = self._path(profile_id)
        flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as error:
            raise ProfileNotFoundError("The PresentCoach profile was not found") from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_BYTES:
                raise PresentationStorageError("The encrypted history file is invalid")
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                descriptor = -1
                raw = handle.read(MAX_BYTES + 1)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        try:
            envelope = json.loads(raw.decode())
            if envelope.get("schema") != ENVELOPE_SCHEMA:
                raise ValueError("Envelope schema is invalid")
            nonce = bytes.fromhex(envelope["nonce"])
            ciphertext = bytes.fromhex(envelope["ciphertext"])
            if len(nonce) != 12:
                raise ValueError("Envelope nonce is invalid")
            key = self.key_store.load(profile_id)
            payload = AESGCM(key).decrypt(nonce, ciphertext, self._aad(profile_id))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, InvalidTag) as error:
            raise PresentationStorageError("The encrypted history failed authentication") from error
        return self._deserialize(profile_id, payload)

    def create_profile(self, name: str) -> dict[str, str]:
        archive = PresentationArchive(secrets.token_hex(16), validate_display_name(name))
        key = secrets.token_bytes(32)
        self.key_store.save(archive.profile_id, key)
        try:
            self._write(archive, key)
        except Exception:
            try:
                self.key_store.delete(archive.profile_id)
            except Exception:
                pass
            raise
        return {"id": archive.profile_id, "name": archive.name}

    def list_profiles(self) -> list[dict[str, str]]:
        self._directory()
        profiles: list[dict[str, str]] = []
        for path in sorted(self.data_dir.glob("*.presentcoach")):
            try:
                archive = self.load_profile(path.stem)
            except (PresentationStorageError, ProfileNotFoundError, ValueError):
                continue
            profiles.append({"id": archive.profile_id, "name": archive.name})
        return profiles

    def append(self, profile_id: str, stored: StoredPresentation) -> None:
        archive = self.load_profile(profile_id)
        if any(item.session.session_id == stored.session.session_id for item in archive.sessions):
            raise ValueError("This presentation session is already stored")
        updated = PresentationArchive(
            archive.profile_id, archive.name, archive.sessions + (stored,), archive.calibration
        )
        self._write(updated, self.key_store.load(profile_id))

    def replace_feedback(self, profile_id: str, session_id: str, feedback: dict[str, object]) -> None:
        archive = self.load_profile(profile_id)
        replaced = False
        sessions: list[StoredPresentation] = []
        for item in archive.sessions:
            if item.session.session_id == session_id:
                sessions.append(StoredPresentation(item.session, item.metrics, feedback))
                replaced = True
            else:
                sessions.append(item)
        if not replaced:
            raise ValueError("The presentation session was not found")
        self._write(
            PresentationArchive(
                archive.profile_id, archive.name, tuple(sessions), archive.calibration
            ),
            self.key_store.load(profile_id),
        )

    def update_calibration(self, profile_id: str, calibration: dict[str, object]) -> None:
        archive = self.load_profile(profile_id)
        allowed = {"baseline_session_id", "baseline_confirmed"}
        if not set(calibration).issubset(allowed):
            raise ValueError("Calibration contains unsupported fields")
        baseline_id = calibration.get("baseline_session_id")
        if baseline_id is not None:
            validate_profile_id(str(baseline_id))
            if not any(item.session.session_id == baseline_id for item in archive.sessions):
                raise ValueError("The calibration baseline session was not found")
        canonical = {
            "baseline_session_id": baseline_id,
            "baseline_confirmed": bool(calibration.get("baseline_confirmed", False)),
        }
        self._write(
            PresentationArchive(
                archive.profile_id, archive.name, archive.sessions, canonical
            ),
            self.key_store.load(profile_id),
        )
