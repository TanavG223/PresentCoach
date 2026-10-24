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
import struct
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator, Protocol

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
MEDIA_SCHEMA = "presentcoach-media-v2"
MEDIA_MAGIC = b"PCMEDIA2"
MEDIA_HEADER_BYTES = 4
MEDIA_TAG_BYTES = 16
MAX_MEDIA_HEADER_BYTES = 4096
MAX_MEDIA_BYTES = 512 * 1024 * 1024
MEDIA_CHUNK_BYTES = 1024 * 1024
MEDIA_NONCE_PREFIX_BYTES = 8
MAX_MEDIA_CHUNKS = (MAX_MEDIA_BYTES + MEDIA_CHUNK_BYTES - 1) // MEDIA_CHUNK_BYTES
MAX_ENCRYPTED_MEDIA_BYTES = (
    len(MEDIA_MAGIC) + MEDIA_HEADER_BYTES + MAX_MEDIA_HEADER_BYTES
    + MAX_MEDIA_BYTES + MAX_MEDIA_CHUNKS * MEDIA_TAG_BYTES
)
MEDIA_MIME_TYPES = frozenset({"video/mp4", "video/quicktime", "video/webm"})
MEDIA_SOURCE_KINDS = frozenset({"upload", "recording"})
PROFILE_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class PresentationStorageError(RuntimeError):
    """Raised when encrypted presentation history cannot be trusted."""


class ProfileNotFoundError(PresentationStorageError):
    """Raised when a profile key or encrypted archive is missing."""


class SessionMediaNotFoundError(PresentationStorageError):
    """Raised when a profile session has no retained local video."""


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
    _cache: dict[str, bytes] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _cache_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

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
        with self._cache_lock:
            self._cache[safe_profile] = bytes(key)

    def load(self, profile: str) -> bytes:
        safe_profile = validate_profile_id(profile)
        with self._cache_lock:
            cached = self._cache.get(safe_profile)
        if cached is not None:
            return cached
        result = self._run(
            ["find-generic-password", "-a", safe_profile, "-s", self.service, "-w"]
        )
        if result.returncode == 44:
            raise ProfileNotFoundError("The PresentCoach profile key is missing")
        if result.returncode != 0:
            raise PresentationStorageError("The profile key could not be read from Keychain")
        key = _decode_key(result.stdout)
        with self._cache_lock:
            self._cache[safe_profile] = key
        return key

    def delete(self, profile: str) -> None:
        safe_profile = validate_profile_id(profile)
        result = self._run(
            ["delete-generic-password", "-a", safe_profile, "-s", self.service]
        )
        if result.returncode not in {0, 44}:
            raise PresentationStorageError("The profile key could not be deleted")
        with self._cache_lock:
            self._cache.pop(safe_profile, None)


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


@dataclass(frozen=True)
class SessionMedia:
    """Non-sensitive metadata for one encrypted session video."""

    profile_id: str
    session_id: str
    mime_type: str
    source: str
    plaintext_bytes: int


@dataclass
class DecryptedSessionMedia:
    """A short-lived plaintext media lease that the caller must close."""

    path: Path
    metadata: SessionMedia
    _temporary: tempfile.TemporaryDirectory[str] = field(repr=False)

    def close(self) -> None:
        self._temporary.cleanup()


@dataclass
class SessionMediaReader:
    """Authenticated random-access reader over independently encrypted chunks."""

    metadata: SessionMedia
    _handle: BinaryIO = field(repr=False)
    _key: bytes = field(repr=False)
    _header: bytes = field(repr=False)
    _nonce_prefix: bytes = field(repr=False)
    _ciphertext_offset: int = field(repr=False)
    _authenticated_chunks: int = field(default=0, init=False, repr=False)
    _encrypted_bytes_read: int = field(default=0, init=False, repr=False)

    def iter_range(self, start: int, end: int) -> Iterator[bytes]:
        """Yield plaintext bytes in ``[start, end)`` with bounded memory."""

        size = self.metadata.plaintext_bytes
        if not 0 <= start < end <= size:
            raise ValueError("The requested session video range is invalid")
        first_chunk = start // MEDIA_CHUNK_BYTES
        last_chunk = (end - 1) // MEDIA_CHUNK_BYTES
        aad = PresentationStore._media_aad(self._header)
        cipher = AESGCM(self._key)
        for chunk_index in range(first_chunk, last_chunk + 1):
            plaintext_start = chunk_index * MEDIA_CHUNK_BYTES
            plaintext_size = min(MEDIA_CHUNK_BYTES, size - plaintext_start)
            encrypted_offset = (
                self._ciphertext_offset
                + chunk_index * (MEDIA_CHUNK_BYTES + MEDIA_TAG_BYTES)
            )
            self._handle.seek(encrypted_offset)
            encrypted = self._handle.read(plaintext_size + MEDIA_TAG_BYTES)
            self._encrypted_bytes_read += len(encrypted)
            if len(encrypted) != plaintext_size + MEDIA_TAG_BYTES:
                raise PresentationStorageError(
                    "The encrypted session video chunk is truncated"
                )
            nonce = self._nonce_prefix + struct.pack(">I", chunk_index)
            try:
                plaintext = cipher.decrypt(
                    nonce,
                    encrypted,
                    aad + struct.pack(">I", chunk_index),
                )
            except InvalidTag as error:
                raise PresentationStorageError(
                    "The encrypted session video failed authentication"
                ) from error
            self._authenticated_chunks += 1
            local_start = max(start, plaintext_start) - plaintext_start
            local_end = min(end, plaintext_start + plaintext_size) - plaintext_start
            yield plaintext[local_start:local_end]

    def authenticate_range(self, start: int, end: int) -> None:
        """Authenticate all requested chunks before an HTTP response starts.

        Plaintext is discarded one chunk at a time. This lets the server return
        a clean error for corruption in a later requested chunk without a
        plaintext cache or authentication of unrelated media.
        """

        for _ in self.iter_range(start, end):
            pass

    def stats(self) -> dict[str, int]:
        return {
            "authenticated_chunks": self._authenticated_chunks,
            "encrypted_bytes_read": self._encrypted_bytes_read,
            "chunk_bytes": MEDIA_CHUNK_BYTES,
        }

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


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
    _mutation_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
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

    def _media_path(self, profile_id: str, session_id: str) -> Path:
        return self.data_dir / (
            f"{validate_profile_id(profile_id)}-{validate_profile_id(session_id)}"
            ".presentcoach-media"
        )

    @staticmethod
    def _media_header(
        *,
        profile_id: str,
        session_id: str,
        mime_type: str,
        source: str,
        plaintext_bytes: int,
        nonce_prefix: bytes,
    ) -> bytes:
        if mime_type not in MEDIA_MIME_TYPES:
            raise ValueError("The session video MIME type is unsupported")
        if source not in MEDIA_SOURCE_KINDS:
            raise ValueError("The session video source is invalid")
        if not 1 <= plaintext_bytes <= MAX_MEDIA_BYTES:
            raise ValueError("Session videos must be between 1 byte and 512 MB")
        if len(nonce_prefix) != MEDIA_NONCE_PREFIX_BYTES:
            raise ValueError("The session video nonce prefix is invalid")
        document = {
            "schema": MEDIA_SCHEMA,
            "profile_id": validate_profile_id(profile_id),
            "session_id": validate_profile_id(session_id),
            "mime_type": mime_type,
            "source": source,
            "plaintext_bytes": plaintext_bytes,
            "chunk_bytes": MEDIA_CHUNK_BYTES,
            "nonce_prefix": nonce_prefix.hex(),
        }
        header = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        if len(header) > MAX_MEDIA_HEADER_BYTES:
            raise ValueError("The session video metadata is too large")
        return header

    @staticmethod
    def _media_aad(header: bytes) -> bytes:
        return MEDIA_MAGIC + struct.pack(">I", len(header)) + header

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

    def _require_session(self, profile_id: str, session_id: str) -> PresentationArchive:
        safe_session_id = validate_profile_id(session_id)
        archive = self.load_profile(profile_id)
        if not any(
            item.session.session_id == safe_session_id for item in archive.sessions
        ):
            raise SessionMediaNotFoundError("The presentation session was not found")
        return archive

    def _read_media_header(
        self,
        handle,
        *,
        file_size: int,
        profile_id: str,
        session_id: str,
    ) -> tuple[SessionMedia, bytes, bytes, int]:
        prefix = handle.read(len(MEDIA_MAGIC) + MEDIA_HEADER_BYTES)
        if len(prefix) != len(MEDIA_MAGIC) + MEDIA_HEADER_BYTES:
            raise PresentationStorageError("The encrypted session video is truncated")
        if not secrets.compare_digest(prefix[: len(MEDIA_MAGIC)], MEDIA_MAGIC):
            raise PresentationStorageError("The encrypted session video format is invalid")
        header_size = struct.unpack(">I", prefix[len(MEDIA_MAGIC) :])[0]
        if not 1 <= header_size <= MAX_MEDIA_HEADER_BYTES:
            raise PresentationStorageError("The encrypted session video header is invalid")
        header = handle.read(header_size)
        if len(header) != header_size:
            raise PresentationStorageError("The encrypted session video header is truncated")
        try:
            document = json.loads(header.decode("utf-8"))
            if not isinstance(document, dict):
                raise ValueError("Media header must be an object")
            nonce_prefix = bytes.fromhex(str(document.get("nonce_prefix", "")))
            if int(document.get("chunk_bytes", 0)) != MEDIA_CHUNK_BYTES:
                raise ValueError("Media chunk size is invalid")
            media = SessionMedia(
                profile_id=validate_profile_id(str(document.get("profile_id", ""))),
                session_id=validate_profile_id(str(document.get("session_id", ""))),
                mime_type=str(document.get("mime_type", "")),
                source=str(document.get("source", "")),
                plaintext_bytes=int(document.get("plaintext_bytes", 0)),
            )
            canonical = self._media_header(
                profile_id=media.profile_id,
                session_id=media.session_id,
                mime_type=media.mime_type,
                source=media.source,
                plaintext_bytes=media.plaintext_bytes,
                nonce_prefix=nonce_prefix,
            )
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise PresentationStorageError(
                "The encrypted session video metadata is invalid"
            ) from error
        if (
            document.get("schema") != MEDIA_SCHEMA
            or not secrets.compare_digest(header, canonical)
            or not secrets.compare_digest(media.profile_id, validate_profile_id(profile_id))
            or not secrets.compare_digest(media.session_id, validate_profile_id(session_id))
        ):
            raise PresentationStorageError(
                "The encrypted session video identity is invalid"
            )
        expected_size = (
            len(MEDIA_MAGIC) + MEDIA_HEADER_BYTES + header_size
            + media.plaintext_bytes
            + (
                (media.plaintext_bytes + MEDIA_CHUNK_BYTES - 1)
                // MEDIA_CHUNK_BYTES
            ) * MEDIA_TAG_BYTES
        )
        if file_size != expected_size:
            raise PresentationStorageError("The encrypted session video size is invalid")
        return media, header, nonce_prefix, len(prefix) + header_size

    def session_media(self, profile_id: str, session_id: str) -> SessionMedia:
        """Return validated metadata for media owned by an existing session."""

        safe_profile = validate_profile_id(profile_id)
        safe_session = validate_profile_id(session_id)
        self._require_session(safe_profile, safe_session)
        path = self._media_path(safe_profile, safe_session)
        flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as error:
            raise SessionMediaNotFoundError(
                "No local video is retained for this presentation session"
            ) from error
        except OSError as error:
            raise PresentationStorageError(
                "The encrypted session video could not be opened safely"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size <= 0
                or metadata.st_size > MAX_ENCRYPTED_MEDIA_BYTES
            ):
                raise PresentationStorageError("The encrypted session video file is invalid")
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                descriptor = -1
                media, _header, _nonce, _offset = self._read_media_header(
                    handle,
                    file_size=metadata.st_size,
                    profile_id=safe_profile,
                    session_id=safe_session,
                )
                return media
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def put_session_media(
        self,
        profile_id: str,
        session_id: str,
        source_path: Path,
        *,
        mime_type: str,
        source: str,
    ) -> SessionMedia:
        """Encrypt one video at rest and bind it to a stored profile session."""

        safe_profile = validate_profile_id(profile_id)
        safe_session = validate_profile_id(session_id)
        self._require_session(safe_profile, safe_session)
        self._directory()
        flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
        try:
            source_descriptor = os.open(source_path, flags)
        except (FileNotFoundError, OSError) as error:
            raise PresentationStorageError("The session video could not be opened") from error
        temporary_path: Path | None = None
        destination_descriptor = -1
        try:
            source_metadata = os.fstat(source_descriptor)
            if (
                not stat.S_ISREG(source_metadata.st_mode)
                or not 1 <= source_metadata.st_size <= MAX_MEDIA_BYTES
            ):
                raise PresentationStorageError(
                    "Session videos must be regular files between 1 byte and 512 MB"
                )
            nonce_prefix = secrets.token_bytes(MEDIA_NONCE_PREFIX_BYTES)
            header = self._media_header(
                profile_id=safe_profile,
                session_id=safe_session,
                mime_type=mime_type,
                source=source,
                plaintext_bytes=source_metadata.st_size,
                nonce_prefix=nonce_prefix,
            )
            aad = self._media_aad(header)
            cipher = AESGCM(self.key_store.load(safe_profile))
            destination_descriptor, temporary = tempfile.mkstemp(
                prefix=".presentcoach-media-", dir=self.data_dir
            )
            temporary_path = Path(temporary)
            os.fchmod(destination_descriptor, 0o600)
            with (
                os.fdopen(source_descriptor, "rb", closefd=True) as source_handle,
                os.fdopen(destination_descriptor, "wb", closefd=True) as destination_handle,
            ):
                source_descriptor = -1
                destination_descriptor = -1
                destination_handle.write(aad)
                remaining = source_metadata.st_size
                chunk_index = 0
                while remaining:
                    chunk = source_handle.read(min(MEDIA_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise PresentationStorageError(
                            "The session video changed while it was being encrypted"
                        )
                    remaining -= len(chunk)
                    nonce = nonce_prefix + struct.pack(">I", chunk_index)
                    destination_handle.write(cipher.encrypt(
                        nonce,
                        chunk,
                        aad + struct.pack(">I", chunk_index),
                    ))
                    chunk_index += 1
                if source_handle.read(1):
                    raise PresentationStorageError(
                        "The session video changed while it was being encrypted"
                    )
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            os.replace(temporary_path, self._media_path(safe_profile, safe_session))
            temporary_path = None
            directory_descriptor = os.open(self.data_dir, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            return SessionMedia(
                safe_profile, safe_session, mime_type, source, source_metadata.st_size
            )
        except (OSError, ValueError) as error:
            if isinstance(error, PresentationStorageError):
                raise
            raise PresentationStorageError(
                "The session video could not be encrypted safely"
            ) from error
        finally:
            if source_descriptor >= 0:
                os.close(source_descriptor)
            if destination_descriptor >= 0:
                os.close(destination_descriptor)
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    def open_session_media(
        self, profile_id: str, session_id: str
    ) -> SessionMediaReader:
        """Open a bounded authenticated random-access reader for one video."""

        safe_profile = validate_profile_id(profile_id)
        safe_session = validate_profile_id(session_id)
        self._require_session(safe_profile, safe_session)
        path = self._media_path(safe_profile, safe_session)
        flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as error:
            raise SessionMediaNotFoundError(
                "No local video is retained for this presentation session"
            ) from error
        except OSError as error:
            raise PresentationStorageError(
                "The encrypted session video could not be opened safely"
            ) from error
        try:
            encrypted_metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(encrypted_metadata.st_mode)
                or encrypted_metadata.st_size <= 0
                or encrypted_metadata.st_size > MAX_ENCRYPTED_MEDIA_BYTES
            ):
                raise PresentationStorageError("The encrypted session video file is invalid")
            handle = os.fdopen(descriptor, "rb", closefd=True)
            descriptor = -1
            try:
                media, header, nonce_prefix, ciphertext_offset = self._read_media_header(
                    handle,
                    file_size=encrypted_metadata.st_size,
                    profile_id=safe_profile,
                    session_id=safe_session,
                )
                return SessionMediaReader(
                    metadata=media,
                    _handle=handle,
                    _key=self.key_store.load(safe_profile),
                    _header=header,
                    _nonce_prefix=nonce_prefix,
                    _ciphertext_offset=ciphertext_offset,
                )
            except Exception:
                handle.close()
                raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def decrypt_session_media(
        self, profile_id: str, session_id: str
    ) -> DecryptedSessionMedia:
        """Decrypt a complete video for explicit non-streaming callers."""

        reader = self.open_session_media(profile_id, session_id)
        temporary = tempfile.TemporaryDirectory(prefix="presentcoach-playback-")
        temporary_path = Path(temporary.name)
        os.chmod(temporary_path, 0o700)
        suffix = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/webm": ".webm",
        }[reader.metadata.mime_type]
        output_descriptor = -1
        try:
            output_descriptor, output_name = tempfile.mkstemp(
                prefix="session-", suffix=suffix, dir=temporary_path
            )
            output_path = Path(output_name)
            os.fchmod(output_descriptor, 0o600)
            with os.fdopen(output_descriptor, "wb", closefd=True) as output_handle:
                output_descriptor = -1
                for plaintext in reader.iter_range(
                    0, reader.metadata.plaintext_bytes
                ):
                    output_handle.write(plaintext)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            return DecryptedSessionMedia(output_path, reader.metadata, temporary)
        except (OSError, ValueError, PresentationStorageError):
            temporary.cleanup()
            raise
        finally:
            reader.close()
            if output_descriptor >= 0:
                os.close(output_descriptor)

    def delete_session_media(self, profile_id: str, session_id: str) -> None:
        """Remove a single encrypted media blob; missing media is idempotent."""

        safe_profile = validate_profile_id(profile_id)
        safe_session = validate_profile_id(session_id)
        self._require_session(safe_profile, safe_session)
        path = self._media_path(safe_profile, safe_session)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as error:
            raise PresentationStorageError(
                "The encrypted session video could not be removed safely"
            ) from error
        directory_descriptor = os.open(self.data_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

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
        with self._mutation_lock:
            archive = self.load_profile(profile_id)
            if any(item.session.session_id == stored.session.session_id for item in archive.sessions):
                raise ValueError("This presentation session is already stored")
            updated = PresentationArchive(
                archive.profile_id, archive.name, archive.sessions + (stored,), archive.calibration
            )
            self._write(updated, self.key_store.load(profile_id))

    def replace_feedback(self, profile_id: str, session_id: str, feedback: dict[str, object]) -> None:
        with self._mutation_lock:
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
        with self._mutation_lock:
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
