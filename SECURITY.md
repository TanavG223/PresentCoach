# Security

Please do not open a public issue for a suspected vulnerability involving local
recordings, encryption, Keychain use, request validation, or command execution.
Use GitHub's private vulnerability reporting for this repository instead.

PresentCoach intentionally binds only to `127.0.0.1`, disables CORS and browser
camera/microphone APIs, requires an authorized browser session for sensitive
reads and writes, authenticates state-changing requests with a CSRF token, and
does not accept configurable Ollama endpoints. A change that makes the service
remotely reachable or sends frames, audio, transcripts, or metrics off-device
is outside the project's privacy contract and must be clearly reviewed as a
security-sensitive design change.

At each server start, PresentCoach creates a new random launch capability and
writes its full local URL to
`$HOME/Library/Application Support/PresentCoach/access-url`. The application
support directory is `0700` and the launch file is atomically written with
`0600` permissions. A request to `/?access=...` compares the capability,
creates a separate random authorization nonce in the signed HTTP-only browser
session, and returns a `303` redirect to `/` with a `no-referrer` policy so the
capability is not retained in the visible app URL or sent on the redirect. The
base page and all profile, bootstrap, recording,
upload, transcript, metric, and retained-replay APIs fail closed without that
authorized session. PresentCoach's startup log records only the access-file
path, not the capability URL.

`/api/health` remains public on loopback so the macOS launcher can detect the
running process. Read-only `/api/test-media/...` routes also remain public on
loopback, but serve only manifest-allowlisted evaluation media after a pinned
SHA-256 check; they do not expose user data. Static frontend assets are public
and contain no session data. This capability boundary does not claim to defend
against code already running as the same macOS user that can read the `0600`
launch file, browser cookie, or Keychain-authorized process memory.

Retained session videos are bound to random profile and session identifiers and
stored as independently authenticated AES-256-GCM chunks. Playback requires a
same-origin, one-hour signed grant containing the current authorized browser
session nonce. Copying a playback URL to a fresh or independently authorized
browser session fails closed. Byte-range requests authenticate and decrypt only
the required chunks in memory; PresentCoach does not maintain a plaintext video
cache on disk. After the user authorizes one Keychain read, the profile key is
cached only for the lifetime of the local Flask process so browser range
requests do not trigger repeated Keychain prompts; restarting the app clears
that in-memory cache and invalidates the launch capability and browser session.

Analysis uses private `0700` operating-system temporary directories and
`0600` plaintext working files before the retained replay is encrypted. Normal
completion removes those files. A process or machine crash can leave an
operating-system temporary working file until the OS removes its temporary
directory; the application never treats that working file as a replay cache.

Supported version: the latest commit on `main`.
