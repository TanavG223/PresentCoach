# Security

Please do not open a public issue for a suspected vulnerability involving local
recordings, encryption, Keychain use, request validation, or command execution.
Use GitHub's private vulnerability reporting for this repository instead.

PresentCoach intentionally binds only to `127.0.0.1`, disables CORS and browser
camera/microphone APIs, authenticates state-changing requests with a session
token, and does not accept configurable Ollama endpoints. A change that makes
the service remotely reachable or sends frames, audio, transcripts, or metrics
off-device is outside the project's privacy contract and must be clearly
reviewed as a security-sensitive design change.

Supported version: the latest commit on `main`.
