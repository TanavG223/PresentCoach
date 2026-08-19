# PresentCoach

PresentCoach is an MIT-licensed, local-first macOS presentation coach. It
records a practice talk, overlays MediaPipe facial landmarks, measures camera
contact and movement at 15+ analyzed frames per second, transcribes speech with
native `whisper.cpp`, and asks a local Ollama model for feedback that must cite
an exact metric, number, and timestamp.

[Project site](https://tanavg223.github.io/PresentCoach/)

It is a runnable desktop application, not a hosted service. The Python service
binds only to `127.0.0.1:8765`; the interface is intended for the Codex in-app
browser. The launcher never opens Safari, Chrome, or the default browser.

## Install and run

Clone and install on an Apple-silicon Mac:

```bash
git clone https://github.com/TanavG223/PresentCoach.git
cd PresentCoach
zsh scripts/setup_presentcoach.sh
open "$HOME/Applications/PresentCoach.app"
```

Then open `http://127.0.0.1:8765/` in the Codex browser. To run directly from
source:

```bash
.venv/bin/python -m stroke_screening.presentation_server
```

Requirements are Python 3.11, Node/npm for rebuilding the React bundle,
Homebrew `whisper-cpp`, and Ollama. `scripts/setup_presentcoach.sh` downloads
and verifies the pinned MediaPipe and Whisper model artifacts, creates the
local Ollama model, and builds the app. Model weights are not stored in this
repository.

## What it measures

Vision is analyzed at 15 FPS and stored in one-second samples:

- camera-contact proxy from iris position plus head pose;
- head rotation and normalized position variation;
- mouth/brow movement variation;
- face-presence percentage and per-metric quality flags.

Audio is captured locally at 16 kHz mono. Raw audio is discarded after
transcription. Stored results include:

- full word-level transcript with timestamps;
- strict `um`/`uh` counts plus separately tracked `like`, `you know`, and `so` phrases;
- words per minute in 15-second windows;
- gaps between Whisper word timestamps longer than two seconds and total
  duration. These are transcript gaps, not waveform-verified silence.

You can also choose **Upload video** to analyze an existing MP4, MOV, M4V, or
WebM file. Imports use the same MediaPipe, Whisper, metric, encryption, and LLM
guardrails as a live recording. Files are limited to 512 MB and 30 minutes;
the temporary video copy and decoded raw audio are discarded after analysis.

The main public functions live in
`src/stroke_screening/presentation_core.py`:

1. `analyze_session(video_frames, audio)` creates a session record.
2. `compute_metrics(session)` creates aggregates and notable moments.
3. `generate_feedback(metrics, llm)` creates and verifies structured feedback.

## Transparent coaching policy

Any quality-approved recording of at least 30 seconds receives actionable,
number-and-time-backed coaching immediately. Python applies a versioned,
transparent default policy before the local LLM sees the metrics:

- 100–165 WPM is the broad speaking-pace reference;
- 0–2 measured `um`/`uh` occurrences per minute is the practice target, while
  over 4 per minute is called frequent;
- 70% camera orientation and five-second gaze breaks are explicitly labeled
  PresentCoach product heuristics, not universal scientific norms;
- two-second transcript gaps remain neutral; only longer or repeated gaps are
  review markers.

The pace and filler targets follow [Microsoft Speaker
Coach](https://support.microsoft.com/en-US/PowerPoint/suggestions-from-speaker-coach)
and [Trent University presentation
guidance](https://www.trentu.ca/academicskills/how-guides/how-present-university-and-beyond/delivering-oral-presentation-public-speaking).
The camera and event-duration bands are disclosed app defaults. Personal
comparative language additionally requires:

1. one recording of at least 30 seconds to inspect raw baseline numbers;
2. explicit confirmation of that personal reference;
3. two nearly identical recordings whose key metrics pass declared
   repeatability tolerances.

Calibration upgrades the same panel with verified personal-reference
comparisons; it is not required to receive default-policy coaching.

## LLM guardrails

- Every displayed claim cites an exact allowed metric, number, unit, and time.
- Python owns strength/improvement roles; unsupported LLM proposals are
  discarded.
- No comments on appearance, accent, voice quality, personality, or anything
  unmeasured.
- No claims about confidence, nervousness, posture, or reading from notes.
  Face landmarks measure camera/head behavior, not those hidden causes. Full
  posture feedback remains unavailable until a quality-gated body-pose model
  is added.
- Bad-quality metrics are labeled insufficient rather than guessed.
- Sessions shorter than 30 seconds are refused for feedback.
- Feedback describes the recording and never scores or grades the person.

Run the 30-case adversarial evaluation:

```bash
.venv/bin/python scripts/evaluate_presentcoach_llm.py
```

The machine-readable result is
`reports/presentcoach_llm_eval.json` and includes grounded, excellent-session,
missing-data, short-session, and appearance/accent/confidence/reading/posture
inference-bait cases. Expected and returned metric sets must match exactly.

## Reproducible public-domain video tests

Download the three pinned Wikimedia Commons test clips:

```bash
zsh scripts/download_test_videos.sh
```

The short CC0 clip exercises insufficient-face handling, the 41-second NASA
public-domain clip exercises mixed-shot handling, and a public-domain White
House weekly address provides a stable camera-facing speaker for the full
duration and transcription path. Their source pages, licenses, exact download
URLs, and SHA-256 digests are documented in `test_media/README.md`. The video
binaries are deliberately excluded from Git.

## Privacy and security

- Camera and microphone are owned by Python; Chromium `getUserMedia` is
  disabled to avoid the high-load browser camera path.
- Requests are loopback-only, same-origin, CSRF-protected, and use a restrictive
  Content Security Policy. No CORS endpoint is enabled.
- Session history is authenticated with AES-256-GCM, written atomically with
  private permissions, and keyed through macOS Keychain.
- Whisper is a pinned 57 MB local model verified by SHA-256.
- Ollama receives structured metrics only—never camera frames or audio.

The published repository excludes model weights, recordings, encrypted local
profiles, Keychain material, build environments, and unrelated device research.

## Test

```bash
npm --prefix frontend run build
.venv/bin/python -m pytest
.venv/bin/python -m pip check
```

License: MIT. See `LICENSE`.
