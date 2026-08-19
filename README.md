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

PresentCoach does not open a personal browser. After it starts, read its private
per-process launch URL:

```bash
cat "$HOME/Library/Application Support/PresentCoach/access-url"
```

Paste that complete URL into the Codex in-app browser. The first request
authorizes only that browser session and redirects to the clean
`http://127.0.0.1:8765/` address, removing the secret from the visible URL.
Opening the base address directly shows a locked page and does not return
profile or session data. Do not share or bookmark the private launch URL.

To run directly from source:

```bash
.venv/bin/python -m stroke_screening.presentation_server
```

The same `access-url` file is created for a source run and removed when the
server exits normally.

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

Audio is captured locally at 16 kHz mono. The standalone raw audio buffer is
discarded after transcription and, for a live run, local muxing into the
encrypted replay. Stored results include:

- full word-level transcript with timestamps;
- strict `um`/`uh` counts plus separately tracked `like`, `you know`, and `so` phrases;
- words per minute in 15-second windows;
- gaps between Whisper word timestamps longer than two seconds and total
  duration. These are transcript gaps, not waveform-verified silence.

You can also choose **Upload video** to analyze an existing MP4, MOV, M4V, or
WebM file. Imports use the same MediaPipe, Whisper, metric, encryption, and LLM
guardrails as a live recording. Files are limited to 512 MB and 30 minutes.
The temporary import and decoded raw audio are removed after analysis; an
authenticated encrypted H.264/yuv420p MP4 playback copy is retained locally for
synchronized session review. Audio is converted to AAC only when the source has
an audio stream; silent videos remain video-only. If this bounded local
normalization fails, the measurements and coaching cues are still saved and the
session is marked as having no replay.

Imported vision frames pass through a two-thread local FFmpeg decoder capped at
15 FPS and 960 pixels on either axis before landmark inference. Phone rotation
metadata is applied before that size limit, and sources above 3840x2160-equivalent
pixel area are rejected before decoding. The retained replay is capped at 30 FPS
and 1920x1080, with each FFmpeg decoder, filter, and encoder stage limited to two
threads so high-frame-rate or high-resolution sources cannot create an unbounded
analysis workload.

## Timestamped replay review

After a live recording or upload, the Flask app presents a two-column review
workspace: the retained local video on the left and a scrollable measured-coaching
timeline on the right. Selecting a timed cue or transcript word seeks the
replay to that measured timestamp; aggregate and capture-quality observations are
labeled session-wide rather than attached to a false moment. Cues cover strict `um`/`uh` events, camera-contact
breaks, face-tracking gaps, pace spikes, long transcript gaps, and measured
mouth/brow movement. Quality failures appear as explicit insufficiency notes.

The review UI does not infer emotion, confidence, nervousness, or whether a
speaker was reading. It can report a prolonged camera-contact break or measured
facial-landmark movement, but it cannot prove the hidden reason for either.

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

## Reproducible licensed-video evaluation

The [downloader](scripts/download_test_videos.sh) fetches and verifies 11
pinned Wikimedia Commons files: the three original pipeline-test clips and
eight additional presentation/interview clips used by the tracking benchmark.

```bash
zsh scripts/download_test_videos.sh
```

The original three still exercise the complete video pipeline: a short CC0
distant-face clip, a 41-second NASA public-domain mixed-shot clip, and a
public-domain White House address with a stable camera-facing speaker. Their
existing end-to-end result remains in
[`reports/presentcoach_video_eval.json`](reports/presentcoach_video_eval.json).
All media binaries are excluded from Git; source pages, licenses, reuse notes,
and exact digests are in the [test-media documentation](test_media/README.md).

### Frozen anonymous face-tracking benchmark

Run the benchmark after the clips and pinned face model have been installed:

```bash
.venv/bin/python scripts/evaluate_face_tracking.py
```

The frozen [12-case manifest](test_media/face_tracking_manifest.json) contains
three regression cases, four development cases, four holdout-designated cases,
and one derived case. It combines the three existing clips, eight new Commons
presentations/interviews, and an in-memory two-face challenge made by
duplicating a fixed crop from the already licensed public-domain address. No
twelfth recording is downloaded or created.

The checked-in [tracking report](reports/presentcoach_tracking_eval.json)
records **12/12 cases passed (100%)**. Every case ran twice, and its
rounded deterministic aggregate tracking metrics matched exactly across both
runs. The benchmark
checks input and model digests, analyzed cadence, valid single-face frames,
contiguous dropouts, timestamped reacquisition, normalized rigid-landmark
jitter in defensibly stable segments, and multi-face abstention. Media
timestamps—not CPU speed—drive smoothing. Passing means the declared checks
held for these exact artifacts and excerpts; it is not a claim of universal
face-tracking accuracy.

This benchmark is evaluation-only. It does not perform identity recognition,
persist landmark arrays or face templates, or retrain/fine-tune the pinned
MediaPipe Face Landmarker. A real training claim would require a separate
consented dataset, locked labels and splits, a documented training procedure,
and independent held-out evaluation.

The downloaded media is not covered by PresentCoach's MIT license. Each clip
retains the license on its Commons source page: CC BY material requires credit,
CC BY-SA material also carries share-alike conditions for modified media, and
U.S. federal public-domain status can vary outside the United States. The VOA
source also carries a warning about episodic third-party elements. The project
therefore distributes only the manifest, attribution, and downloader—not the
video binaries—and does not treat a copyright license as clearance of privacy
or personality rights.

## Privacy and security

- Camera and microphone are owned by Python; Chromium `getUserMedia` is
  disabled to avoid the high-load browser camera path.
- Requests are loopback-only and use an unguessable per-process launch
  capability stored in a user-only `0600` file. The capability is exchanged
  for an HTTP-only, same-site browser session and stripped from the URL before
  the app loads. Sensitive reads and writes require that authorized session;
  writes are additionally CSRF-protected. No CORS endpoint is enabled.
- Only health status and integrity-checked, allowlisted evaluation clips are
  readable without the authorized session. They contain no profile, transcript,
  recording, or retained-session data.
- Session history is authenticated with AES-256-GCM, written atomically with
  private permissions, and keyed through macOS Keychain.
- Session videos use independently authenticated 1 MiB encrypted chunks.
  One-hour signed, same-origin playback grants are bound to the authorized
  browser-session nonce and decrypt only requested ranges in memory. A copied
  grant is rejected from another browser session, and every explicit HTTP range
  response is capped at 4 MiB—there is no plaintext playback cache or
  whole-video decrypt per seek.
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
