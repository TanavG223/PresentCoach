# PresentCoach

PresentCoach is an MIT-licensed, local-first macOS presentation coach. It
records a practice talk, overlays MediaPipe facial landmarks, measures camera
contact and movement at 15+ analyzed frames per second, transcribes speech with
native `whisper.cpp`, and asks a local Ollama model for feedback that must cite
an exact metric, number, and timestamp.

[Project site](https://tanavg223.github.io/PresentCoach/)

It is a standalone, open-source desktop application, not a hosted service. A
native AppKit/WebKit window starts and owns the Python engine, which binds only
to `127.0.0.1:8765`. PresentCoach never opens Safari, Chrome, Codex, or the
default browser.

## Install and run

Clone and install on an Apple-silicon Mac:

```bash
git clone https://github.com/TanavG223/PresentCoach.git
cd PresentCoach
zsh scripts/setup_presentcoach.sh
open "$HOME/Applications/PresentCoach.app"
```

PresentCoach opens its own macOS window, appears in the Dock, supports normal
Quit/Hide/About commands, and shuts down its child Python process when the app
quits. The embedded WebKit session privately consumes the per-process launch
capability and strips it from the visible address before loading any profile or
session data. That capability should not be shared or bookmarked.

To run directly from source:

```bash
.venv/bin/python -m presentcoach.presentation_server
```

The same private `access-url` file is created for a source run and removed when
the server exits normally. It exists for the native shell's authorization
exchange and is not part of the ordinary user workflow.

Requirements are Python 3.11, Node/npm for rebuilding the React bundle,
the macOS Swift compiler, Homebrew `whisper-cpp`, and Ollama.
`scripts/setup_presentcoach.sh` downloads
and verifies the pinned MediaPipe and Whisper model artifacts, creates the
local Ollama model, builds the React assets, compiles the native arm64 app, and
installs it to `~/Applications/PresentCoach.app`. Model weights are not stored
in this repository. The native shell source is
[`macos/PresentCoachNative.swift`](macos/PresentCoachNative.swift).

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
`src/presentcoach/presentation_core.py`:

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
The refreshed schema-v2 report records **30/30 cases passed (100%)** against
the pinned local Ollama model and binds the model, prompt policy, response
schema, evaluator, and case-contract digests. This is a guardrail and grounding
result, not a claim that the LLM measures video by itself.

The separate [real-video API E2E report](reports/presentcoach_e2e_development.json)
records **10/10 development presentations passed** through the actual Flask
upload endpoint: bounded video decoding, MediaPipe measurements, local Whisper
timestamps, local Ollama selection, encrypted storage, signed range replay,
timestamped review cues, and bootstrap reload. Two natural-video cases needed
the documented deterministic completion path because the 4B model omitted
approved findings; Python completed only already-derived candidate IDs and
reverified every metric, role, quality flag, value, unit, and timestamp. The
model-output holdout was not reused for this API evaluation.

## Reproducible licensed-video evaluation

The [downloader](scripts/download_test_videos.sh) fetches and verifies 26
pinned Wikimedia Commons source files: three pipeline clips, eight additional
tracking inputs, and fifteen moving-speaker TEDx/lightning-talk sources. It also
recreates the fifteen deterministic 60-second H.264/AAC presentation excerpts.

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

### Moving-presenter observed-state benchmark

The stricter [moving-presenter manifest](test_media/dynamic_presentation_manifest.json)
contains fifteen real 60-second talks with speaker motion, scene cuts, slides,
wide shots, occlusion, and multiple-camera edits. Ten clips form the declared
development split and five separately sourced clips form an iteration-2
model-output holdout. Development clips run twice and must produce identical
non-runtime aggregate measurements; the holdout runs once.

The final development report passed all six frozen pooled engineering gates:
82.1% trackable-face coverage, 9.8% false-valid rate, 97.1% multi-face
abstention, 5.48 dropouts per annotated minute, 100% reacquisition, and a
1.53-second maximum reacquisition latency. The slowest clip processed at 83.5
FPS on the development machine, above the 15 FPS analysis cadence.

The one-shot holdout is intentionally preserved as a **failed** result rather
than rerun: it measured 96.3% trackable coverage and 100% reacquisition, but a
25.9% false-valid rate on slide/wide-shot intervals, 6.06 dropouts per minute,
and no annotated multi-face interval, leaving that required metric not
evaluated. No thresholds or tracker code were changed after seeing this output.
The earlier iteration-1 manifest and reports are retained under
`dynamic_presentation_manifest_iteration1.json` and
`presentcoach_dynamic_iteration1_*.json` so the failed research path is not
erased.

These are single-curator observed-state engineering checks on fixed licensed
clips—not population face-tracking accuracy, gaze accuracy, speaker identity,
or proof of model training. MediaPipe and YuNet remain pinned pretrained
anonymous geometry models; PresentCoach learns no face identity and stores no
face embedding or landmark template.

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
