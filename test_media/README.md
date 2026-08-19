# Licensed test videos

The 11 unique media files used by the video and face-tracking evaluations are
downloaded locally and excluded from Git. Run the pinned
[downloader](../scripts/download_test_videos.sh) to reproduce and SHA-256-check
the exact inputs:

```bash
zsh scripts/download_test_videos.sh
```

## `tarun-speaking-cc0.webm`

- Purpose: small/distant-face and short-duration abstention test.
- Source page: <https://commons.wikimedia.org/wiki/File:Tarun_speaking_01.webm>
- Author/source listed by Commons: Anirudh, Anish, and Tarun; Drupal + Wikipedia
  25th Anniversary celebrations at ICFOSS, Thiruvananthapuram.
- License: CC0 1.0 public-domain dedication.
- Local derivative: Wikimedia's 480p VP9/Opus transcode, 8.6 seconds.
- SHA-256: `338f79f313972a57d064c835ca2b7c12421a034bd6e8529c3834ce20e7c13923`

## `stephen-hawking-nasa-public-domain.webm`

- Purpose: 41-second duration, face-presence, audio, and Whisper integration test.
- Source page: <https://commons.wikimedia.org/wiki/File:StephenHawking-videoselection-2018.webm>
- Author/source listed by Commons: NASA.
- License: public domain in the United States as a work solely created by NASA.
- Local derivative: Wikimedia's 480p VP9/Opus transcode, 41.2 seconds.
- SHA-256: `f342090520d248594a6819d7bee45960eeab9efac9c3551a0c73c76511cb5091`

## `weekly-address-public-domain-full.webm`

- Purpose: stable, camera-facing speaker test with clear speech and more than
  30 seconds of usable material.
- Source page: <https://commons.wikimedia.org/wiki/File:2015-11-21_President_Obama%27s_Weekly_Address.webm>
- Author/source listed by Commons: The White House; the speaker is then-Vice
  President Joe Biden.
- License: public domain as a work of the United States federal government.
- Local derivative: Wikimedia's 480p VP9/Opus transcode, 4 minutes 41 seconds.
- SHA-256: `17127f6dcfbc7e0913d42d75a01f4bba37a36c205047ffccd661bad5674e5ba2`

For a quick full-path run, a 45-second excerpt beginning around 15 seconds is
enough; the source remains the verified file above.

## Additional Commons tracking inputs

The benchmark adds eight low-resolution presentation/interview files. The
240p derivatives keep the run bounded while covering compression, scene cuts,
portrait framing, off-axis gaze, distant speakers, slides, groups, and
occlusion. Exact derivative URLs, SHA-256 digests, excerpt boundaries, roles,
and expectations are locked in the [manifest](face_tracking_manifest.json).

| Local file | Role | Source and license | Evaluation purpose |
| --- | --- | --- | --- |
| `elsie-kanza-voa-240p.webm` | Development | [VOA interview excerpt](https://commons.wikimedia.org/wiki/File:Elsie_Kanza_Director,_Head_of_Africa_on_Grow_Africa.webm), U.S. public domain; see the source-page warning | Compressed, stable close face inside broadcast graphics |
| `niosh-sudha-240p.webm` | Holdout-designated | [NIOSH Science Speaks](<https://commons.wikimedia.org/wiki/File:Science_Speaks_-_Talking_to_Women_in_Science_(Sudha,_Medical_Officer).webm>), U.S. federal public domain | Close speaker alternating with no-face title cards |
| `white-house-picnic-240p.webm` | Development | [White House portrait video](https://commons.wikimedia.org/wiki/File:President_Trump_welcomes_members_of_the_House,_Senate_for_the_annual_Congressional_Picnic.webm), U.S. federal public domain | Distant-group abstention in portrait framing |
| `ala-jodi-jill-240p.webm` | Holdout-designated | [ALA 2025 presentation](https://commons.wikimedia.org/wiki/File:ALA_2025_Speaker_Jodi_Jill,_Founder_of_National_Puzzle_Day_and_Puzzle_Month_talks_about_Puzzles_in_Libraries.webm), CC0 1.0 | Pillarboxed distant speaker requiring quality abstention |
| `immersion-weekend-240p.webm` | Holdout-designated | [About Immersion Weekend](https://commons.wikimedia.org/wiki/File:About_Immersion_Weekend-10698293.webm), CC BY-SA 3.0, Tricia Fulks Kelley | Low-resolution motion, groups, occlusion, and no-face intervals |
| `wikimania-brianna-240p.webm` | Development | [Wikimania 2008 presentation](https://commons.wikimedia.org/wiki/File:Wikimania_2008_-_Brianna_Laugher_-_Segment_of_State_of_Commons.ogv), CC BY-SA 3.0, Cary Bass | Distant stage face, laptop glances, head turns, and gestures |
| `wikimania-fop-240p.webm` | Holdout-designated | [Wikimania 2023 lightning talk](<https://commons.wikimedia.org/wiki/File:Wikimania_2023_-_Plenary_-_18_August_-_Lightning_Talk_-_A_Survey_of_Freedom_Of_Panorama_(FOP)_in_the_Philippines.webm>), CC BY-SA 4.0, Wikimania 2023 | Slide-heavy composition with distant and tiny inset speakers |
| `monbiot-interview-240p.webm` | Development | [The Green Interview](https://commons.wikimedia.org/wiki/File:George_Monbiot_interview_with_The_Green_Interview.webm), CC BY 2.5, The Green Interview | Close subjects, glasses, off-axis gaze, inserts, and cuts |

## Frozen benchmark design and result

The [face-tracking manifest](face_tracking_manifest.json) declares 12 anonymous
evaluation cases built from the 11 files above:

- 3 established regression cases;
- 4 development cases;
- 4 holdout-designated cases;
- 1 derived two-face abstention case.

The derived case duplicates a fixed crop from the licensed weekly address in
memory. It verifies that downstream scoring abstains when multiple faces are
detected; it is not a new recording and is never written as a video.

The [machine-readable report](../reports/presentcoach_tracking_eval.json)
records **12/12 cases passed (100%)**. Each case was evaluated twice, with no
differences in the rounded deterministic aggregate metric set between runs.
Wall-clock timing is
reported separately and is not part of the exact-repeatability claim.

This is an evaluation of the existing pinned MediaPipe Face Landmarker and
PresentCoach's anonymous temporal tracking/quality logic. It is not identity
recognition, face enrollment, or deep-model retraining. The harness does not
persist face templates, identities, or landmark arrays, and no claim should be
made about the people shown.

## Licensing and reuse

The videos retain their source licenses; PresentCoach's MIT license does not
cover them. CC BY files require attribution, a license reference, and a change
notice. CC BY-SA files add share-alike requirements for modified media. CC0
files have no copyright conditions, though provenance is preserved here.
U.S. federal public-domain designations are stated for the United States and
may be treated differently elsewhere. The VOA page additionally warns that
episodic third-party elements may be de minimis, so do not assume a cropped
derivative is public domain.

Copyright status does not automatically clear privacy, publicity, personality,
trademark, or other non-copyright rights. For that reason the repository ships
only this documentation, the manifest, report, and downloader; it does not
redistribute the video binaries.
