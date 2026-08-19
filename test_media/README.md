# Licensed test videos

The binary clips in this directory are downloaded locally and excluded from
Git. Run `zsh scripts/download_test_videos.sh` to reproduce the exact inputs.

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

These clips test measurement behavior only. They are not training data, and no
claim should be made about the speakers themselves.
