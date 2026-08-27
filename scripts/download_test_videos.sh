#!/bin/zsh
set -eu

PROJECT_ROOT="${0:A:h:h}"
MEDIA_DIR="${PROJECT_ROOT}/test_media"
/bin/mkdir -p "${MEDIA_DIR}"

verify_sha256() {
  local output="$1"
  local expected="$2"
  local actual
  actual=$(/usr/bin/shasum -a 256 "${output}" | /usr/bin/awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Test-video integrity check failed: ${output}" >&2
    echo "Expected SHA-256: ${expected}" >&2
    echo "Actual SHA-256:   ${actual}" >&2
    exit 1
  fi
}

download_and_verify() {
  local output="$1"
  local url="$2"
  local expected="$3"
  if [[ ! -f "${output}" ]]; then
    local partial="${output}.part"
    /bin/rm -f "${partial}"
    /usr/bin/curl \
      --fail \
      --location \
      --retry 5 \
      --retry-delay 2 \
      --user-agent "PresentCoach-fixture-downloader/1.0" \
      --output "${partial}" \
      "${url}"
    verify_sha256 "${partial}" "${expected}"
    /bin/mv "${partial}" "${output}"
  fi
  verify_sha256 "${output}" "${expected}"
}

normalize_dynamic_excerpt() {
  local source="$1"
  local output="$2"
  local start_seconds="$3"
  local expected="$4"

  if [[ ! -f "${output}" ]]; then
    local ffmpeg_bin
    ffmpeg_bin="${PRESENTCOACH_FFMPEG_BIN:-$(command -v ffmpeg || true)}"
    if [[ -z "${ffmpeg_bin}" ]]; then
      echo "FFmpeg is required to create ${output}." >&2
      echo "Install FFmpeg 9.0.1 or set PRESENTCOACH_FFMPEG_BIN." >&2
      exit 1
    fi
    local partial="${output}.part.mp4"
    /bin/rm -f "${partial}"
    "${ffmpeg_bin}" \
      -nostdin \
      -hide_banner \
      -loglevel error \
      -i "${source}" \
      -ss "${start_seconds}" \
      -t 60 \
      -map 0:v:0 \
      -map '0:a:0?' \
      -vf 'scale=854:480:force_original_aspect_ratio=decrease:flags=lanczos,pad=854:480:(ow-iw)/2:(oh-ih)/2:color=black,fps=30,format=yuv420p' \
      -c:v libx264 \
      -preset medium \
      -crf 23 \
      -threads 1 \
      -x264-params 'threads=1:lookahead_threads=1:sliced_threads=0' \
      -c:a aac \
      -b:a 96k \
      -ar 16000 \
      -ac 1 \
      -fflags +bitexact \
      -flags:v +bitexact \
      -flags:a +bitexact \
      -map_metadata -1 \
      -metadata creation_time=1970-01-01T00:00:00Z \
      -movflags +faststart \
      -y "${partial}"
    verify_sha256 "${partial}" "${expected}"
    /bin/mv "${partial}" "${output}"
  fi
  verify_sha256 "${output}" "${expected}"
}

download_and_verify \
  "${MEDIA_DIR}/tarun-speaking-cc0.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f9/Tarun_speaking_01.webm/Tarun_speaking_01.webm.480p.vp9.webm" \
  "338f79f313972a57d064c835ca2b7c12421a034bd6e8529c3834ce20e7c13923"

download_and_verify \
  "${MEDIA_DIR}/stephen-hawking-nasa-public-domain.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/7/7a/StephenHawking-videoselection-2018.webm/StephenHawking-videoselection-2018.webm.480p.vp9.webm" \
  "f342090520d248594a6819d7bee45960eeab9efac9c3551a0c73c76511cb5091"

download_and_verify \
  "${MEDIA_DIR}/weekly-address-public-domain-full.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/f/fb/2015-11-21_President_Obama%27s_Weekly_Address.webm/2015-11-21_President_Obama%27s_Weekly_Address.webm.480p.vp9.webm" \
  "17127f6dcfbc7e0913d42d75a01f4bba37a36c205047ffccd661bad5674e5ba2"

# Additional low-resolution external robustness clips. These are deliberately
# downloaded rather than committed. The 240p derivatives keep a full run
# bounded while exercising compression, distant faces, edits, multiple people,
# portrait framing, and off-axis presentation behavior.
download_and_verify \
  "${MEDIA_DIR}/elsie-kanza-voa-240p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/5/56/Elsie_Kanza_Director%2C_Head_of_Africa_on_Grow_Africa.webm/Elsie_Kanza_Director%2C_Head_of_Africa_on_Grow_Africa.webm.240p.vp9.webm" \
  "1c989395bb9d1fbd2b317ede458d34d818781fa9f6f690daa2dba58378d44d52"

download_and_verify \
  "${MEDIA_DIR}/niosh-sudha-240p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/d/d9/Science_Speaks_-_Talking_to_Women_in_Science_%28Sudha%2C_Medical_Officer%29.webm/Science_Speaks_-_Talking_to_Women_in_Science_%28Sudha%2C_Medical_Officer%29.webm.240p.vp9.webm" \
  "aeb16967d03321455fd29569c8a4da7809c799d7b85caa7b883ff792c18aa3e8"

download_and_verify \
  "${MEDIA_DIR}/white-house-picnic-240p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/6/61/President_Trump_welcomes_members_of_the_House%2C_Senate_for_the_annual_Congressional_Picnic.webm/President_Trump_welcomes_members_of_the_House%2C_Senate_for_the_annual_Congressional_Picnic.webm.240p.vp9.webm" \
  "ef7de53bbe9f16750de49c7b029076c6fcd70d858ffe95f1b3692fff53e86aba"

download_and_verify \
  "${MEDIA_DIR}/ala-jodi-jill-240p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/b/bf/ALA_2025_Speaker_Jodi_Jill%2C_Founder_of_National_Puzzle_Day_and_Puzzle_Month_talks_about_Puzzles_in_Libraries.webm/ALA_2025_Speaker_Jodi_Jill%2C_Founder_of_National_Puzzle_Day_and_Puzzle_Month_talks_about_Puzzles_in_Libraries.webm.240p.vp9.webm" \
  "c5fb48107684fd0409e01b20c0f22a7c4358d35f4a342dbec6a4479a8ada1bb9"

download_and_verify \
  "${MEDIA_DIR}/immersion-weekend-240p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/8/88/About_Immersion_Weekend-10698293.webm/About_Immersion_Weekend-10698293.webm.240p.vp9.webm" \
  "dfa91f14ff24e3de45d91e3d14177f79c22d1b57977d292ff01e202593dc3f10"

download_and_verify \
  "${MEDIA_DIR}/wikimania-brianna-240p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/7/75/Wikimania_2008_-_Brianna_Laugher_-_Segment_of_State_of_Commons.ogv/Wikimania_2008_-_Brianna_Laugher_-_Segment_of_State_of_Commons.ogv.240p.vp9.webm" \
  "bb907e7278d2851ec182ada147ac1dcd2744aad5d5bf7d7f17038d3aa1948265"

download_and_verify \
  "${MEDIA_DIR}/wikimania-fop-240p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/e/ef/Wikimania_2023_-_Plenary_-_18_August_-_Lightning_Talk_-_A_Survey_of_Freedom_Of_Panorama_%28FOP%29_in_the_Philippines.webm/Wikimania_2023_-_Plenary_-_18_August_-_Lightning_Talk_-_A_Survey_of_Freedom_Of_Panorama_%28FOP%29_in_the_Philippines.webm.240p.vp9.webm" \
  "0e53c02e8df059ae10dc16763b173e07e8a4800292e7a59294e6b5af34fbdd6f"

download_and_verify \
  "${MEDIA_DIR}/monbiot-interview-240p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/d/dd/George_Monbiot_interview_with_The_Green_Interview.webm/George_Monbiot_interview_with_The_Green_Interview.webm.240p.vp9.webm" \
  "40936946ea6a5f05ca7af94734e08e329e3351d9c655fc7862ee02b179fdd040"

# Frozen dynamic-presentation corpus. Each complete 480p Commons derivative is
# pinned before a deterministic 60-second H.264/AAC excerpt is created. The
# excerpts are evaluation fixtures, never training or identity-recognition data.
download_and_verify \
  "${MEDIA_DIR}/dynamic-source-david-roberts-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/9d/Climate_change_is_simple_David_Roberts_at_TEDxTheEvergreenStateCollege.webm/Climate_change_is_simple_David_Roberts_at_TEDxTheEvergreenStateCollege.webm.480p.vp9.webm" \
  "a7e04dc58b663a19a6a5621fcfe73bac6bf69332b2c890f60a5ce2a473f8be7f"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-source-david-roberts-480p.webm" \
  "${MEDIA_DIR}/dynamic-david-roberts-60s.mp4" \
  60 \
  "cfb418ac4280a6c3f34083ec6ca785b2319ed6a8e309b213c0df9529507f7593"

download_and_verify \
  "${MEDIA_DIR}/dynamic-source-jack-andraka-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/3/35/Jack_Andraka_-_Detecting_pancreatic_cancer_at_15.webm/Jack_Andraka_-_Detecting_pancreatic_cancer_at_15.webm.480p.vp9.webm" \
  "a9f83c6f6cba2f701219bbab2d41b33020b900cf3c8ec4c646b129d7cce71ce0"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-source-jack-andraka-480p.webm" \
  "${MEDIA_DIR}/dynamic-jack-andraka-60s.mp4" \
  60 \
  "25c878e2ce3d8918078d2224de4346c1cf3cacaa0cc8da81f6564c16459dea68"

download_and_verify \
  "${MEDIA_DIR}/dynamic-source-richard-gill-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/5/54/TEDxSydney_-_Richard_Gill_-_The_Value_of_Music_Education.webm/TEDxSydney_-_Richard_Gill_-_The_Value_of_Music_Education.webm.480p.vp9.webm" \
  "6af9b349f243ba6bd554cfd6516f4fbcaff1be9d75f00dd305f9314ff2353f44"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-source-richard-gill-480p.webm" \
  "${MEDIA_DIR}/dynamic-richard-gill-60s.mp4" \
  120 \
  "098531cb0b7dc42299ac9c1bc60d41f9f9cfaeddce7a2cf05975dc27754ed974"

download_and_verify \
  "${MEDIA_DIR}/dynamic-source-leslie-cannold-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/e/e7/I_had_an_abortion..._Or_maybe_I_didn%27t_Leslie_Cannold_at_TEDxCanberra_2012.webm/I_had_an_abortion..._Or_maybe_I_didn%27t_Leslie_Cannold_at_TEDxCanberra_2012.webm.480p.vp9.webm" \
  "7112eb4254af85f23585a72ac0651a05701650252ccc524338546b76f912cf4e"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-source-leslie-cannold-480p.webm" \
  "${MEDIA_DIR}/dynamic-leslie-cannold-60s.mp4" \
  120 \
  "1938e1920d58561a69807b4e30a92f1647ad099f3ec778bd39fa3d84291e99a5"

download_and_verify \
  "${MEDIA_DIR}/dynamic-source-you-should-speak-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/b/b3/You_Should_Speak.webm/You_Should_Speak.webm.480p.vp9.webm" \
  "a295f5f5e13539ddc866bdc7ee543e9f1154cd77bf0151558d95203110bd4525"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-source-you-should-speak-480p.webm" \
  "${MEDIA_DIR}/dynamic-you-should-speak-60s.mp4" \
  180 \
  "277eb60f8288f21586c9fdf881af08764b5ba03d2faf1743236bc2f70452fbd8"

download_and_verify \
  "${MEDIA_DIR}/dynamic-source-constance-adams-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/e/e9/TEDxHouston_2011_-_Constance_Adams_-_Space_Architect.webm/TEDxHouston_2011_-_Constance_Adams_-_Space_Architect.webm.480p.vp9.webm" \
  "70ff79c673ef9433ec561958f6828f73e6ec9de2d767de58fb306d97c32900e6"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-source-constance-adams-480p.webm" \
  "${MEDIA_DIR}/dynamic-constance-adams-60s.mp4" \
  180 \
  "fc6ab8a665a8da7f3b617c34ce8ba10651e9a391a8dccefbccce5e25a19c0eef"

download_and_verify \
  "${MEDIA_DIR}/dynamic-source-simon-stone-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/e/eb/What_is_theatre_capable_of_Simon_Stone_at_TEDxSydney.webm/What_is_theatre_capable_of_Simon_Stone_at_TEDxSydney.webm.480p.vp9.webm" \
  "745e20b511ec79af5b930eef34cdb714892a76f69cdfd250e57c50cd88e3a49f"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-source-simon-stone-480p.webm" \
  "${MEDIA_DIR}/dynamic-simon-stone-60s.mp4" \
  120 \
  "359ab9cf4e99fd5c14802397f28c82e150a34f45b39fa2a130cf373c8e1d5f4e"

download_and_verify \
  "${MEDIA_DIR}/dynamic-source-sujiwo-tejo-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/3/3e/TEDxBandung_-_Sujiwo_Tejo_-_Math_Finding_Harmony_In_Chaos.webm/TEDxBandung_-_Sujiwo_Tejo_-_Math_Finding_Harmony_In_Chaos.webm.480p.vp9.webm" \
  "a5ec4fed9e8aa82acb5e7e0caa04db601afca48ffc50f0dbc386107098f9ae1b"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-source-sujiwo-tejo-480p.webm" \
  "${MEDIA_DIR}/dynamic-sujiwo-tejo-60s.mp4" \
  120 \
  "933a780afd4052218abbf64dfea1bc3cace3a59a0ce92c3e9b8291e41e9d2254"

download_and_verify \
  "${MEDIA_DIR}/dynamic-source-find-link-lightning-talk-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/0/0e/Find_link_-_lightning_talk.webm/Find_link_-_lightning_talk.webm.480p.vp9.webm" \
  "450935f5b42b3f90d1846737119a5e618e2d27cde8281775f0086fcf461a8fd0"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-source-find-link-lightning-talk-480p.webm" \
  "${MEDIA_DIR}/dynamic-find-link-lightning-talk-60s.mp4" \
  159 \
  "2cf2ab523951cc774317d18c7a4eb6a50217c2e310990f5291680cc3f76cf8fd"

download_and_verify \
  "${MEDIA_DIR}/dynamic-source-martha-forsyth-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/3/34/Martha_Forsyth_lightning_talk.webm/Martha_Forsyth_lightning_talk.webm.480p.vp9.webm" \
  "64e8fe901ca4f029db631573e9c9af3116f6d308b7e45debd8539d55fc95bbc3"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-source-martha-forsyth-480p.webm" \
  "${MEDIA_DIR}/dynamic-martha-forsyth-60s.mp4" \
  60 \
  "8ca9b3621209507740750d2bb15437fcf8c3fe516ce895b0b0f976ba46889e33"

# Iteration-2 holdout: five separately sourced moving presenters. These were
# selected and manually annotated before any model output was inspected.
download_and_verify \
  "${MEDIA_DIR}/dynamic-v2-source-fr4-global-trends-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/d/d7/FR4_-_Lightning_Talk_1.webm/FR4_-_Lightning_Talk_1.webm.480p.vp9.webm" \
  "10146382a797564c4ecce4f0f258445c8b4e0ff7ece3caee00ff5b9366bfb15f"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-v2-source-fr4-global-trends-480p.webm" \
  "${MEDIA_DIR}/dynamic-v2-fr4-global-trends-60s.mp4" \
  120 \
  "69c18b99d9d6013ec0de6d6fa5e44f646949efe0d3ef094920f54a4856f306d0"

download_and_verify \
  "${MEDIA_DIR}/dynamic-v2-source-mohammed-awal-240p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/0/0f/Mohammed_Awal%27s_Lightning_Talk.webm/Mohammed_Awal%27s_Lightning_Talk.webm.240p.vp9.webm" \
  "500178d0c1e3f638407bb6eabb92a6e8987a07c4113ee5a4d26592218ad6d745"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-v2-source-mohammed-awal-240p.webm" \
  "${MEDIA_DIR}/dynamic-v2-mohammed-awal-60s.mp4" \
  30 \
  "e7ffbed50d554f0e99ddbc01b41d2432093ee9e0652b5154b2c2cc21a0f5734e"

download_and_verify \
  "${MEDIA_DIR}/dynamic-v2-source-chris-anderson-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/3/3c/Lessons_worth_spreading-_Chris_Anderson_at_TEDxSydney.webm/Lessons_worth_spreading-_Chris_Anderson_at_TEDxSydney.webm.480p.vp9.webm" \
  "57f9551039bb175e95200a7db0d9da44aae98e074af109e43deaafbf86652079"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-v2-source-chris-anderson-480p.webm" \
  "${MEDIA_DIR}/dynamic-v2-chris-anderson-60s.mp4" \
  120 \
  "0d6d5ff1fb6317b6547562ca62a96b361ae8576d4f2cbce53cc00c8ab1c50147"

download_and_verify \
  "${MEDIA_DIR}/dynamic-v2-source-richard-stallman-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/d/da/Free_software%2C_free_society_Richard_Stallman_at_TEDxGeneva_2014.webm/Free_software%2C_free_society_Richard_Stallman_at_TEDxGeneva_2014.webm.480p.vp9.webm" \
  "f3fe84aa7c05196041249047745bf8f373142bdcb9f83a35cbff32f878c3f4b1"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-v2-source-richard-stallman-480p.webm" \
  "${MEDIA_DIR}/dynamic-v2-richard-stallman-60s.mp4" \
  60 \
  "0a8b73c4470219ac726a721f90bbea7bb003b707c178de1f5462594c7335657d"

download_and_verify \
  "${MEDIA_DIR}/dynamic-v2-source-wade-adams-480p.webm" \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/e/ea/TEDxHouston_2011_-_Wade_Adams_-_Nanotechnology_and_Energy.webm/TEDxHouston_2011_-_Wade_Adams_-_Nanotechnology_and_Energy.webm.480p.vp9.webm" \
  "1e13c75f3003187e5612a7d62067023492a4a3899c20dc7fbcc73bb902b1db21"
normalize_dynamic_excerpt \
  "${MEDIA_DIR}/dynamic-v2-source-wade-adams-480p.webm" \
  "${MEDIA_DIR}/dynamic-v2-wade-adams-60s.mp4" \
  0 \
  "a24121c9dbe5faba88e8e77a208bcc3b42593625d872db3a445f7e646e3091c2"

echo "Verified public test clips in ${MEDIA_DIR}"
