#!/bin/zsh
set -eu

PROJECT_ROOT="${0:A:h:h}"
MEDIA_DIR="${PROJECT_ROOT}/test_media"
/bin/mkdir -p "${MEDIA_DIR}"

download_and_verify() {
  local output="$1"
  local url="$2"
  local expected="$3"
  if [[ ! -f "${output}" ]]; then
    /usr/bin/curl --fail --location --retry 3 --output "${output}" "${url}"
  fi
  local actual
  actual=$(/usr/bin/shasum -a 256 "${output}" | /usr/bin/awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Test-video integrity check failed: ${output}" >&2
    exit 1
  fi
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

echo "Verified public test clips in ${MEDIA_DIR}"
