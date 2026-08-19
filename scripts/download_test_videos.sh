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

echo "Verified public test clips in ${MEDIA_DIR}"
