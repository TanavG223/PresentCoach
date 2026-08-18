#!/bin/zsh
set -eu

PROJECT_ROOT="${0:A:h:h}"
MODEL_DIR="${PROJECT_ROOT}/models/whisper"
MODEL_PATH="${MODEL_DIR}/ggml-base.en-q5_1.bin"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin"
MODEL_SHA256="4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f"
FACE_MODEL_PATH="${PROJECT_ROOT}/models/face_landmarker.task"
FACE_MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_SHA256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

cd "${PROJECT_ROOT}"
for required_tool in brew curl npm ollama; do
  if ! command -v "${required_tool}" >/dev/null 2>&1; then
    echo "Missing required tool: ${required_tool}" >&2
    exit 1
  fi
done
if ! command -v whisper-cli >/dev/null 2>&1; then
  brew install whisper-cpp
fi
/bin/mkdir -p "${MODEL_DIR}"
if [[ ! -f "${MODEL_PATH}" ]]; then
  curl --fail --location --retry 3 --output "${MODEL_PATH}" "${MODEL_URL}"
fi
if [[ ! -f "${FACE_MODEL_PATH}" ]]; then
  curl --fail --location --retry 3 --output "${FACE_MODEL_PATH}" "${FACE_MODEL_URL}"
fi
verify_model() {
  local model_path="$1"
  local expected_digest="$2"
  local label="$3"
  local actual_digest
  actual_digest=$(/usr/bin/shasum -a 256 "${model_path}" | /usr/bin/awk '{print $1}')
  if [[ "${actual_digest}" != "${expected_digest}" ]]; then
    echo "${label} integrity check failed" >&2
    exit 1
  fi
}
verify_model "${MODEL_PATH}" "${MODEL_SHA256}" "Whisper model"
verify_model "${FACE_MODEL_PATH}" "${FACE_MODEL_SHA256}" "MediaPipe face model"
if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  python3 -m venv "${PROJECT_ROOT}/.venv"
fi
"${PROJECT_ROOT}/.venv/bin/python" -m pip install --upgrade pip
"${PROJECT_ROOT}/.venv/bin/python" -m pip install -e '.[dev]'
ollama create presentcoach-local -f models/PresentCoach.Modelfile
npm --prefix frontend install
npm --prefix frontend run build
zsh "${PROJECT_ROOT}/scripts/install_presentcoach_app.sh"
