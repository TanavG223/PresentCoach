#!/bin/zsh
set -eu

PROJECT_ROOT="${0:A:h:h}"
SOURCE_APP="${PROJECT_ROOT}/macos/PresentCoach.app"
TARGET_PARENT="${HOME}/Applications"
TARGET_APP="${TARGET_PARENT}/PresentCoach.app"

/bin/chmod 755 "${SOURCE_APP}/Contents/MacOS/PresentCoach"
/bin/mkdir -p "${TARGET_PARENT}"
/usr/bin/ditto "${SOURCE_APP}" "${TARGET_APP}"
/bin/mkdir -p "${TARGET_APP}/Contents/Resources"
/usr/bin/printf '%s\n' "${PROJECT_ROOT}" >"${TARGET_APP}/Contents/Resources/project-root"
/bin/chmod 600 "${TARGET_APP}/Contents/Resources/project-root"
/usr/bin/xattr -dr com.apple.quarantine "${TARGET_APP}" 2>/dev/null || true
/usr/bin/codesign --force --deep --sign - "${TARGET_APP}"
echo "Installed ${TARGET_APP}"
