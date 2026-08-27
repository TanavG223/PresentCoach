#!/bin/zsh
set -eu

PROJECT_ROOT="${0:A:h:h}"
SOURCE_APP="${PROJECT_ROOT}/macos/PresentCoach.app"
NATIVE_SOURCE="${PROJECT_ROOT}/macos/PresentCoachNative.swift"
TARGET_PARENT="${HOME}/Applications"
TARGET_APP="${TARGET_PARENT}/PresentCoach.app"

/bin/mkdir -p "${TARGET_PARENT}"
/usr/bin/ditto "${SOURCE_APP}" "${TARGET_APP}"
/bin/mkdir -p "${TARGET_APP}/Contents/Resources"
/usr/bin/xcrun swiftc \
  -parse-as-library \
  -O \
  -target arm64-apple-macos13.0 \
  -framework AppKit \
  -framework AVFoundation \
  -framework WebKit \
  -o "${TARGET_APP}/Contents/MacOS/PresentCoach" \
  "${NATIVE_SOURCE}"
/bin/chmod 755 "${TARGET_APP}/Contents/MacOS/PresentCoach"
/usr/bin/printf '%s\n' "${PROJECT_ROOT}" >"${TARGET_APP}/Contents/Resources/project-root"
/bin/chmod 600 "${TARGET_APP}/Contents/Resources/project-root"
/usr/bin/xattr -dr com.apple.quarantine "${TARGET_APP}" 2>/dev/null || true
/usr/bin/codesign --force --deep --sign - "${TARGET_APP}"
echo "Installed ${TARGET_APP}"
