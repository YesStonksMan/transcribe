#!/bin/bash
# Build Transcribe.app — production bundle with manual fix-up for mlx.
#
# py2app can't analyze `mlx` (PEP 420 namespace package, no __init__.py) so we
# build the bundle without it, then copy the mlx package tree (including its
# native dylibs) into the right place inside the bundle.
#
# Usage:
#   ./build.sh             # build + install to /Applications
#   ./build.sh --no-install # build only, leave in dist/

set -euo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
VENV="${TRANSCRIBE_VENV:-$HOME/Claude code/video-pipeline/bin/whisperx-env}"
PY="$VENV/bin/python"
SITE_PKG="$VENV/lib/python3.11/site-packages"

cd "$PROJECT"

echo "▸ Cleaning previous build artifacts…"
rm -rf build dist

echo "▸ Running py2app…"
"$PY" setup.py py2app 2>&1 | tail -5

BUNDLE="$PROJECT/dist/Transcribe.app"
BUNDLE_LIB="$BUNDLE/Contents/Resources/lib/python3.11"

if [ ! -d "$BUNDLE" ]; then
    echo "✗ Bundle not produced — check py2app output above." >&2
    exit 1
fi

echo "▸ Stripping py2app's mlx stubs from python311.zip…"
# py2app put fake mlx/__init__.pyc + core.pyc (stub loaders) inside
# python311.zip. The __init__.pyc makes mlx a REGULAR package, which
# blocks the PEP 420 namespace package aggregation from lib-dynload.
# Strip all mlx* entries from the zip so Python falls through to the
# real namespace package we install below.
PY_ZIP="$BUNDLE/Contents/Resources/lib/python311.zip"
"$PY" -c "
import zipfile, shutil
src = '$PY_ZIP'
tmp = '$PY_ZIP.tmp'
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.namelist():
        if item.startswith('mlx/') or item.startswith('mlx-') or item.startswith('mlx_'):
            continue
        zout.writestr(item, zin.read(item))
shutil.move(tmp, src)
print('stripped')
"

echo "▸ Installing real mlx + mlx_whisper + mlx_metal packages into bundle…"
DYNLOAD_MLX="$BUNDLE_LIB/lib-dynload/mlx"
rm -rf "$DYNLOAD_MLX"
mkdir -p "$BUNDLE_LIB"
# Copy the canonical mlx tree to lib-dynload/mlx (where py2app put core.so —
# Python will find the namespace package here once the zip stubs are gone).
cp -R "$SITE_PKG/mlx" "$BUNDLE_LIB/lib-dynload/"

# Sanity check
for required in \
    "$DYNLOAD_MLX/lib/libmlx.dylib" \
    "$DYNLOAD_MLX/_reprlib_fix.py" \
    "$DYNLOAD_MLX/core.cpython-311-darwin.so"; do
    if [ ! -e "$required" ]; then
        echo "✗ Missing required file: $required" >&2
        exit 1
    fi
done

echo "▸ Installing custom app icon (overrides py2app's PythonApplet default)…"
ICON_SRC="$PROJECT/icon.icns"
if [ -f "$ICON_SRC" ]; then
    # Overwrite the default icon in-place so the plist's existing reference
    # (CFBundleIconFile: PythonApplet) picks up our art.
    cp "$ICON_SRC" "$BUNDLE/Contents/Resources/PythonApplet.icns"
    # Also drop a clean icon.icns alongside for completeness and rewrite the
    # plist key to the explicit name.
    cp "$ICON_SRC" "$BUNDLE/Contents/Resources/icon.icns"
    /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile icon" "$BUNDLE/Contents/Info.plist" 2>/dev/null \
        || /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string icon" "$BUNDLE/Contents/Info.plist"
else
    echo "  (no $ICON_SRC — keeping py2app's default icon)"
fi

echo "▸ Ad-hoc signing main executable + mlx files (no hardened runtime, no library validation)…"
# We use --options=runtime=DISABLED is not a real flag; instead the simplest
# path is: sign every Mach-O ad-hoc WITHOUT --options=runtime so hardened
# runtime restrictions don't apply. The default ad-hoc signature already does
# this — `codesign -s -` produces a non-hardened signature.
# What matters: each Mach-O (binary, .so, .dylib) needs SOME signature on
# Apple Silicon, or dlopen fails. But none should be hardened.
# `set -o pipefail` is on, so we wrap every codesign in `|| true` — a single
# non-fatal signing failure (e.g. liblzma.5.dylib vendored from Homebrew that
# fails strict validation) used to abort the whole script before the install
# step ran. Signing failures on individual subcomponents are non-fatal here.
{ find "$BUNDLE" -type f \( -name "*.so" -o -name "*.dylib" \) -print0 | \
    xargs -0 -I {} codesign --force --sign - {} 2>/dev/null ; } || true
# Main executable + bundle get the local self-signed certificate shared with
# Mini-Flow, so the Full Disk Access grant (needed to read Brave/Chrome/Safari
# cookies for Instagram) survives rebuilds — an ad hoc signature changes the
# app's identity every build and macOS silently drops the grant.
KEYCHAIN="$HOME/Library/Application Support/MiniFlow/signing.keychain-db"
IDENTITY="MiniFlow Local Signing"
SIGN=(--sign -)
if [ -f "$KEYCHAIN" ]; then
    security unlock-keychain -p "" "$KEYCHAIN"
    SIGN=(--sign "$IDENTITY" --keychain "$KEYCHAIN")
else
    echo "⚠ $KEYCHAIN absent — ad hoc signature, Full Disk Access will need re-granting" >&2
fi
xattr -cr "$BUNDLE"
# The liblzma py2app vendors can't be signed ("internal error in Code Signing
# subsystem"), which made every bundle signature fail. Swap in Homebrew's copy.
LZMA="$BUNDLE/Contents/Frameworks/liblzma.5.dylib"
if [ -f "$LZMA" ] && [ -f /opt/homebrew/opt/xz/lib/liblzma.5.dylib ]; then
    cp -f /opt/homebrew/opt/xz/lib/liblzma.5.dylib "$LZMA"
    chmod u+w "$LZMA"
    install_name_tool -id @executable_path/../Frameworks/liblzma.5.dylib "$LZMA" 2>/dev/null
    codesign --force --sign - "$LZMA"
fi
codesign --force "${SIGN[@]}" "$BUNDLE/Contents/MacOS/Transcribe" 2>&1 | tail -2 || true
# Sign the bundle wrapper LAST. No --deep (its strict validation fails on
# subcomponents like liblzma.5.dylib that py2app vendored from Homebrew).
codesign --force "${SIGN[@]}" --identifier fr.adham.transcribe "$BUNDLE" 2>&1 | tail -2 || true

SIZE=$(du -sh "$BUNDLE" | cut -f1)
echo "✓ Built $BUNDLE ($SIZE)"

if [ "${1:-}" != "--no-install" ]; then
    echo "▸ Installing to /Applications…"
    # Make sure no running instance has the bundle's files open — macOS will
    # refuse to delete a running app's contents and rm -rf will silently fail.
    pkill -9 -f "Transcribe.app/Contents/MacOS/Transcribe" 2>/dev/null || true
    sleep 1
    if [ -d /Applications/Transcribe.app ]; then
        rm -rf /Applications/Transcribe.app || {
            echo "✗ Could not remove old /Applications/Transcribe.app — close any running instance and re-run" >&2
            exit 2
        }
    fi
    mv "$BUNDLE" /Applications/ || {
        echo "✗ Failed to move bundle to /Applications" >&2
        exit 2
    }
    # Refresh Launch Services so the new icon shows up in Dock/Launchpad
    /System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister \
        -f /Applications/Transcribe.app 2>/dev/null || true
    killall Dock 2>/dev/null || true
    echo "✓ Installed to /Applications/Transcribe.app"
    echo ""
    echo "Launch it from Launchpad, Spotlight, or:"
    echo "  open /Applications/Transcribe.app"
fi
