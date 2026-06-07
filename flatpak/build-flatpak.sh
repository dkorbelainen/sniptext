#!/bin/bash
# Build and install SnipText as a Flatpak

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$SCRIPT_DIR/io.github.dkorbelainen.SnipText.json"
APP_ID="io.github.dkorbelainen.SnipText"

if [ ! -f "$MANIFEST" ]; then
    echo "Error: $MANIFEST not found."
    exit 1
fi

echo "Building SnipText Flatpak..."
flatpak-builder --user --install --force-clean build-dir "$MANIFEST"

echo "✓ SnipText installed as Flatpak"
echo "Launch with: flatpak run $APP_ID"
