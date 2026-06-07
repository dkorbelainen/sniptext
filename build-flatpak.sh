#!/bin/bash
# Build and install SnipText as a Flatpak

set -e

MANIFEST="io.github.dkorbelainen.SnipText.json"
APP_ID="io.github.dkorbelainen.SnipText"

if [ ! -f "$MANIFEST" ]; then
    echo "Error: $MANIFEST not found. Run from repository root."
    exit 1
fi

echo "Building SnipText Flatpak..."
flatpak-builder --user --install --force-clean build-dir "$MANIFEST"

echo "✓ SnipText installed as Flatpak"
echo "Launch with: flatpak run $APP_ID"
