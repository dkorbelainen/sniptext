#!/bin/bash
# Build and install SnipText as a Flatpak

set -e

MANIFEST="org.sniptext.SnipText.json"
APP_ID="org.sniptext.SnipText"

if [ ! -f "$MANIFEST" ]; then
    echo "Error: $MANIFEST not found. Run from repository root."
    exit 1
fi

echo "Building SnipText Flatpak..."
flatpak-builder --user --install --force-clean build-dir "$MANIFEST"

echo "✓ SnipText installed as Flatpak"
echo "Launch with: flatpak run $APP_ID"
