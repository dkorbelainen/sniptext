#!/bin/bash
# Quick OCR capture script
# Use this with system hotkey settings

cd "$(dirname "$0")"
source venv/bin/activate
exec python main.py --capture-now
