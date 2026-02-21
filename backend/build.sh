#!/usr/bin/env bash
# ─── Render Build Script ──────────────────────────────────
# This script runs during Render's build phase.
# Render docs: https://render.com/docs/deploy-python

set -o errexit   # exit on error
set -o pipefail  # catch pipe failures

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Build complete ✓"
