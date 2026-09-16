#!/usr/bin/env bash
# Download BRISC 2025 from Kaggle into data/raw/.
# Requires a Kaggle API token (Kaggle -> Settings -> API -> create token).
# Alternative: download the zip from
#   https://www.kaggle.com/datasets/briscdataset/brisc2025
# in the browser and unzip it into data/raw/ yourself.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Run: pip install kaggle" >&2
  exit 1
fi

mkdir -p data/raw
kaggle datasets download -d briscdataset/brisc2025 -p data/raw --unzip

echo
echo "Downloaded. Folder layout (3 levels):"
find data/raw -maxdepth 3 -type d | sort
echo
echo "Image counts per folder:"
find data/raw -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
  | sed 's|/[^/]*$||' | sort | uniq -c
echo
echo "Next: python -m src.prepare_data"
