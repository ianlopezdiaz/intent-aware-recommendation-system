#!/usr/bin/env bash
# Execute all notebooks in place, then render and publish the Quarto site
# to the gh-pages branch. Push to main triggers this automatically via
# .github/workflows/publish.yml — run this manually only for an out-of-band
# publish (e.g. republishing without a new commit).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Executing notebooks..."
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

echo "Publishing to GitHub Pages..."
quarto publish gh-pages --no-browser --no-prompt
