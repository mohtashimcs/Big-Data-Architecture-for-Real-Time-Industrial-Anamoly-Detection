#!/usr/bin/env bash
# Regenerates research_paper.pdf (IEEE two-column) and defense_cheat_sheet.pdf
# (one page) from their HTML sources using headless Chromium's built-in
# print-to-pdf. No Python/LaTeX dependency -- just a Chromium/Chrome binary.
#
# Usage:
#   ./render_pdfs.sh
#   CHROME_PATH=/usr/bin/google-chrome ./render_pdfs.sh
set -euo pipefail
cd "$(dirname "$0")"

find_chrome() {
  local candidates=(
    "${CHROME_PATH:-}"
    /opt/pw-browsers/chromium-*/chrome-linux/chrome
    "$(command -v chromium 2>/dev/null || true)"
    "$(command -v chromium-browser 2>/dev/null || true)"
    "$(command -v google-chrome 2>/dev/null || true)"
  )
  for pattern in "${candidates[@]}"; do
    [ -z "$pattern" ] && continue
    for c in $pattern; do
      if [ -x "$c" ]; then echo "$c"; return 0; fi
    done
  done
  return 1
}

CHROME="$(find_chrome)" || {
  echo "error: no Chromium/Chrome binary found." >&2
  echo "Set CHROME_PATH=/path/to/chrome, or install Chromium/Google Chrome." >&2
  exit 1
}
echo "using $CHROME"

"$CHROME" --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf=research_paper.pdf ieee_paper.html
echo "wrote research_paper.pdf"

"$CHROME" --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf=defense_cheat_sheet.pdf defense_cheat_sheet.html
echo "wrote defense_cheat_sheet.pdf"
