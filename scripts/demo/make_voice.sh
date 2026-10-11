#!/usr/bin/env bash
# Build the narration track with macOS `say` — no API key, no network.
#
# Two outputs:
#   voice.aiff      the whole track, for a quick listen
#   voice/NN.aiff   one file per spoken line, plus timings.json
#
# Per-line matters: record.ts holds each scene for exactly its line's measured
# duration, so the picture and the voice stay together by construction instead of
# by guessing. `make video` concatenates the per-line files with the pad as real
# silence, which is why the merge needs no drift correction.
#
# Note: `say -f narration.txt` directly would read the "#" comments and the
# "[scene cue]" lines out loud, so the spoken-only text is extracted first.
set -euo pipefail

cd "$(dirname "$0")"
VOICE="${VOICE:-}"                 # e.g. VOICE=Samantha; empty = the system voice
RATE="${RATE:-}"                   # e.g. RATE=180 words per minute
PAD="${PAD:-0.6}"                  # seconds of silence after each line

say_args=()
[[ -n "$VOICE" ]] && say_args+=(-v "$VOICE")
[[ -n "$RATE" ]] && say_args+=(-r "$RATE")

rm -rf voice voice.aiff narration.spoken.txt
mkdir -p voice

# one source of truth: the parser strips cues and comments
node narration.ts --spoken > narration.spoken.txt
n=$(grep -c . narration.spoken.txt)
echo "narration: $n spoken lines"

# the whole track, for listening to
say "${say_args[@]}" -o voice.aiff -f narration.spoken.txt
echo "wrote voice.aiff ($(afinfo voice.aiff | awk '/estimated duration/{printf "%.1fs", $3}'))"

# per line, measured
i=0
secs=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  f=$(printf "voice/%02d.aiff" "$i")
  say "${say_args[@]}" -o "$f" "$line"
  d=$(afinfo "$f" | awk '/estimated duration/{print $3}')
  secs+=("$d")
  printf "  %02d  %6.2fs  %s\n" "$i" "$d" "$(echo "$line" | cut -c1-58)"
  i=$((i + 1))
done < narration.spoken.txt

# timings.json drives the waits in record.ts
{
  printf '{\n "pad": %s,\n "voice": "%s",\n "seconds": [' "$PAD" "${VOICE:-system}"
  for j in "${!secs[@]}"; do
    [[ $j -gt 0 ]] && printf ", "
    printf "%s" "${secs[$j]}"
  done
  printf ']\n}\n'
} > timings.json

total=$(awk -v p="$PAD" 'BEGIN{t=0} {t+=$1} END{print t}' <<< "$(printf "%s\n" "${secs[@]}")")
echo
echo "wrote timings.json — $i lines, ${total}s of speech + $(echo "$i * $PAD" | bc)s of pads"
node narration.ts --plan | tail -3
