#!/usr/bin/env bash
# Merge the recording with the narration into assets/demo.mp4 and assets/demo.gif.
#
#   ./scripts/demo/make_video.sh            (after `make voice` and `make record`)
#
# The voice track is built by concatenating the per-line files from `make voice`,
# each followed by PAD seconds of real silence. record.ts held every scene for
# exactly that long, so picture and sound line up by construction — there is no
# drift to correct and no manual offset to tune.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null; then
  cat >&2 <<'MSG'
ffmpeg is not installed.

  brew install ffmpeg

Playwright ships an ffmpeg binary, but it is built with --disable-everything and
has no libx264, aac, gif encoder, drawtext or concat filter, so it cannot do this.
MSG
  exit 1
fi

PAD="${PAD:-0.6}"
TITLE_SECONDS="${TITLE_SECONDS:-2}"
GIF_SECONDS="${GIF_SECONDS:-15}"
FPS="${FPS:-30}"
W=1440
H=900
ASSETS="../../assets"
mkdir -p "$ASSETS"

[[ -d voice ]] || { echo "no voice/ — run 'make voice' first" >&2; exit 1; }
RECORDING=$(ls -t recording/*.webm 2>/dev/null | head -1 || true)
[[ -n "$RECORDING" ]] || { echo "no recording — run 'make record' first" >&2; exit 1; }
[[ -f title.png ]] || node title_card.ts

echo "recording: $RECORDING"

# ---- 1. the voice track: each line, then PAD seconds of silence ------------
# NB: no `mapfile` — macOS ships bash 3.2, where it does not exist.
LINES=()
while IFS= read -r f; do LINES+=("$f"); done < <(ls voice/*.aiff | sort)
echo "voice: ${#LINES[@]} lines + ${PAD}s pads"
args=(); filter=""; n=0
for f in "${LINES[@]}"; do
  args+=(-i "$f")
  filter+="[${n}:a]aresample=48000,apad=pad_dur=${PAD}[a${n}];"
  n=$((n + 1))
done
for ((i = 0; i < n; i++)); do filter+="[a${i}]"; done
filter+="concat=n=${n}:v=0:a=1[voice]"
ffmpeg -y -v error "${args[@]}" -filter_complex "$filter" -map "[voice]" -ac 2 -ar 48000 voice.wav
VOICE_SECS=$(ffprobe -v error -show_entries format=duration -of csv=p=0 voice.wav)
echo "voice.wav: ${VOICE_SECS}s"

# ---- 2. title card + recording, with the voice delayed past the card -------
# The card is silent, so the narration starts when the product appears.
ffmpeg -y -v error \
  -loop 1 -t "$TITLE_SECONDS" -i title.png \
  -i "$RECORDING" \
  -i voice.wav \
  -filter_complex "\
    [0:v]scale=${W}:${H},fps=${FPS},format=yuv420p,setsar=1[v0]; \
    [1:v]scale=${W}:${H},fps=${FPS},format=yuv420p,setsar=1[v1]; \
    [v0][v1]concat=n=2:v=1:a=0[v]; \
    [2:a]adelay=$((TITLE_SECONDS * 1000))|$((TITLE_SECONDS * 1000))[a]" \
  -map "[v]" -map "[a]" \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart \
  -c:a aac -b:a 160k \
  "$ASSETS/demo.mp4"

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$ASSETS/demo.mp4")
printf "assets/demo.mp4  %.1fs  %s\n" "$DUR" "$(du -h "$ASSETS/demo.mp4" | cut -f1)"
awk -v d="$DUR" 'BEGIN{ if (d >= 120) { print "ERROR: over the 2-minute limit"; exit 1 } }'

# ---- 3. a 15s GIF of the launch slider, the one thing nothing else shows ---
GIF_START=$(awk -v s="$(node narration.ts --start 'week slider')" -v t="$TITLE_SECONDS" \
  'BEGIN{ printf "%.2f", s + t - 1 }')
echo "gif window: ${GIF_START}s +${GIF_SECONDS}s (the launch week slider)"
ffmpeg -y -v error -ss "$GIF_START" -t "$GIF_SECONDS" -i "$ASSETS/demo.mp4" \
  -filter_complex "fps=12,scale=900:-1:flags=lanczos,split[s0][s1];\
                   [s0]palettegen=max_colors=160[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
  "$ASSETS/demo.gif"
printf "assets/demo.gif  %ss  %s\n" "$GIF_SECONDS" "$(du -h "$ASSETS/demo.gif" | cut -f1)"
