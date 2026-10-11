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

# FFMPEG can point at any full build. The static binary from the imageio-ffmpeg
# wheel works and needs no brew:
#   FFMPEG=$(uv run --with imageio-ffmpeg python -c \
#     "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
FFMPEG="${FFMPEG:-$(command -v ffmpeg || true)}"
if [[ -z "$FFMPEG" ]] || ! "$FFMPEG" -hide_banner -encoders 2>/dev/null | grep -q " libx264 "; then
  cat >&2 <<'MSG'
No usable ffmpeg (needs libx264, aac and the gif encoder).

  brew install ffmpeg
or, with no brew:
  FFMPEG=$(uv run --with imageio-ffmpeg python -c \
    "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") \
    ./scripts/demo/make_video.sh

Playwright's bundled ffmpeg cannot stand in: it is built --disable-everything,
with no libx264, aac, gif encoder, drawtext or concat filter.
MSG
  exit 1
fi
echo "ffmpeg: $FFMPEG"

# The static build ships no ffprobe, so read the duration out of ffmpeg itself.
duration_of() {
  # `ffmpeg -i X` with no output file always exits 1, which under `pipefail`
  # would kill the script, so swallow the status and read the banner.
  { "$FFMPEG" -hide_banner -i "$1" 2>&1 || true; } \
    | awk -F'[ ,]' '/Duration:/ {split($4,t,":"); print t[1]*3600 + t[2]*60 + t[3]; exit}'
}

# OUT_NAME picks the output stem; RECORDING_DIR picks which capture to use;
# SILENT=1 burns no audio (the subtitles are already in the picture).
OUT_NAME="${OUT_NAME:-demo}"
RECORDING_DIR="${RECORDING_DIR:-recording}"
SILENT="${SILENT:-0}"
MAKE_GIF="${MAKE_GIF:-1}"
PAD="${PAD:-0.6}"
TITLE_SECONDS="${TITLE_SECONDS:-2}"
GIF_SECONDS="${GIF_SECONDS:-15}"
FPS="${FPS:-30}"
W=1440
H=900
ASSETS="../../assets"
mkdir -p "$ASSETS"

if [[ "$SILENT" != "1" ]]; then
  [[ -d voice ]] || { echo "no voice/ — run 'make voice' first" >&2; exit 1; }
fi
RECORDING=$(ls -t "$RECORDING_DIR"/*.webm 2>/dev/null | head -1 || true)
[[ -n "$RECORDING" ]] || { echo "no recording in $RECORDING_DIR — run 'make record' first" >&2; exit 1; }
[[ -f title.png ]] || node title_card.ts

echo "recording: $RECORDING"

# ---- 1. the voice track: each line, then PAD seconds of silence ------------
if [[ "$SILENT" == "1" ]]; then
  echo "silent build: subtitles are burned into the picture, no audio track"
  "$FFMPEG" -y -v error \
    -loop 1 -t "$TITLE_SECONDS" -i title.png \
    -i "$RECORDING" \
    -filter_complex "\
      [0:v]scale=${W}:${H},fps=${FPS},format=yuv420p,setsar=1[v0]; \
      [1:v]scale=${W}:${H},fps=${FPS},format=yuv420p,setsar=1[v1]; \
      [v0][v1]concat=n=2:v=1:a=0[v]" \
    -map "[v]" -an \
    -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart \
    "$ASSETS/${OUT_NAME}.mp4"
  DUR=$(duration_of "$ASSETS/${OUT_NAME}.mp4")
  printf "assets/%s.mp4  %.1fs  %s\n" "$OUT_NAME" "$DUR" "$(du -h "$ASSETS/${OUT_NAME}.mp4" | cut -f1)"
  awk -v d="$DUR" 'BEGIN{ if (d >= 120) { print "ERROR: over the 2-minute limit"; exit 1 } }'
  exit 0
fi

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
"$FFMPEG" -y -v error "${args[@]}" -filter_complex "$filter" -map "[voice]" -ac 2 -ar 48000 voice.wav
VOICE_SECS=$(duration_of voice.wav)
echo "voice.wav: ${VOICE_SECS}s"

# ---- 2. title card + recording, with the voice delayed past the card -------
# The card is silent, so the narration starts when the product appears.
"$FFMPEG" -y -v error \
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
  "$ASSETS/${OUT_NAME}.mp4"

DUR=$(duration_of "$ASSETS/${OUT_NAME}.mp4")
printf "assets/%s.mp4  %.1fs  %s\n" "$OUT_NAME" "$DUR" "$(du -h "$ASSETS/${OUT_NAME}.mp4" | cut -f1)"
awk -v d="$DUR" 'BEGIN{ if (d >= 120) { print "ERROR: over the 2-minute limit"; exit 1 } }'

# ---- 3. a 15s GIF of the launch slider, the one thing nothing else shows ---
if [[ "$MAKE_GIF" != "1" ]]; then exit 0; fi
GIF_START=$(awk -v s="$(node narration.ts --start 'week slider')" -v t="$TITLE_SECONDS" \
  'BEGIN{ printf "%.2f", s + t - 1 }')
echo "gif window: ${GIF_START}s +${GIF_SECONDS}s (the launch week slider)"
"$FFMPEG" -y -v error -ss "$GIF_START" -t "$GIF_SECONDS" -i "$ASSETS/${OUT_NAME}.mp4" \
  -filter_complex "fps=12,scale=900:-1:flags=lanczos,split[s0][s1];\
                   [s0]palettegen=max_colors=160[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
  "$ASSETS/${OUT_NAME}.gif"
printf "assets/%s.gif  %ss  %s\n" "$OUT_NAME" "$GIF_SECONDS" "$(du -h "$ASSETS/${OUT_NAME}.gif" | cut -f1)"
