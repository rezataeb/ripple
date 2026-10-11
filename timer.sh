#!/bin/zsh
# Ripple milestone timer for macOS.
# Usage:  ./timer.sh            (start now)
#         ./timer.sh 25         (you started 25 minutes ago)
# Stop:   kill the process, or press Ctrl+C in its terminal.
# It shows a macOS notification + a sound + a spoken message
# 5 minutes before and at each checkpoint in MILESTONES.md.

OFFSET_MIN=${1:-0}
START=$(( $(date +%s) - OFFSET_MIN * 60 ))

# minute|label|what to check
CHECKPOINTS=(
  "15|Setup|Repo + contract + web runs"
  "40|Data|Demo CSV, truth, 3 launches"
  "75|Model|posterior.json exported, parity passes"
  "105|Launch tracker|update + calibration + /launch slider"
  "130|Agent + API|Ask works. If BYOD not done: CUT it"
  "145|Pages|/method done. If /data not done: CUT it"
  "155|Deploy|Live link works. Else stop features"
  "170|Video|demo.mp4 under 2 min"
  "180|SUBMIT|Submit now"
)

notify() {
  local title="$1" msg="$2"
  osascript -e "display notification \"$msg\" with title \"$title\" sound name \"Glass\"" 2>/dev/null
  say "$title. $msg" 2>/dev/null &
  printf "[%s] %s — %s\n" "$(date +%H:%M)" "$title" "$msg" | tee -a timer.log
}

elapsed_min() { echo $(( ($(date +%s) - START) / 60 )); }

echo "Ripple timer started. T+$(elapsed_min) min. Log: timer.log"
for cp in "${CHECKPOINTS[@]}"; do
  IFS='|' read -r MIN LABEL CHECK <<< "$cp"
  for WHEN in $(( MIN - 5 )) $MIN; do
    (( WHEN < 0 )) && continue
    TARGET=$(( START + WHEN * 60 ))
    NOW=$(date +%s)
    (( TARGET <= NOW )) && continue
    sleep $(( TARGET - NOW ))
    if (( WHEN == MIN )); then
      notify "CHECKPOINT T+${MIN}: ${LABEL}" "${CHECK}. Open MILESTONES.md and decide now."
    else
      notify "5 min to ${LABEL}" "${CHECK}"
    fi
  done
done
notify "Time is up" "Submit now."
