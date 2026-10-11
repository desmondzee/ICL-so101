#!/usr/bin/env bash
# Periodically upload durable checkpoints, then sync the complete finished run.
set -euo pipefail

REPO=/root/ICL-so101
RUN_ID="${1:?RUN_ID required}"
INTERVAL_SECONDS="${2:-300}"
BUCKET="${3:-nikgeo/icl-so101-weights}"
RUN_DIR="$REPO/outputs/zero_wam/sim_training/$RUN_ID"
DEST="hf://buckets/$BUCKET/runs/$RUN_ID"
STATE="/tmp/zero_wam_bucket_sync_${RUN_ID}.steps"

[[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ && "$INTERVAL_SECONDS" -gt 0 ]] || {
    echo "INTERVAL_SECONDS must be a positive integer" >&2
    exit 2
}
[[ -d "$RUN_DIR" ]] || {
    echo "missing run directory: $RUN_DIR" >&2
    exit 1
}

cd "$REPO"
set -a
source "$REPO/.env"
set +a
touch "$STATE"

while true; do
    for checkpoint in "$RUN_DIR"/checkpoints/checkpoint_step_*; do
        [[ -d "$checkpoint" ]] || continue
        step="${checkpoint##*_}"
        [[ -f "$checkpoint/training_state.pt" \
            && -f "$checkpoint/episode_exposure.json" \
            && -f "$RUN_DIR/validation/step_$(printf '%04d' "$step").json" ]] \
            || continue
        grep -qx "$step" "$STATE" && continue
        hf buckets sync "$checkpoint" "$DEST/checkpoints/checkpoint_step_$step" \
            --no-delete --exclude "*.partial" --format agent
        echo "$step" >> "$STATE"
        echo "$(date -u +%FT%TZ) synced durable checkpoint step $step"
    done

    if [[ -f "$RUN_DIR/completed.json" ]]; then
        hf buckets sync "$RUN_DIR" "$DEST" --no-delete \
            --exclude "*.partial" --format agent
        echo "$(date -u +%FT%TZ) synced completed run $RUN_ID"
        exit 0
    fi
    if [[ -f "$RUN_DIR/failed.json" ]]; then
        hf buckets sync "$RUN_DIR" "$DEST" --no-delete \
            --exclude "*.partial" --format agent
        echo "$(date -u +%FT%TZ) synced failed run metadata $RUN_ID"
        exit 1
    fi
    sleep "$INTERVAL_SECONDS"
done
