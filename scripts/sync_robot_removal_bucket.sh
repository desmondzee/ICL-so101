#!/bin/bash
# Hourly catch-up of outputs/robot_removal onto the nikgeo bucket.
# Existing remote files are skipped. v4_last segmentation dumps stay local.
set -u

exec 9>/tmp/robot_removal_bucket_sync.lock
if ! flock -n 9; then
  echo "$(date -Is) sync already running" >> /var/log/robot_removal_bucket_sync.log
  exit 0
fi

# shellcheck disable=SC1091
source /venv/main/bin/activate
set -a
# shellcheck disable=SC1091
source /workspace/ICL-so101/.env
set +a

log=/var/log/robot_removal_bucket_sync.log
if [[ -z "${HF_TOKEN2:-}" ]]; then
  echo "$(date -Is) HF_TOKEN2 is unset" >> "$log"
  exit 1
fi
export HF_TOKEN="$HF_TOKEN2"

{
  echo "===== $(date -Is) ====="
  hf buckets sync \
    /workspace/ICL-so101/outputs/robot_removal \
    hf://buckets/nikgeo/ICL-so101/robot_removal \
    --exclude '*/seg.json' \
    --exclude '*/seg_failed.json' \
    --exclude '*/lf_*.png'
  status=$?
  echo "exit=${status}"
  exit "${status}"
} >> "$log" 2>&1
