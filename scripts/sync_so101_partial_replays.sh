#!/usr/bin/env bash
set -uo pipefail
cd /root/ICL-so101
exec 9>/tmp/so101_partial_bucket_sync.lock
flock -n 9 || exit 0
set -a
source .env
set +a
export HF_HUB_DISABLE_PROGRESS_BARS=1
runs=(train_replay_v2b_partial8_raw_full_conditioning_3tasks_2seeds train_replay_v2b_partial8_ensemble_full_conditioning_3tasks_2seeds train_replay_v2b_partial8_forecast4_raw_full_conditioning_3tasks_2seeds train_replay_v2b_partial8_forecast4_ensemble_full_conditioning_3tasks_2seeds train_replay_v2b_partial8_raw_full_conditioning_3tasks_2seeds_scenev2 train_replay_v2b_partial8_ensemble_full_conditioning_3tasks_2seeds_scenev2 train_replay_v2b_partial8_forecast4_raw_full_conditioning_3tasks_2seeds_scenev2 train_replay_v2b_partial8_forecast4_ensemble_full_conditioning_3tasks_2seeds_scenev2 train_replay_scene_audit)
base=hf://buckets/akoniti/ICL-so101/zero_wam_evaluation_results/2026-10-14
runs+=(
  train_replay_v2b_partial8_raw_full_conditioning_3tasks_2seeds_scenev2_b4
  train_replay_v2b_partial8_ensemble_full_conditioning_3tasks_2seeds_scenev2_b4
  train_replay_v2b_partial8_forecast4_raw_full_conditioning_3tasks_2seeds_scenev2_b4
  train_replay_v2b_partial8_forecast4_ensemble_full_conditioning_3tasks_2seeds_scenev2_b4
  partial_batch4_reference_history
  partial_batch4_f4_ensemble_padding
)
while true; do
  failed=0
  for run in "${runs[@]}"; do
    src="outputs/zero_wam/$run"
    [[ -d "$src" ]] || continue
    if timeout 240 /root/.local/bin/hf buckets sync "$src" "$base/$run" --no-delete --exclude '*.tmp' --exclude '*.partial' --quiet; then
      echo "$(date -u +%FT%TZ) synced $run"
    else
      failed=1; echo "$(date -u +%FT%TZ) sync failed $run; retry next pass" >&2
    fi
  done
  if ! timeout 120 /root/.local/bin/hf buckets cp docs/ZERO_WAM_PARTIAL_CHUNK_REPLAY_TRACKER.md "$base/partial_chunk_replay_plan/ZERO_WAM_PARTIAL_CHUNK_REPLAY_TRACKER.md"; then failed=1; fi
  if [[ -d docs/results/zero_wam_partial_chunk_replay ]]; then
    if ! timeout 240 /root/.local/bin/hf buckets sync docs/results/zero_wam_partial_chunk_replay "$base/partial_chunk_replay_analysis" --no-delete --quiet; then failed=1; fi
  fi
  echo "$(date -u +%FT%TZ) pass finished failed=$failed next=300s"
  sleep 300
done
