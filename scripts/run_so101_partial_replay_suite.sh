#!/usr/bin/env bash
# Run the frozen four-arm experiment sequentially on one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/third_party/Zero-WAM${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

run_arm() {
    local frames="$1" mode="$2" output="$3"
    local -a extra=()
    if [[ "$mode" == ensemble ]]; then extra+=(--ensemble); fi
    echo "[arm-start] $(date -u +%FT%TZ) forecast=$frames mode=$mode output=$output"
    .venv-zero-wam/bin/python scripts/run_so101_partial_chunk_replay.py \
        --forecast-frames "$frames" "${extra[@]}" --output "$output" \
        > "/tmp/so101_partial_scenev2_f${frames}_${mode}.log" 2>&1
    echo "[arm-done] $(date -u +%FT%TZ) forecast=$frames mode=$mode"
}

run_arm 2 raw outputs/zero_wam/train_replay_v2b_partial8_raw_full_conditioning_3tasks_2seeds_scenev2
run_arm 2 ensemble outputs/zero_wam/train_replay_v2b_partial8_ensemble_full_conditioning_3tasks_2seeds_scenev2
run_arm 4 raw outputs/zero_wam/train_replay_v2b_partial8_forecast4_raw_full_conditioning_3tasks_2seeds_scenev2
run_arm 4 ensemble outputs/zero_wam/train_replay_v2b_partial8_forecast4_ensemble_full_conditioning_3tasks_2seeds_scenev2
echo "[suite-done] $(date -u +%FT%TZ)"
