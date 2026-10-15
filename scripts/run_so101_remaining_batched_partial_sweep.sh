#!/usr/bin/env bash
# Continue only after the separately launched F2/raw batch cohort completes.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/third_party/Zero-WAM${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
raw=outputs/zero_wam/train_replay_v2b_partial8_raw_full_conditioning_3tasks_2seeds_scenev2_b4
raw_pid="${1:?Pass the running F2/raw Python PID}"
accept_drift="${2:-}"
.venv-zero-wam/bin/python - "$raw" "$raw_pid" "$accept_drift" <<'PY'
import json,sys,time,os
from pathlib import Path
from scripts.run_so101_batched_train_replay import COHORT_IDS
root=Path(sys.argv[1])
if (not json.loads((root/"reference_comparison.json").read_text())["pass"]
        and sys.argv[3] != "--accept-reviewed-closed-loop-drift"):
    raise SystemExit("autonomous reference comparison failed; refusing automatic continuation")
if sys.argv[3] == "--accept-reviewed-closed-loop-drift":
    print("User authorized batch-four sweep after reviewing closed-loop drift; comparison remains failed.",
          flush=True)
while True:
    complete=True
    for ep in COHORT_IDS:
        try:
            result=json.loads((root/ep/"result.json").read_text())
            complete &= bool(result.get("partial_protocol_sha256"))
        except (FileNotFoundError,json.JSONDecodeError):
            complete=False
    if complete:
        print("F2/raw cohort complete",flush=True)
        break
    try:
        os.kill(int(sys.argv[2]),0)
    except ProcessLookupError:
        raise SystemExit("F2/raw worker exited before complete cohort; refusing continuation")
    time.sleep(30)
PY
run_arm() {
    local frames="$1" mode="$2" output="$3"
    local -a extra=()
    if [[ "$mode" == ensemble ]]; then extra+=(--ensemble); fi
    echo "[arm-start] $(date -u +%FT%TZ) forecast=$frames mode=$mode"
    .venv-zero-wam/bin/python scripts/run_so101_batched_partial_replay.py \
      --forecast-frames "$frames" "${extra[@]}" --output "$output" \
      > "/tmp/so101_partial_scenev2_b4_f${frames}_${mode}.log" 2>&1
    echo "[arm-done] $(date -u +%FT%TZ) forecast=$frames mode=$mode"
}
run_arm 2 ensemble outputs/zero_wam/train_replay_v2b_partial8_ensemble_full_conditioning_3tasks_2seeds_scenev2_b4
run_arm 4 raw outputs/zero_wam/train_replay_v2b_partial8_forecast4_raw_full_conditioning_3tasks_2seeds_scenev2_b4
run_arm 4 ensemble outputs/zero_wam/train_replay_v2b_partial8_forecast4_ensemble_full_conditioning_3tasks_2seeds_scenev2_b4
echo "[suite-done] $(date -u +%FT%TZ)"
