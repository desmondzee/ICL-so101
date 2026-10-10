#!/usr/bin/env bash
# Zero-WAM SO-101 ICL sim training launcher for the local 8xA100 node.
#
#   scripts/run_zero_wam_so101_a100.sh start RUN_ID [NUM_STEPS] [ACCUM_STEPS]
#   scripts/run_zero_wam_so101_a100.sh resume RUN_ID CHECKPOINT_DIR [NUM_STEPS] [ACCUM_STEPS]
#   scripts/run_zero_wam_so101_a100.sh status RUN_ID
#   scripts/run_zero_wam_so101_a100.sh stop RUN_ID
#
# `start` refuses to launch unless the A100 preflight artifacts exist and are
# ok, and refuses to start a second copy while a live PID is registered.
# Secrets come from the repo .env (sourced with `set -a`, never echoed).
set -euo pipefail

REPO=/root/ICL-so101
VENV="$REPO/.venv-zero-wam"
LOADER_ROOT="$REPO/data/hf_bucket_ICL-so101/zero_wam_loader_v1"
LATENT_ROOT="$REPO/data/hf_bucket_ICL-so101/zero_wam_latents_v1"
MODEL_PATH="$REPO/data/models/zero-wam-pretrain"
RUN_BASE="$REPO/outputs/zero_wam/sim_training"
PREFLIGHT="$REPO/outputs/zero_wam/a100_preflight"

usage() {
    echo "usage: $0 start RUN_ID [NUM_STEPS] [ACCUM_STEPS] | resume RUN_ID CHECKPOINT_DIR [NUM_STEPS] [ACCUM_STEPS] | status RUN_ID | stop RUN_ID" >&2
    exit 2
}

pid_file_for() { echo "$RUN_BASE/$1/run.pid"; }

live_pid() {
    local pidfile
    pidfile=$(pid_file_for "$1")
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    return 1
}

preflight_ok() {
    # both artifacts must exist and carry ok=true
    "$VENV/bin/python" - "$PREFLIGHT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for name in ("host_inventory.json", "loader_smoke.json"):
    try:
        payload = json.loads((root / name).read_text())
    except (OSError, json.JSONDecodeError):
        print(f"missing/unreadable preflight artifact: {root / name}")
        sys.exit(1)
    ok = payload.get("ok")
    if name == "host_inventory.json":
        ok = payload.get("gpu_contract", {}).get("ok", ok)
    if ok is not True:
        print(f"preflight artifact not ok: {root / name}")
        sys.exit(1)
PY
}

launch_torchrun() {
    # launch_torchrun RUN_ID RUN_DIR PIDFILE NUM_STEPS ACCUM_STEPS [EXTRA_ARGS...]
    local run_id="$1" run_dir="$2" pidfile="$3" num_steps="$4" accum_steps="$5"
    shift 5

    # shellcheck disable=SC1091
    set -a; source "$REPO/.env"; set +a
    export PYTHONPATH="$REPO:$REPO/third_party/Zero-WAM"
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF=expandable_segments:True

    nohup "$VENV/bin/torchrun" --standalone --nproc_per_node=8 \
        -m zero_wam.so101_a100_train \
        --loader-root "$LOADER_ROOT" \
        --latent-root "$LATENT_ROOT" \
        --model-path "$MODEL_PATH" \
        --run-root "$RUN_BASE" \
        --run-id "$run_id" \
        --num-steps "$num_steps" \
        --gradient-accumulation-steps "$accum_steps" \
        "$@" \
        >> "$run_dir/run.log" 2>&1 &
    echo $! > "$pidfile"
    echo "started run $run_id pid $(cat "$pidfile") num_steps=$num_steps accum_steps=$accum_steps"
    echo "log: $run_dir/run.log"
}

cmd_start() {
    local run_id="${1:?RUN_ID required}"
    local num_steps="${2:-4000}"
    local accum_steps="${3:-8}"
    local run_dir="$RUN_BASE/$run_id"
    local pidfile
    pidfile=$(pid_file_for "$run_id")

    if [[ ! "$num_steps" =~ ^[0-9]+$ || ! "$accum_steps" =~ ^[0-9]+$ \
        || "$accum_steps" -lt 1 ]]; then
        echo "NUM_STEPS and ACCUM_STEPS must be positive integers" >&2
        exit 1
    fi
    if pid=$(live_pid "$run_id"); then
        echo "run $run_id already live (pid $pid); refusing" >&2
        exit 1
    fi
    if ! preflight_ok; then
        echo "preflight gate failed; see $PREFLIGHT" >&2
        exit 1
    fi

    mkdir -p "$run_dir"
    launch_torchrun "$run_id" "$run_dir" "$pidfile" "$num_steps" "$accum_steps"
}

cmd_resume() {
    local run_id="${1:?RUN_ID required}"
    local checkpoint_dir="${2:?CHECKPOINT_DIR required}"
    local num_steps="${3:-4000}"
    local accum_steps="${4:-8}"
    local run_dir="$RUN_BASE/$run_id"
    local pidfile
    pidfile=$(pid_file_for "$run_id")

    if [[ ! "$num_steps" =~ ^[0-9]+$ || ! "$accum_steps" =~ ^[0-9]+$ \
        || "$accum_steps" -lt 1 ]]; then
        echo "NUM_STEPS and ACCUM_STEPS must be positive integers" >&2
        exit 1
    fi
    if [[ ! -f "$run_dir/run_manifest.json" \
        || ! -f "$run_dir/wandb_run_id.txt" ]]; then
        echo "resume requires run_manifest.json and wandb_run_id.txt under $run_dir" >&2
        exit 1
    fi
    if [[ ! -d "$checkpoint_dir" \
        || ! -f "$checkpoint_dir/transformer/config.json" \
        || ! -f "$checkpoint_dir/training_state.pt" ]]; then
        echo "checkpoint dir must contain transformer/config.json and training_state.pt: $checkpoint_dir" >&2
        exit 1
    fi
    if pid=$(live_pid "$run_id"); then
        echo "run $run_id already live (pid $pid); refusing" >&2
        exit 1
    fi
    if ! preflight_ok; then
        echo "preflight gate failed; see $PREFLIGHT" >&2
        exit 1
    fi

    # archive prior terminal sentinels so the resumed run's state is unambiguous
    if [[ -f "$run_dir/completed.json" ]]; then
        local final_step dest
        final_step=$("$VENV/bin/python" -c \
            "import json,sys; print(json.load(open(sys.argv[1]))['final_step'])" \
            "$run_dir/completed.json")
        dest="$run_dir/completed.step_${final_step}.json"
        if [[ -e "$dest" ]]; then
            echo "refusing to overwrite archive $dest" >&2
            exit 1
        fi
        mv "$run_dir/completed.json" "$dest"
        echo "archived prior completed.json -> $dest"
    fi
    if [[ -f "$run_dir/failed.json" ]]; then
        mv "$run_dir/failed.json" \
            "$run_dir/failed.resume_$(date -u +%Y%m%dT%H%M%SZ).json"
        echo "archived prior failed.json"
    fi

    launch_torchrun "$run_id" "$run_dir" "$pidfile" "$num_steps" \
        "$accum_steps" --resume-from "$checkpoint_dir"
}

cmd_status() {
    local run_id="${1:?RUN_ID required}"
    local run_dir="$RUN_BASE/$run_id"
    if pid=$(live_pid "$run_id"); then
        echo "run $run_id: LIVE pid $pid"
    else
        echo "run $run_id: not running"
    fi
    [[ -f "$run_dir/completed.json" ]] && echo "completed: yes"
    [[ -f "$run_dir/failed.json" ]] && echo "failed: yes"
    if compgen -G "$run_dir/checkpoints/checkpoint_step_*" > /dev/null; then
        ls -d "$run_dir"/checkpoints/checkpoint_step_* | sort -t_ -k3 -n | tail -3
    fi
    [[ -f "$run_dir/run.log" ]] && tail -n 5 "$run_dir/run.log"
    return 0
}

cmd_stop() {
    local run_id="${1:?RUN_ID required}"
    if pid=$(live_pid "$run_id"); then
        kill -TERM "$pid"   # graceful termination only, never forced
        echo "sent SIGTERM to run $run_id (pid $pid)"
    else
        echo "run $run_id: no live pid" >&2
        exit 1
    fi
}

[[ $# -ge 1 ]] || usage
cmd=$1; shift
case "$cmd" in
    start)  cmd_start "$@" ;;
    resume) cmd_resume "$@" ;;
    status) cmd_status "$@" ;;
    stop)   cmd_stop "$@" ;;
    *)      usage ;;
esac
