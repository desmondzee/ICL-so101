#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
ZERO_WAM_ROOT="${ROOT}/third_party/Zero-WAM"
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/workspace/Robotwin}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON:-${ROBOTWIN_ROOT}/.venv/bin/python}
MODE=${ZERO_WAM_MODE:-text}
SAVE_ROOT=${SAVE_ROOT:-${ROOT}/outputs/zero_wam/${MODE}}
# The eval client chdirs into ROBOTWIN_ROOT, so a relative SAVE_ROOT would
# write results there instead of here.
[[ ${SAVE_ROOT} = /* ]] || SAVE_ROOT="$(pwd)/${SAVE_ROOT}"
TEST_NUM=${TEST_NUM:-1}
SEED=${SEED:-0}
TASK=${1:-place_empty_cup}

if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
    echo "RoboTwin Python not found: ${ROBOTWIN_PYTHON}" >&2
    exit 1
fi

export ROBOTWIN_ROOT ZERO_WAM_MODE="${MODE}"
export PYTHONPATH="${ZERO_WAM_ROOT}:${ROOT}:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"

cd "${ROOT}"
exec "${ROBOTWIN_PYTHON}" -m zero_wam.modal_cli run zero_wam/modal_app.py::rollout \
    --mode "${MODE}" \
    --task "${TASK}" \
    --test-num "${TEST_NUM}" \
    --seed "${SEED}" \
    --save-root "${SAVE_ROOT}"
