# Shared settings for experiments/k_*.sh (sourced, not executed). Plain bash, same convention as launch.sh.
#   INFLUENCE   influence config listing the arms        (default configs/influence_k.yaml)
#   SOURCES     arms to run, keys of that config          (default: all, i.e. "k0 k2 k10")
#   GPU         CUDA device id                            (default 0)
#   MAX_PARALLEL_RUNS  vmapped runs per batch, 0 = all 30 (default 0)
#   FORCE=1     redo evaluation / analysis stages even if their done-marker exists
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
INFLUENCE=${INFLUENCE:-configs/influence_k.yaml}
PY=${PY:-python}
kcfg() { $PY experiments/kcfg.py "$INFLUENCE" "$@"; }
SOURCES=${SOURCES:-$(kcfg sources)}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # lets several JAX processes share one GPU
MAX_PARALLEL_RUNS=${MAX_PARALLEL_RUNS:-0}
FORCE=${FORCE:-0}
first=${SOURCES%% *}
DATA_ROOT=$(kcfg "$first" data_root)
LOGS="$DATA_ROOT/logs"
mkdir -p "$LOGS"
stamp() { date +%Y-%m-%d_%H-%M-%S; }
say() { echo "[$(date '+%F %T')] $*"; }
