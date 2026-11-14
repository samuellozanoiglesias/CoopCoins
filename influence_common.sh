# Shared settings for run_exp*.sh (sourced, not executed).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
INFLUENCE=${INFLUENCE:-configs/influence.yaml}
cfg_get() {  # cfg_get exp1 data_root|training_id|experiment
  python -c "
from coopcoins.utils import _yaml, load_experiment
s = _yaml('$INFLUENCE')['experiments']['$1']
print(load_experiment(s['experiment'])['data_root'] if '$2' == 'data_root' else s['$2'])"
}
DATA_ROOT=$(cfg_get exp1 data_root)
mkdir -p "$DATA_ROOT/logs"
