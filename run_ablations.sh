#!/usr/bin/env bash
# Run every variant of every sweep file given, then build one report over all of them.
#
#   ./run_ablations.sh configs/sweeps/cbcr.yaml [configs/sweeps/loss.yaml ...] [-- key=value ...]
#
# Each variant becomes results/runs/<sweep>_<variant>; anything after `--` is passed to every
# run as config overrides (for example data.n_eval=300). The report goes to
# results/reports/<sweeps joined by +>.
# Runs are pinned off the CPU cores that are taken offline for GPU jobs; set PIN="" to disable.
set -euo pipefail

PIN=${PIN-"taskset -c 0-5,8-21,24-31"}
PY=(conda run --no-capture-output -n plate-cloak python)

sweeps=()
while [[ $# -gt 0 && $1 != "--" ]]; do sweeps+=("$1"); shift; done
[[ ${1-} == "--" ]] && shift
overrides=("$@")
if [[ ${#sweeps[@]} -eq 0 ]]; then
    echo "usage: $0 SWEEP.yaml [SWEEP.yaml ...] [-- key=value ...]" >&2
    exit 2
fi

runs=()
stems=()
for sweep in "${sweeps[@]}"; do
    stem=$(basename "$sweep" .yaml)
    stems+=("$stem")
    mapfile -t variants < <("${PY[@]}" -c \
        "import sys; from plate_cloak.config import variants; print(*variants(sys.argv[1]), sep='\n')" \
        "$sweep")
    for variant in "${variants[@]}"; do
        name="${stem}_${variant}"
        echo "=== $name"
        $PIN "${PY[@]}" -m plate_cloak run --config "$sweep" --variant "$variant" \
            --name "$name" "${overrides[@]}"
        runs+=("results/runs/$name")
    done
done

out="results/reports/$(IFS=+; echo "${stems[*]}")"
"${PY[@]}" -m plate_cloak report --runs "${runs[@]}" --out "$out"
