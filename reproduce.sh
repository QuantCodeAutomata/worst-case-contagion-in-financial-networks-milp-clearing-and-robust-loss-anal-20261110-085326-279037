#!/usr/bin/env bash
# Runs the unit tests and every experiment of the replication, in order.
set -uo pipefail

cd "$(dirname "$0")"

echo "=== Unit tests ==="
python -m pytest -q || exit 1

for exp in exp_1 exp_2 exp_3 exp_4 exp_5; do
    echo "=== Running python -m exp.${exp} ==="
    python -m "exp.${exp}" || exit 1
done

echo "=== Done. Results saved in results/. ==="
