#!/usr/bin/env bash
# Reproduce the whole proof of concept from an empty checkout.
#
#   ./scripts/run_poc.sh              # full 3-year run, ~1.5 h on 8 CPU threads
#   ./scripts/run_poc.sh quick        # 1-year run, ~10 min
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
MODE=${1:-full}

if [ "$MODE" = "quick" ]; then
  CONFIG=configs/quick.yaml
  DATA=data/processed_quick
  EXTRA="-o paths.processed=$DATA -o paths.interim=data/interim_quick"
else
  CONFIG=configs/nio_full.yaml
  DATA=data/processed
  EXTRA=""
fi

echo "== 1/6 dependencies =="
[ -x "$PY" ] || { python -m venv .venv && .venv/bin/pip install -q -U pip \
  && .venv/bin/pip install -q -r requirements.txt; }

echo "== 2/6 dataset =="
[ -f "$DATA/manifest.json" ] || $PY -m oceanembed.cli build -c "$CONFIG" $EXTRA

echo "== 3/6 train =="
$PY -m oceanembed.cli train -c "$CONFIG" -d "$DATA"
CKPT=$(ls -t outputs/checkpoints/*_best.pt | head -1)
echo "checkpoint: $CKPT"

echo "== 4/6 evaluate =="
$PY -m oceanembed.cli evaluate -c "$CONFIG" -d "$DATA" --checkpoint "$CKPT"

echo "== 5/6 figures and product =="
$PY -m oceanembed.cli figures -c "$CONFIG" -d "$DATA" "$CKPT" \
    --results outputs/reports/evaluation.json
$PY -m oceanembed.cli predict -c "$CONFIG" -d "$DATA" "$CKPT"

echo "== 6/6 results document =="
$PY scripts/make_results.py

echo
echo "Done."
echo "  report   : outputs/reports/evaluation.md"
echo "  summary  : docs/RESULTS.md"
echo "  figures  : outputs/figures/"
echo "  product  : outputs/predictions/"
echo "  dashboard: .venv/bin/streamlit run app/dashboard.py -- --ckpt $CKPT"
