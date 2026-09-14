# Convenience targets. Activate the virtualenv first, or use PY=.venv/bin/python
PY ?= python
CONFIG ?= configs/nio_full.yaml
CKPT ?= outputs/checkpoints/nio_full_cnn_vit_monotone_best.pt

.PHONY: help setup data train evaluate predict figures dashboard test clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s $$'\t'

setup:          ## create a virtualenv and install dependencies
	python -m venv .venv && .venv/bin/pip install -q -U pip && .venv/bin/pip install -r requirements.txt

data:           ## build the analysis-ready dataset (OSSE twin, ~5 min)
	$(PY) -m oceanembed.cli build -c configs/default.yaml

quickdata:      ## one-year dataset for a fast demo
	$(PY) -m oceanembed.cli build -c configs/quick.yaml

train:          ## train the reconstruction model
	$(PY) -m oceanembed.cli train -c $(CONFIG)

evaluate:       ## skill against all baselines and the withheld in-situ profiles
	$(PY) -m oceanembed.cli evaluate -c $(CONFIG) --checkpoint $(CKPT)

predict:        ## write the daily 3-D NetCDF product for the test period
	$(PY) -m oceanembed.cli predict -c $(CONFIG) $(CKPT)

figures:        ## regenerate the proof-of-concept figures
	$(PY) -m oceanembed.cli figures -c $(CONFIG) $(CKPT) --results outputs/reports/evaluation.json

dashboard:      ## launch the interactive demo
	.venv/bin/streamlit run app/dashboard.py -- --ckpt $(CKPT)

test:           ## run the test suite
	$(PY) -m pytest -q

clean:          ## remove generated data and outputs (keeps the coastline cache)
	rm -rf data/processed* data/interim* outputs/checkpoints outputs/figures outputs/predictions
