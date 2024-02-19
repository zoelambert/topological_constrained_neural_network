#!/usr/bin/env bash
SHELL := /bin/bash

# PROJECT PARAMETERS
PROJECT_ROOT := $(shell pwd)

# CONDA ENV
VENV_NAME = venv
CONDA_BASE := $(conda info --base)
VENV_DIR = $(PROJECT_ROOT)/$(VENV_NAME)

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh && conda activate && conda activate ./venv


.PHONY: info
info :
	@grep '^[[:alnum:]_-]*:.* ##' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.* ## "};  {printf "%-25s %s\n", $$1, $$2};'


# ======================
# VENV
# ======================
venv:  ## init env in local
	conda create --prefix ./venv python=3.9.13 --no-default-packages -y
	@$(CONDA_ACTIVATE); python -m pip install --upgrade pip
	@$(CONDA_ACTIVATE); python -m pip install -e . -r requirements.txt  --no-cache-dir

.PHONY: update
update: venv ## update env with requirements-dev.txt
	@$(CONDA_ACTIVATE); python -m pip install --upgrade pip 
	@$(CONDA_ACTIVATE); python -m pip install -e . -r requirements.txt --no-cache-dir


# ======================
# TRAIN
# ======================
.PHONY: train
train: ## train the model
	@$(CONDA_ACTIVATE); python src/topological_constrained/main.py






	