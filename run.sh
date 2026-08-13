#!/usr/bin/env bash
# TrialSense — one-command launch.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment…"
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

if [ ! -f models/ddi_model.pkl ]; then
  echo "Training models (about 2.5 minutes, one time only)…"
  .venv/bin/python train.py
fi

echo "Starting TrialSense…"
exec .venv/bin/streamlit run app.py
