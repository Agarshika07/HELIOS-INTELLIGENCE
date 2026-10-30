#!/usr/bin/env bash
set -e
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m unittest discover -s tests && python -m helios.server
