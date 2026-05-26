#!/usr/bin/env bash
set -e
# export GROQ_API_KEY="your_key_here"
eval "$(conda shell.bash hook)"
conda activate swarm-llm
python full_trial_demo.py
