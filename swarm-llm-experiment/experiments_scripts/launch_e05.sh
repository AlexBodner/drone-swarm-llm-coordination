#!/usr/bin/env zsh
cd "$(dirname "$0")"
rm -rf results_e05
mkdir -p results_e05
nohup /opt/anaconda3/envs/swarm-llm/bin/python run_e05.py > results_e05/run.log 2>&1 &
echo "PID: $!"
