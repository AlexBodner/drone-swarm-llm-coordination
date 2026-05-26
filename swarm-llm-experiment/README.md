# Swarm LLM Experiment Core

This directory contains the core logic for the LLM-drone swarm coordination system.

## Core Modules

- `simulator.py`: High-level wrapper around `gym-pybullet-drones`.
- `executor.py`: Sandbox for executing LLM-generated Python code.
- `llm_connector.py`: Handles API calls to Groq and other LLM providers.
- `prompt_builder.py`: Constructs prompts from swarm state and task descriptions.
- `representations.py`: Implements different state representations (raw, relative, graph, etc.).
- `reward.py`: Calculates formation and rendezvous rewards.
- `experiment.py`: Framework for running systematic multi-trial experiments.
- `analysis.py`: Tools for plotting and analyzing experimental results.

## Key Scripts

- `full_trial_demo.py`: A complete end-to-end run with video generation.
- `waypoint_demo.py`: Tests the waypoint-following controller.
- `record_simulation.py`: Utility to record high-quality MP4 videos.

## Subdirectories

- `experiments_scripts/`: All individual experiment runners (`run_e0*.py`).
- `results_archive/`: Data and plots from completed experiments.
- `videos/`: Output folder for demo and experiment recordings.
