# Swarm LLM Experiment Core

This directory contains the entry-point scripts and infrastructure for the LLM-drone swarm experiments.

## 📂 Structure

- **`src/`**: The "engine" of the project. Contains core Python modules:
  - `simulator.py`: High-level wrapper around `gym-pybullet-drones`.
  - `executor.py`: Sandbox for executing LLM-generated code.
  - `llm_connector.py`: Groq/LLM integration.
  - `prompt_builder.py`: Prompt engineering.
  - `representations.py`: State space encoders.
  - `reward.py`: Reward and task completion logic.
  - `experiment.py`: Multi-trial experiment framework.

- **`experiments_scripts/`**: Individual runners for production experiments (Direct JSON Waypoints).

- **`videos/`**: Output directory for generated MP4 recordings.

## 🚀 Key Scripts

- **`full_trial_demo.py`**: The main end-to-end demo. Generates an annotated video.
- **`run_demo.sh`**: A simple bash wrapper to activate the environment and run the full demo.
- **`waypoint_demo.py`**: Validates the timed waypoint controller.
- **`record_simulation.py`**: Helper to record high-resolution videos of specific scenarios.
