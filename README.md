# LLM-Coordinated Drone Swarm Experiments

This repository contains the codebase and experiment results for the thesis on LLM-coordinated drone swarms using `gym-pybullet-drones`.

## Repository Structure

- `swarm-llm-experiment/`: Main project directory.
  - `full_trial_demo.py`: Main demo script showing the full pipeline (Task -> Prompt -> Plan -> Simulation -> Video).
  - `waypoint_demo.py`: Demo focusing on waypoint tracking.
  - `simulator.py`: Drone simulation wrapper.
  - `llm_connector.py`: Groq/LLM integration.
  - `prompt_builder.py`: Prompt engineering logic.
  - `experiments_scripts/`: Collection of experimental runner scripts.
  - `results_archive/`: Archived results from previous experiments.
- `experiments/`: Per-experiment documentation and notebooks.
- `videos/`: Generated videos and visualizations.

## Getting Started

1. Create a `.env` file in `swarm-llm-experiment/` (see `.env.example`).
2. Install dependencies (recommended to use a virtual environment).
3. Run the demo:
   ```bash
   cd swarm-llm-experiment
   ./run_demo.sh
   ```

For more details, see [SIMULATOR_GUIDE.MD](SIMULATOR_GUIDE.MD) and [EXPERIMENTS.MD](EXPERIMENTS.MD).
