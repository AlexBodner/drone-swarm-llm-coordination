# LLM-Coordinated Drone Swarm Experiments

This repository contains the codebase and experiment results for the thesis on LLM-coordinated drone swarms using `gym-pybullet-drones`.

## Repository Structure

- `swarm-llm-experiment/`: Main project directory.
  - `src/`: Core simulation and planning modules.
  - `experiments_scripts/`: **Current** experiments using **Direct JSON Waypoint** generation.
  - `full_trial_demo.py`: Main demo script.
- `docs/`: Project documentation and archives.
  - `notebooks/`: Detailed analysis and hypotheses for each experiment.
  - `archive/`:
    - `code_generation_legacy/`: Previous phase experiments that used **Python Code Generation**.
    - `results/`: Processed data and plots from current experiments.

## Methodology

This project evolved through two distinct phases:

1.  **Legacy (Code Generation)**: The LLM generated Python code to calculate plans. While effective, this allowed the LLM to offload complex arithmetic to the Python interpreter.
2.  **Current (Direct Waypoints)**: The LLM generates timed 4D waypoint trajectories (`t, x, y, z`) directly in JSON. This provides a more rigorous test of the LLM's spatial reasoning and arithmetic capabilities.

## Getting Started

1. Create a `.env` file in `swarm-llm-experiment/` (see `.env.example`).
2. Install dependencies (recommended to use a virtual environment).
3. Run the demo:
   ```bash
   cd swarm-llm-experiment
   ./run_demo.sh
   ```

For more details, see [SIMULATOR_GUIDE.MD](SIMULATOR_GUIDE.MD) and [EXPERIMENTS.MD](EXPERIMENTS.MD).
