# LLM-Coordinated Drone Swarm Experiments

This repository contains the infrastructure, experimental data, and findings for a research project on LLM-coordinated drone swarms. The project evaluates the ability of Large Language Models (LLMs) to function as high-level planners for multi-agent systems, specifically examining how different state space representations affect spatial reasoning.

## Project Overview

The primary research objective is to evaluate whether an LLM can plan collision-free formations for N-drone swarms based solely on environmental text descriptions.

The implementation utilizes a custom pipeline connecting LLMs (via Groq/Gemini/Claude APIs) with the `gym-pybullet-drones` physics engine. Key features include:
- **N-Drone Scaling**: Evaluation across swarms of size N ∈ {3, 6, 10, 15}.
- **Task Variety**: Implementation of Circle formations, Rendezvous, and Line expansion tasks.
- **Representation Analysis**: Comparative study of 5 state representations: Raw Coordinates, Relative Frames, Graph-based, Aggregate Statistics, and Natural Language.

## Demos and Examples

### Full Trial Pipeline (`full_trial_demo.py`)
This script demonstrates the complete experimental flow: prompt construction, LLM API call, execution of the generated 4D waypoint plan using a PID controller, and generation of an annotated simulation video.
- **Output**: `swarm-llm-experiment/videos/full_trial_demo.mp4`

### Simulation Results
- **Circle Formation**: Transition from randomized initial positions to a 2m radius circular formation.
- **Rendezvous**: Swarm convergence to a shared centroid with collision avoidance.
- **Video Archive**: Pre-recorded results are located in `swarm-llm-experiment/videos/`.

## Methodology Evolution

The research was conducted in two primary phases:

### Phase 1: Python Code Generation (Legacy)
The initial approach required the LLM to output executable Python code.
- **Observations**: LLMs tended to offload spatial arithmetic to the Python interpreter (e.g., using `math` libraries), which partially masked their inherent spatial reasoning limitations.
- **Archive**: Documentation and data in `docs/archive/code_generation_legacy/`.

### Phase 2: Direct Waypoint Trajectories (Current)
To isolate the LLM's spatial reasoning capabilities, the methodology shifted to direct generation of timed 4D coordinates (t, x, y, z) in JSON format.
- **Observations**: This mode revealed significant representation-dependent arithmetic failures that were previously obscured.
- **Current Scripts**: Source files in `swarm-llm-experiment/experiments_scripts/`.

## Repository Structure

- **`swarm-llm-experiment/`**: Core project directory.
  - `src/`: Implementation of simulator wrappers, reward functions, and representations.
  - `experiments_scripts/`: Active experiment runner scripts.
  - `full_trial_demo.py`: Main demonstration entry point.
- **`docs/`**: Research documentation.
  - `notebooks/`: Detailed hypotheses and analysis for each experiment.
  - `archive/`: Historical data and legacy scripts.
  - [EXPERIMENTS.MD](docs/EXPERIMENTS.MD): Comprehensive index of all experimental trials.
  - [SIMULATOR_GUIDE.MD](docs/SIMULATOR_GUIDE.MD): Technical integration details for `gym-pybullet-drones`.

## Setup and Execution

1.  **API Configuration**: Create a `.env` file in `swarm-llm-experiment/` using `.env.example` as a template. Generate a key via the [Groq Cloud Console](https://console.groq.com/keys).
2.  **Environment**: Requires Python 3.10 (Conda environment recommended).
3.  **Run Demo**:
    ```bash
    cd swarm-llm-experiment
    source .env
    ./run_demo.sh
    ```

## Preliminary Findings
- **Trajectory Planning**: Utilizing timed waypoints instead of single endpoints resulted in a significant improvement in convergence accuracy (approx. 47x).
- **Representation Sensitivity**: High-fidelity representations (like raw coordinates) showed greater robustness as swarm size (N) increased compared to more abstract or verbal descriptions.
