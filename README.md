# LLM-Coordinated Drone Swarm Experiments

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Simulator](https://img.shields.io/badge/simulator-gym--pybullet--drones-green.svg)](https://github.com/utiasDSL/gym-pybullet-drones)

This repository contains the infrastructure, experimental data, and research findings for a thesis on **LLM-Coordinated Drone Swarms**. The project evaluates the ability of Large Language Models (LLMs) to act as high-level planners for multi-agent systems, focusing on how different **state space representations** affect their spatial reasoning.

---

## 🚀 Overview

The core challenge of this research is: **Can an LLM plan a collision-free formation for N drones given only a text description of the environment?**

We built a custom pipeline that bridges LLMs (via Groq/Gemini/Claude) with the `gym-pybullet-drones` physics engine. The system supports:
- **N-Drone Scaling**: Experiments ranging from 3 to 15+ drones.
- **Multiple Tasks**: Circle formations, Rendezvous, Line expansion, and more.
- **Representation Comparison**: Evaluating 5 different ways to "describe" the swarm to the LLM (Raw Coords, Relative Frames, Graph-based, Aggregate Stats, and Natural Language).

---

## 🎬 Nice Examples & Demos

The best way to see the system in action is via our pre-recorded demos and the "Full Trial" script:

### 1. The Full Trial Demo (`full_trial_demo.py`)
This script executes a complete pipeline: it builds a prompt, calls the LLM, executes the generated 4D waypoint plan with a PID controller, and produces an **annotated MP4 video**.
- **See the result**: `swarm-llm-experiment/videos/full_trial_demo.mp4`

### 2. Formation Transitions
- **Circle Formation (N=6)**: Drones rising from a random jittered start to a perfect 2m radius circle.
- **Rendezvous**: All drones converging to a shared centroid without colliding.
- **Videos**: Located in `swarm-llm-experiment/videos/`.

---

## 🧪 Research Methodology

This project evolved through two critical phases of discovery:

### Phase 1: Legacy Code Generation
Initially, we asked the LLM to write a Python `plan(state)` function.
- **Finding**: LLMs are "too smart" at coding—they would simply call `math.cos()` or `sum(x)/N`, offloading the actual spatial reasoning to the Python interpreter.
- **Archive**: Located in `docs/archive/code_generation_legacy/`.

### Phase 2: Direct Waypoint Reasoning (Current)
To truly test the LLM's "brain," we moved to **Direct JSON Waypoints**. The LLM must now output a list of timed coordinates `(t, x, y, z)` directly.
- **Finding**: This exposes significant arithmetic failures in certain representations (like `relative` frames), which were previously hidden by code generation.
- **Current Scripts**: Located in `swarm-llm-experiment/experiments_scripts/`.

---

## 📂 Repository Structure

- **`swarm-llm-experiment/`**: The active codebase.
  - `src/`: Core logic (Simulator, Reward, Representations).
  - `experiments_scripts/`: Production experiment runners.
  - `full_trial_demo.py`: The easiest way to see it working.
- **`docs/`**: The research hub.
  - `notebooks/`: Detailed hypothesis and findings for every experiment.
  - `archive/`: Historical data from both the Code-Gen and Direct-Waypoint phases.
  - [EXPERIMENTS.MD](docs/EXPERIMENTS.MD): A master index of every trial run.
  - [SIMULATOR_GUIDE.MD](docs/SIMULATOR_GUIDE.MD): Technical reference for the PyBullet integration.

---

## 🛠️ Getting Started

1.  **Get a Groq API Key**:
    - Go to the [Groq Cloud Console](https://console.groq.com/keys).
    - Create a free account and generate a new API key.
2.  **Configuration**:
    - Copy the `.env.example` file to a new file named `.env` in the `swarm-llm-experiment/` directory.
    - Open `.env` and paste your key: `export GROQ_API_KEY="your_key_here"`.
3.  **Environment**: We recommend a Conda environment with `python=3.10`.
4.  **Run**:
    ```bash
    cd swarm-llm-experiment
    # Ensure your .env is sourced or the key is in your shell
    source .env
    ./run_demo.sh  # Runs the full trial demo
    ```

---

## 📈 Key Findings at a Glance
- **Waypoints vs Endpoints**: Moving from single-target planning to timed waypoints improved convergence accuracy by over **47x**.
- **Representation Matters**: Raw coordinates are surprisingly robust with the current metrics, because its hard to define the formation reward. Natural language descriptions lose the precision needed for tight formations as N increases.
