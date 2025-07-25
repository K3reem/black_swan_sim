# Supply Chain Black Swan Simulator

**Industry:** Industrial Engineering / Logistics\
**Objective:** Train AI agents to adapt supply chain decisions during unpredictable global events using scenario-based reinforcement learning.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation & Visualization](#evaluation--visualization)
- [Extending the Simulator](#extending-the-simulator)
- [License](#license)

## Overview

Simulates a multi-stage supply chain under normal operations and injects "black swan" disruptions (e.g., supplier delays, strikes, delivery blocks). Uses SimPy for discrete event simulation and RL (Stable-Baselines3 or Ray RLlib) to train autonomous agents that minimize cost and downtime.

## Features

- Discrete-event simulation of sourcing → manufacturing → delivery
- Randomized crisis events managed by a `DisruptionManager`
- Custom OpenAI Gym/Gymnasium environment wrapper
- Reinforcement Learning agents (PPO) for decision-making
- TensorBoard integration for live training metrics
- Scripts for evaluation, plotting, and action logging
- Example hyperparameter sweep with Ray Tune

## Tech Stack

- **Python** 3.8+
- **SimPy** for simulation
- **Gymnasium** (or Gym) for environment API
- **Stable-Baselines3** (SB3) / **Ray RLlib** for RL training
- **TensorBoard** for live metrics
- **Matplotlib** for post-hoc plotting

## Setup

1. Clone repo:
   ```bash
   git clone <repo_url>
   cd black_swan_sim
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Baseline simulation:**
  ```bash
  python env/supply_chain_env.py
  ```
- **Test Gym environment:**
  ```bash
  python test_gym_env.py
  ```
- **Train with Stable-Baselines3 + TensorBoard:**
  ```bash
  python train_sb3_tb.py
  tensorboard --logdir tb_logs
  ```
- **Evaluate & plot histogram:**
  ```bash
  python evaluate_and_plot.py
  ```
- **Plot time-series of actions:**
  ```bash
  python plot_actions.py
  ```

## Project Structure

```
black_swan_sim/
├── .venv/                     # Python virtual environment
├── env/                      # Core SimPy supply chain code
│   ├── supply_chain_env.py   # SimPy environment definition
│   └── supply_chain_gym.py   # Gym wrapper
├── scenarios/                # Disruption event definitions
│   └── disruptions.py        # Black swan manager
├── train_sb3_tb.py           # SB3 training + TensorBoard
├── evaluate_and_plot.py      # Load model, run episodes, histogram
├── plot_actions.py           # Save action time-series plots
├── tune_sweep.py             # Ray Tune hyperparameter sweep example
├── test_gym_env.py           # Quick import/test of Gym env
├── requirements.txt          # pip dependencies
└── README.md                 # Project overview (this file)
```

## Training

- SB3 PPO default settings train on 100k timesteps.
- Hyperparameters can be tuned manually or via the `tune_sweep.py` script using Ray Tune.

## Evaluation & Visualization

- **evaluate\_and\_plot.py** generates a histogram of episode returns.
- **plot\_actions.py** saves time-series plots of production/delivery times and chosen actions.
- TensorBoard displays live metrics and action frequencies during training.

## Extending the Simulator

- Add new disruption types in `scenarios/disruptions.py`.
- Introduce multi-agent coordination by cloning `SupplyChainGymEnv` with multiple nodes.
- Integrate real-world data feeds for demand, commodity prices, or transport delays.

## License

MIT © AbdulKareem Raed Abu Khadair

