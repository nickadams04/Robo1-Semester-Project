# Robotic Arm Kinematics & Simulation

A standalone reference implementation for modeling, simulating, and visualizing a 3-DOF robotic arm. The project couples symbolic modeling (forward/differential kinematics) with numerical simulation, trajectory planning, and high-quality visual outputs.

> **Note:** The historical report `report_03122074.pdf` remains in Greek for archival reasons.

## Quick Preview

| Up-Up | Up-Down |
| --- | --- |
| ![Elbow Up-Up](motion%20gifs/robot_motion_UU.gif) | ![Elbow Up-Down](motion%20gifs/robot_motion_UD.gif) |

| Down-Up | Down-Down |
| --- | --- |
| ![Elbow Down-Up](motion%20gifs/robot_motion_DU.gif) | ![Elbow Down-Down](motion%20gifs/robot_motion_DD.gif) |

## Features

- **Symbolic derivations** (via `sympy`) for DH transforms, Jacobians, and quintic trajectory coefficients.
- **Modular simulation stack** split into control, core engine, and visualization modules.
- **Batch plotting & GIF exports** for workspace projections, state trajectories, and motion snapshots.

## Architecture Overview

| Module | Responsibility |
| --- | --- |
| `scripts/control.py` | Trajectory planning primitives (currently quintic via-point interpolation). |
| `scripts/simulation_engine.py` | Robot model, inverse kinematics, and discrete simulation loop. |
| `scripts/visualization.py` | Workspace plots, result dashboards, GIF generation. |
| `scripts/kinematic_simulation.py` | High-level orchestration (`RobotSimulation` wrapper, legacy helpers). |
| `scripts/theoretical_calculations.py` | Symbolic derivations for kinematic expressions. |

## Requirements

- Python 3.10+
- `numpy`, `sympy`, `matplotlib`, `tqdm`

Install dependencies:

```bash
pip install numpy sympy matplotlib tqdm
```

## Usage

### 1. Run the simulation and export visuals

```bash
python main.py
```

The script plots the workspace, runs the discrete simulation, renders state plots, and saves a motion GIF under `motion gifs/`.

### 2. Reproduce symbolic derivations

Uncomment `run_theoretical_calculations()` in `main.py` and run the script to print the symbolic matrices and quintic coefficients.

## Folder Structure

- `main.py` – entry point exposing CLI helpers.
- `scripts/`
  - `control.py`
  - `simulation_engine.py`
  - `visualization.py`
  - `kinematic_simulation.py`
  - `theoretical_calculations.py`
- `motion gifs/` – pre-rendered sample animations.
- `report_03122074.pdf` – original course report (Greek).

## License

This project is open-source; feel free to adapt it for your own robotic arm experiments.
