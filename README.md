# Robotic Arm Kinematics & Simulation

This project implements the kinematic modeling and simulation of a 3-DOF robotic arm. It includes symbolic derivations of the forward and differential kinematics, as well as a numerical simulation of the robot executing a trajectory planned via quintic polynomials.

## Project Overview

The core objective is to model a robotic manipulator using the Denavit-Hartenberg (DH) convention and simulate its motion between two points in 3D space.

Key features include:
- **Symbolic Computation**: Automating the derivation of Homogeneous Transformation Matrices and the Geometric Jacobian using `sympy`.
- **Forward Kinematics**: Calculating the end-effector position based on joint angles.
- **Inverse Kinematics**: Analytically solving for joint angles given a desired end-effector position.
- **Trajectory Planning**: Utilizing a 5th-degree (quintic) polynomial to generate smooth paths with velocity constraints.
- **3D Visualization**: Visualizing the robot's workspace and animating its movement using `matplotlib`.

## Folder Structure

- `main.py`: The entry point for running calculations and simulations.
- `scripts/`: Contains the core logic modules.
  - `theoretical_calculations.py`: Handles symbolic math for kinematics (DH parameters, Jacobian, etc.).
  - `kinematic_simulation.py`: Implements the simulation loop and visualization tools.
- `motion gifs/`: Directory for saving generated animations.
- `report_03122074.pdf`: (Reference) Original project report in Greek.

## Requirements

- Python 3.x
- `numpy`
- `sympy`
- `matplotlib`
- `tqdm`

You can install the dependencies using pip:
```bash
pip install numpy sympy matplotlib tqdm
```

## How to Run

### 1. Run the Simulation
To visualize the robot's motion and generate plots for position, velocity, and joint angles:

```bash
python main.py
```
*By default, this runs `run_simulation()`, which calculates the trajectory and displays the results.*

### 2. Perform Theoretical Calculations
To see the symbolic derivation of the kinematic matrices and polynomials:

1. Open `main.py`.
2. Uncomment the line `run_theoretical_calculations()`.
3. Run the script:
   ```bash
   python main.py
   ```

## License
This project is open-source.


