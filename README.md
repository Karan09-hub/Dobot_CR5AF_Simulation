# Dobot CR5AF Robot Simulation & Motion Planning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x%20Physics-orange.svg)](https://mujoco.org/)
[![cuRobo](https://img.shields.io/badge/cuRobo-GPU%20Motion%20Planning-green.svg)](https://curobo.org/)
[![Viser](https://img.shields.io/badge/Viser-Interactive%203D%20Web%20UI-purple.svg)](https://github.com/nerfstudio-project/viser)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](tests/)

A high-performance robotics simulation and motion planning framework for the **Dobot CR5 / CR5AF** 6-DOF collaborative arm equipped with an articulated **Robotiq 2F-85** parallel-jaw gripper.

Combines GPU-accelerated collision-free trajectory optimization (**cuRobo**), real-time inverse kinematics (**PyRoKi**), physics simulation (**MuJoCo 3.x**), and an interactive browser-based 3D interface (**Viser**).

---

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Pick & Place Autonomous Pipeline](#pick--place-autonomous-pipeline)
- [Motion Planners (GPU & CPU)](#motion-planners-gpu--cpu)
  - [cuRobo GPU Planner](#curobo-gpu-planner)
  - [CPU Fallback Planner](#cpu-fallback-planner)
- [Project Layout](#project-layout)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Robot Model Pipeline](#robot-model-pipeline)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

The simulator models the physical interaction between the Dobot CR5 manipulator, the Robotiq 2F-85 gripper, and scene obstacles in full 3D physics. All joint limits, motor dynamics, gravity, and contact friction are computed in real time.

```
                 ┌───────────────────────── app.py (50 Hz control loop) ──────────────────────┐
  Browser  ◄────►│  Viser UI          3D interactive view, live joint streaming, task buttons │
  (Client)       │                                                                            │
                 │  Task Script       Pick & Place finite state machine                        │
                 │       │                                                                    │
                 │       ▼                                                                    │
                 │  Motion Planner    cuRobo (GPU trajectory optimization around obstacles)   │
                 │                    or CPU Fallback (height clearance + waypoint profiling)  │
                 │       │                                                                    │
                 │       ▼                                                                    │
                 │  Physics (MuJoCo)  Position-controlled arm, articulated 4-bar linkage     │
                 │                    gripper, contact friction, mass & inertias              │
                 └────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Physics-Accurate Simulation (MuJoCo 3.x):**
  - Articulated Robotiq 2F-85 gripper model with authentic 4-bar linkage kinematics (derived from MuJoCo Menagerie).
  - True contact dynamics: grasping is achieved purely through frictional contact between silicone fingertip pads and objects (no magnet/teleport hacks).
- **Dual Motion Planning Engine:**
  - **cuRobo (GPU):** Full collision-sphere trajectory optimization running on CUDA GPUs, finding minimum-jerk collision-free trajectories around obstacles in milliseconds.
  - **CPU Fallback:** Robust Cartesian / joint-space trajectory planner with obstacle clearance height interpolation and joint velocity profiling, running without CUDA or GPU requirements.
- **Fast Inverse Kinematics (PyRoKi):**
  - Damped least-squares optimization formulated in JAX on SE(3) manifolds.
  - Sub-millimeter TCP positioning accuracy.
- **Interactive Viser 3D Web UI:**
  - Zero desktop UI dependencies: streams live 3D meshes and joint angles directly to modern web browsers.
  - Interactive 6-DOF transform gizmo to drag the end-effector target in real time.
  - Obstacle controls: spawn, relocate, or toggle obstacles dynamically.
  - Optional native MuJoCo passive viewer integration (`--mujoco-viewer`).

---

## Pick & Place Autonomous Pipeline

The simulator includes an autonomous pick-and-place task demonstrating closed-loop coordination:

```
  1. Home Pose        Elbow-up ready configuration
         │
         ▼
  2. Approach         Move TCP above the cube (approach height 120 mm)
         │
         ▼
  3. Pre-Grasp        Descend vertically with open gripper fingers
         │
         ▼
  4. Grasp            Close Robotiq 2F-85 fingers (contact detection via friction)
         │
         ▼
  5. Lift             Vertical lift clear of the table surface
         │
         ▼
  6. Transport        Collision-free trajectory over the obstacle wall to target pad
         │
         ▼
  7. Place            Descend onto destination pad
         │
         ▼
  8. Release          Open gripper fingers and retract vertically
```

Trigger the task directly from the Viser UI button or programmatically via the task state machine.

---

## Motion Planners (GPU & CPU)

### cuRobo GPU Planner
When an NVIDIA GPU with appropriate drivers is available, `make_planner()` initializes NVIDIA cuRobo:
- Collision geometry is represented as an optimized hierarchy of bounding spheres.
- Handles self-collision avoidance and static obstacle avoidance.
- Accounts for grasped objects dynamically by attaching virtual collision spheres to the end-effector.

To build the cuRobo robot configuration:
```bash
python tools/build_curobo_robot.py
```

### CPU Fallback Planner
When running on systems without CUDA (or when passing `--planner cpu`):
- Uses analytical obstacle bounding boxes to compute safe clearance altitudes.
- Interpolates Cartesian waypoints with smooth S-curve joint velocity limits (3.14 rad/s max).
- Validates each waypoint through MuJoCo contact checking to guarantee collision safety.

---

## Project Layout

```text
Dobot_CR5AF_Simulation/
├── app.py                      # Main simulation application & Viser web server
├── requirements.txt            # Python package dependencies
├── LICENSE                     # MIT license
├── README.md                   # Project documentation
├── cr5sim/                     # Core simulation package
│   ├── __init__.py             # Package initialization
│   ├── env_check.py            # Pre-flight environment check
│   ├── kinematics.py           # PyRoKi FK/IK on the Robotiq TCP
│   ├── paths.py                # Asset, URDF, and MJCF path registry
│   ├── physics.py              # MuJoCo world, actuators, obstacle & contact checks
│   ├── planner.py              # cuRobo GPU planner & CPU fallback planner
│   └── viser_scene.py          # MuJoCo-to-Viser scene streaming
├── tools/
│   ├── build_urdf.py           # Generates urdf/cr5_robot.urdf from Dobot xacro
│   ├── build_mujoco_model.py   # Converts DAE to OBJ and builds MJCF
│   ├── build_curobo_robot.py   # Fits collision spheres for cuRobo (GPU)
│   └── install_curobo.sh       # Automated cuRobo installer
├── tests/
│   └── test_simulation.py      # Pytest unit tests for URDF, MuJoCo, IK & planner
├── Assets/                     # Original Dobot mesh files (.dae) & robot xacro
├── meshes/obj/                 # Generated OBJ meshes for visual and collision geoms
├── mjcf/                       # MuJoCo XML scene definitions
│   ├── cr5.xml                 # Arm + Robotiq 2F-85 MJCF model
│   └── scene.xml               # Complete world (table, wall, cube, lighting)
└── third_party/robotiq_2f85/   # MuJoCo Menagerie Robotiq 2F-85 model (BSD-2)
```

---

## Installation & Setup

### 1. Prerequisites
- Linux (Ubuntu 20.04 / 22.04 / 24.04 tested)
- Python 3.10, 3.11, or 3.12
- Optional: NVIDIA GPU (Driver $\ge$ 525 for cuRobo acceleration)

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Dobot_CR5AF_Simulation.git
cd Dobot_CR5AF_Simulation

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Optional: Install cuRobo GPU Acceleration

```bash
# Installs cuRobo v0.8.0 with CUDA PyTorch:
./tools/install_curobo.sh

# Fit collision spheres to generate curobo_cfg/cr5.yml:
python tools/build_curobo_robot.py
```

---

## Usage

### Run the Simulator
```bash
# Auto-detects best planner (cuRobo if GPU is present, CPU otherwise):
python app.py

# Force CPU planner:
python app.py --planner cpu

# Specify custom web port:
python app.py --port 8085

# Also open native MuJoCo desktop viewer:
python app.py --mujoco-viewer
```

Open `http://localhost:8080` in your web browser.

---

## Running Tests

Run the test suite to verify model loading, kinematics, and physics:

```bash
pytest tests -v
```

---

## Robot Model Pipeline

The robot model pipeline guarantees consistency across all representations:

```text
Assets/cr5_robot.xacro + *.dae
        │  tools/build_urdf.py          (+ Robotiq 2F-85 from third_party/)
        ▼
urdf/cr5_robot.urdf  ──────────► PyRoKi (IK), cuRobo (Motion Planning)
        │  tools/build_mujoco_model.py
        ▼
mjcf/cr5.xml + scene.xml ──────► MuJoCo (Physics Simulation)
```

**Modifications & Fixes to Original Dobot Assets:**
- Joint velocity limits corrected from unphysical 100 rad/s to nominal 3.14 rad/s (180°/s).
- Stripped Gazebo-specific ROS tags.
- Converted mesh references to relative paths for standalone operation.
- Open-gripper Robotiq 2F-85 model integrated seamlessly with exact fingertip pinch TCP offset.

---

## Troubleshooting

| Symptom | Cause & Solution |
|---|---|
| `Missing packages: ...` | Python environment missing dependencies. Activate `.venv` or run `pip install -r requirements.txt`. |
| Port 8080 already in use | Viser automatically selects the next available port (e.g. 8081). Use the URL printed in the terminal or pass `--port <num>`. |
| Initial IK pause (5-10 s) | JAX JIT-compiles the IK solver on first execution. Subsequent solves run at sub-millisecond speeds. |
| `cuRobo not found / driver missing` | cuRobo falls back automatically to the robust CPU planner. To enable GPU, install NVIDIA proprietary drivers (`sudo apt install nvidia-driver-550`) and run `./tools/install_curobo.sh`. |

---

## License

- Code and scripts: [MIT License](LICENSE).
- Robotiq 2F-85 model: [BSD-2 License](third_party/robotiq_2f85/LICENSE) (MuJoCo Menagerie / DeepMind).
- Dobot CR5 meshes & descriptions: Copyright (c) Dobot.
