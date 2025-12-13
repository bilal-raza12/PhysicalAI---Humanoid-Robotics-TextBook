---
id: ch09-isaac-intro
title: "Chapter 9: Introduction to NVIDIA Isaac Sim"
sidebar_position: 2
---

# Chapter 9: Introduction to NVIDIA Isaac Sim

**Estimated Time**: 4-5 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Install and configure** NVIDIA Isaac Sim for robotics development
2. **Navigate** the Isaac Sim interface and workspace
3. **Understand** the Omniverse platform architecture
4. **Create** basic simulation scenes with robots
5. **Connect** Isaac Sim to ROS 2 for robot control

---

## 9.1 NVIDIA Isaac Platform Overview

NVIDIA Isaac is a comprehensive platform for developing and deploying AI-powered robots.

### Isaac Platform Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| Isaac Sim | Simulation | Physics, rendering, sensors |
| Isaac ROS | Perception | GPU-accelerated perception |
| Isaac SDK | Development | Robot applications |
| Isaac Gym | RL Training | Parallel environments |

### System Requirements

```
Minimum Requirements:
- GPU: NVIDIA RTX 2070 or higher
- VRAM: 8 GB minimum (16+ GB recommended)
- RAM: 32 GB
- Storage: 50 GB SSD
- OS: Ubuntu 20.04/22.04 or Windows 10/11

Recommended for Humanoid Simulation:
- GPU: NVIDIA RTX 4090 or A6000
- VRAM: 24+ GB
- RAM: 64 GB
- Storage: 100 GB NVMe SSD
```

### Installation

```bash
# Install NVIDIA Omniverse Launcher
# Download from: https://www.nvidia.com/en-us/omniverse/

# Via Omniverse Launcher:
# 1. Open Omniverse Launcher
# 2. Go to Exchange tab
# 3. Search for "Isaac Sim"
# 4. Click Install (2023.1.1 or later)

# Verify installation
cd ~/.local/share/ov/pkg/isaac_sim-2023.1.1
./isaac-sim.sh --help

# Launch Isaac Sim
./isaac-sim.sh
```

---

## 9.2 Isaac Sim Interface

### Workspace Overview

```
┌─────────────────────────────────────────────────────────┐
│  Menu Bar                                               │
├─────────┬───────────────────────────────┬───────────────┤
│         │                               │               │
│  Stage  │       Viewport                │  Property     │
│  Panel  │       (3D View)               │  Panel        │
│         │                               │               │
│         │                               │               │
├─────────┴───────────────────────────────┴───────────────┤
│  Content Browser / Console / Script Editor              │
└─────────────────────────────────────────────────────────┘
```

### Key Panels

| Panel | Function |
|-------|----------|
| Stage | Scene hierarchy (USD prims) |
| Viewport | 3D visualization |
| Property | Selected object properties |
| Content | Asset browser |
| Console | Python output and errors |

### Basic Navigation

```
Mouse Controls:
- Left click: Select
- Right click + drag: Rotate view
- Middle click + drag: Pan view
- Scroll wheel: Zoom

Keyboard Shortcuts:
- F: Frame selected
- G: Move tool
- R: Rotate tool
- S: Scale tool
- Space: Play/Pause simulation
- Ctrl+Z: Undo
```

---

## 9.3 Universal Scene Description (USD)

Isaac Sim uses USD as its scene format.

### USD Basics

```python
# Working with USD in Isaac Sim
from pxr import Usd, UsdGeom, Gf

# Open a stage
stage = Usd.Stage.Open("/path/to/scene.usd")

# Create a new prim
xform = UsdGeom.Xform.Define(stage, "/World/MyRobot")

# Set transform
xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))

# Save stage
stage.Save()
```

### Isaac Sim Python API

```python
# isaac_sim_basics.py
from omni.isaac.kit import SimulationApp

# Launch simulation
simulation_app = SimulationApp({"headless": False})

# Import after SimulationApp is created
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np

# Create world
world = World()

# Add ground plane
world.scene.add_default_ground_plane()

# Add a cube
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="my_cube",
        position=np.array([0, 0, 1.0]),
        scale=np.array([0.5, 0.5, 0.5]),
        color=np.array([1.0, 0, 0])
    )
)

# Reset world
world.reset()

# Run simulation
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
```

---

## 9.4 Loading Robot Models

### Import URDF

```python
# import_humanoid.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.urdf import _urdf
import omni.kit.commands

# Create world
world = World()
world.scene.add_default_ground_plane()

# URDF import configuration
urdf_interface = _urdf.acquire_urdf_interface()

import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.convex_decomp = True
import_config.fix_base = False
import_config.make_default_prim = True
import_config.self_collision = False
import_config.create_physics_scene = True

# Import URDF
urdf_path = "/path/to/humanoid.urdf"
result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
)

print(f"Robot imported at: {prim_path}")

# Reset and run
world.reset()

for i in range(1000):
    world.step(render=True)

simulation_app.close()
```

### Articulation Controller

```python
# articulation_control.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np

world = World()
world.scene.add_default_ground_plane()

# Load robot (assuming already in scene)
robot = world.scene.add(
    Articulation(
        prim_path="/World/humanoid",
        name="humanoid_robot"
    )
)

world.reset()

# Get joint information
num_dof = robot.num_dof
joint_names = robot.dof_names
print(f"DOF: {num_dof}")
print(f"Joints: {joint_names}")

# Get current state
joint_positions = robot.get_joint_positions()
joint_velocities = robot.get_joint_velocities()

print(f"Positions: {joint_positions}")
print(f"Velocities: {joint_velocities}")

# Set joint targets
target_positions = np.zeros(num_dof)
target_positions[0] = 0.5  # First joint to 0.5 rad

# Apply action
action = ArticulationAction(
    joint_positions=target_positions,
    joint_velocities=None,
    joint_efforts=None
)
robot.apply_action(action)

# Run simulation
for i in range(500):
    world.step(render=True)

simulation_app.close()
```

---

## 9.5 ROS 2 Integration

### Enable ROS 2 Bridge

```python
# ros2_bridge_setup.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation

# Enable ROS 2 extension
import omni.graph.core as og
from omni.isaac.core_nodes.scripts.utils import set_target_prims

# Create world
world = World()

# Enable ROS 2 Bridge
import omni.isaac.ros2_bridge

# Create action graph for ROS 2
og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
            ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
            ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
            ("ROS2Context.outputs:context", "PublishJointState.inputs:context"),
            ("ROS2Context.outputs:context", "SubscribeJointState.inputs:context"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("PublishJointState.inputs:topicName", "/joint_states"),
            ("SubscribeJointState.inputs:topicName", "/joint_commands"),
        ],
    },
)

# Set target robot
set_target_prims(
    primPath="/ActionGraph/PublishJointState",
    inputName="inputs:targetPrim",
    targetPrimPaths=["/World/humanoid"]
)

world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
```

### ROS 2 Joint Commander

```python
#!/usr/bin/env python3
"""
isaac_joint_commander.py
Send joint commands to Isaac Sim via ROS 2.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

class IsaacJointCommander(Node):
    def __init__(self):
        super().__init__('isaac_joint_commander')

        # Publisher for joint commands
        self.cmd_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )

        # Subscriber for joint states
        self.state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10
        )

        # Timer for commands
        self.timer = self.create_timer(0.1, self.publish_command)

        self.joint_names = []
        self.current_positions = []
        self.time = 0.0

        self.get_logger().info('Isaac Joint Commander started')

    def state_callback(self, msg):
        self.joint_names = list(msg.name)
        self.current_positions = list(msg.position)

    def publish_command(self):
        if not self.joint_names:
            return

        self.time += 0.1

        # Generate sinusoidal motion
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.joint_names

        positions = []
        for i, name in enumerate(self.joint_names):
            # Different frequency for each joint
            pos = 0.5 * np.sin(self.time + i * 0.5)
            positions.append(pos)

        cmd.position = positions
        cmd.velocity = []
        cmd.effort = []

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacJointCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 9.6 Physics Configuration

### PhysX Settings

```python
# physics_config.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from pxr import UsdPhysics, PhysxSchema

# Create world with custom physics
world = World(
    physics_dt=1.0/240.0,  # 240 Hz physics
    rendering_dt=1.0/60.0,  # 60 Hz rendering
    stage_units_in_meters=1.0
)

# Access physics scene
stage = world.stage
physics_scene_path = "/physicsScene"

# Configure PhysX
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(
    stage.GetPrimAtPath(physics_scene_path)
)

# Set solver iterations (higher = more accurate but slower)
physx_scene.CreateSolverPositionIterationCountAttr().Set(16)
physx_scene.CreateSolverVelocityIterationCountAttr().Set(8)

# Enable GPU dynamics
physx_scene.CreateEnableGPUDynamicsAttr().Set(True)

# Set gravity
physics_scene = UsdPhysics.Scene.Get(stage, physics_scene_path)
physics_scene.CreateGravityDirectionAttr().Set((0, 0, -1))
physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

print("Physics configured for humanoid simulation")

world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
```

---

## Exercises

### Exercise 9.1: Install Isaac Sim

**Objective**: Set up NVIDIA Isaac Sim development environment.

**Difficulty**: Beginner | **Estimated Time**: 60 minutes

#### Instructions

1. Install NVIDIA Omniverse Launcher
2. Install Isaac Sim 2023.1.1+
3. Launch and verify GPU acceleration
4. Explore the sample scenes

#### Expected Outcome

Isaac Sim running with sample scenes loading correctly.

---

### Exercise 9.2: Create a Basic Scene

**Objective**: Build a simple simulation environment.

**Difficulty**: Beginner | **Estimated Time**: 30 minutes

#### Instructions

1. Create new scene
2. Add ground plane
3. Add primitive objects (cubes, spheres)
4. Configure physics and run simulation

---

### Exercise 9.3: Import Humanoid URDF

**Objective**: Load a humanoid robot model into Isaac Sim.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Export URDF from ROS 2 workspace
2. Use URDF importer extension
3. Verify joint configuration
4. Test basic motion

---

### Exercise 9.4: ROS 2 Connection

**Objective**: Establish ROS 2 communication with Isaac Sim.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Enable ROS 2 bridge extension
2. Create action graph for joint state publishing
3. Subscribe to joint commands
4. Control robot from ROS 2 node

---

## Summary

In this chapter, you learned:

- **Isaac Sim** provides GPU-accelerated robotics simulation
- **USD format** enables rich scene composition
- **Python API** allows programmatic scene creation
- **ROS 2 bridge** connects simulation to robot software
- **PhysX** provides accurate physics for humanoids

---

## References

[1] NVIDIA, "Isaac Sim Documentation," [Online]. Available: https://docs.omniverse.nvidia.com/isaacsim/latest/.

[2] Pixar, "Universal Scene Description," [Online]. Available: https://graphics.pixar.com/usd/.

[3] NVIDIA, "PhysX SDK Documentation," [Online]. Available: https://nvidia-omniverse.github.io/PhysX/.

[4] Open Robotics, "ROS 2 Humble," [Online]. Available: https://docs.ros.org/en/humble/.
