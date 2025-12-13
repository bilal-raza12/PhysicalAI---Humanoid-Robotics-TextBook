---
id: ch01-intro-ros2
title: "Chapter 1: Introduction to ROS 2 for Humanoids"
sidebar_position: 2
---

# Chapter 1: Introduction to ROS 2 for Humanoids

**Estimated Time**: 4-5 hours | **Exercises**: 3

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Explain** the evolution from ROS 1 to ROS 2 and why ROS 2 is essential for humanoid robotics
2. **Describe** the core architectural components of ROS 2 (nodes, topics, services, actions)
3. **Set up** a complete ROS 2 Humble development environment on Ubuntu 22.04
4. **Create** your first ROS 2 workspace and package
5. **Run** and understand the classic talker/listener demonstration

---

## 1.1 Introduction

The Robot Operating System (ROS) has become the de facto standard for robotics software development. Just as the human nervous system coordinates signals between the brain, sensors, and muscles, ROS 2 provides the communication infrastructure that connects every component of a modern robot.

For humanoid robotics, ROS 2 is particularly crucial because it handles:

- **Real-time control** of dozens of joints simultaneously
- **Sensor fusion** from cameras, IMUs, force sensors, and more
- **Distributed computing** across multiple processors
- **Safety-critical operations** with deterministic behavior

### Why ROS 2 for Humanoids?

Humanoid robots present unique challenges that ROS 2 is designed to address:

| Challenge | ROS 2 Solution |
|-----------|----------------|
| High joint count (20-50+ DoF) | Efficient message passing with zero-copy |
| Real-time balance control | DDS-based QoS with deadline policies |
| Multi-sensor coordination | Built-in time synchronization |
| Safety requirements | Lifecycle management for controlled startup/shutdown |
| Complex state machines | Actions for long-running tasks with feedback |

---

## 1.2 ROS 2 Architecture Overview

ROS 2 is built on a distributed architecture where independent processes (nodes) communicate through well-defined interfaces.

### Core Concepts

#### Nodes

A **node** is the fundamental unit of computation in ROS 2. Each node is responsible for a single, modular purpose:

```python
# Example: A simple ROS 2 node structure
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.get_logger().info('Humanoid controller initialized')
```

For a humanoid robot, you might have nodes for:
- Joint state publisher
- Balance controller
- Vision processing
- Speech recognition
- Motion planning

#### Topics

**Topics** enable asynchronous, many-to-many communication through a publish/subscribe pattern:

```
┌─────────────┐     /joint_states      ┌─────────────┐
│ Joint State │ ─────────────────────▶ │  Balance    │
│  Publisher  │                        │ Controller  │
└─────────────┘                        └─────────────┘
                                              │
                                              ▼
                                       /cmd_vel
```

#### Services

**Services** provide synchronous request/response communication:

```
┌─────────────┐   Request: SetPose    ┌─────────────┐
│   Client    │ ────────────────────▶ │   Server    │
│             │ ◀──────────────────── │             │
└─────────────┘   Response: Success   └─────────────┘
```

#### Actions

**Actions** handle long-running tasks with feedback and cancellation:

```
┌─────────────┐                       ┌─────────────┐
│   Client    │ ──── Goal ──────────▶ │   Server    │
│             │ ◀─── Feedback ─────── │             │
│             │ ◀─── Result ───────── │             │
└─────────────┘                       └─────────────┘
```

### Communication Middleware: DDS

ROS 2 uses the Data Distribution Service (DDS) standard for communication, providing:

- **Quality of Service (QoS)** policies for reliability, durability, and deadlines
- **Discovery** of nodes without a central master
- **Security** with authentication and encryption
- **Real-time** capabilities for deterministic messaging

---

## 1.3 Development Environment Setup

### System Requirements

| Component | Requirement |
|-----------|-------------|
| Operating System | Ubuntu 22.04 LTS (Jammy Jellyfish) |
| RAM | 8 GB minimum, 16 GB recommended |
| Storage | 20 GB free space |
| GPU | Optional, required for visualization |

### Alternative Environments

If you don't have native Ubuntu:

- **WSL2 on Windows**: Full ROS 2 support with GUI via WSLg
- **Docker**: Containerized ROS 2 environment
- **Virtual Machine**: VMware or VirtualBox with Ubuntu 22.04

---

## 1.4 ROS 2 Humble Installation

ROS 2 Humble Hawksbill is the recommended Long Term Support (LTS) release, supported until May 2027.

### Installation Steps

1. **Set locale**:
```bash
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

2. **Add ROS 2 repository**:
```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

3. **Install ROS 2 Humble Desktop**:
```bash
sudo apt update
sudo apt install ros-humble-desktop
```

4. **Source the setup script**:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

5. **Install development tools**:
```bash
sudo apt install python3-colcon-common-extensions python3-rosdep
sudo rosdep init
rosdep update
```

### Verification

Verify your installation:

```bash
ros2 --version
# Expected output: ros2 0.x.x
```

---

## 1.5 Creating Your First ROS 2 Workspace

A **workspace** is a directory containing ROS 2 packages. The standard structure is:

```
ros2_ws/
├── src/           # Source packages
├── build/         # Build artifacts (generated)
├── install/       # Installed packages (generated)
└── log/           # Build logs (generated)
```

### Workspace Creation

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the empty workspace
colcon build

# Source the workspace
source install/setup.bash
```

### Creating a Python Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python humanoid_basics \
    --dependencies rclpy std_msgs
```

This creates:

```
humanoid_basics/
├── package.xml           # Package metadata
├── setup.py              # Python package setup
├── setup.cfg             # Package configuration
├── resource/             # Package marker
├── humanoid_basics/      # Python module
│   └── __init__.py
└── test/                 # Test files
```

---

## Exercises

### Exercise 1.1: Install ROS 2 Humble

**Objective**: Complete a full ROS 2 Humble installation on your system.

**Difficulty**: Beginner | **Estimated Time**: 30-45 minutes

#### Instructions

1. Follow the installation steps in Section 1.4
2. Verify the installation with `ros2 --version`
3. Test the environment with `ros2 run demo_nodes_cpp talker`

#### Expected Outcome

```bash
$ ros2 --version
ros2 0.10.x  # or similar

$ ros2 run demo_nodes_cpp talker
[INFO] [talker]: Publishing: 'Hello World: 1'
[INFO] [talker]: Publishing: 'Hello World: 2'
...
```

#### Verification Checklist

- [ ] `ros2` command is available in terminal
- [ ] Environment variables are set (check with `printenv | grep ROS`)
- [ ] Demo talker node runs without errors

#### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `ros2: command not found` | Setup not sourced | Run `source /opt/ros/humble/setup.bash` |
| GPG key errors | Repository key missing | Re-run the curl command for the key |
| Package not found | Repository not added | Verify `/etc/apt/sources.list.d/ros2.list` exists |

---

### Exercise 1.2: Create Your First ROS 2 Package

**Objective**: Create a Python package called `humanoid_basics` in a new workspace.

**Difficulty**: Beginner | **Estimated Time**: 20 minutes

#### Instructions

1. Create a workspace at `~/ros2_ws`
2. Create a Python package named `humanoid_basics`
3. Build the workspace with `colcon build`
4. Source the workspace and verify the package exists

#### Expected Outcome

```bash
$ ros2 pkg list | grep humanoid
humanoid_basics
```

#### Verification Checklist

- [ ] Workspace builds without errors
- [ ] Package appears in `ros2 pkg list`
- [ ] Package structure matches the expected layout

#### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `colcon: command not found` | Extensions not installed | `sudo apt install python3-colcon-common-extensions` |
| Build fails | Missing dependencies | Run `rosdep install --from-paths src --ignore-src -y` |
| Package not found after build | Workspace not sourced | Run `source install/setup.bash` |

---

### Exercise 1.3: Run the Talker/Listener Demo

**Objective**: Run the classic ROS 2 demonstration with two communicating nodes.

**Difficulty**: Beginner | **Estimated Time**: 15 minutes

#### Instructions

1. Open two terminal windows
2. In terminal 1, run the talker: `ros2 run demo_nodes_cpp talker`
3. In terminal 2, run the listener: `ros2 run demo_nodes_cpp listener`
4. Observe the communication between nodes
5. Use `ros2 topic list` and `ros2 topic echo` to inspect the topic

#### Expected Outcome

**Terminal 1 (Talker)**:
```
[INFO] [talker]: Publishing: 'Hello World: 1'
[INFO] [talker]: Publishing: 'Hello World: 2'
```

**Terminal 2 (Listener)**:
```
[INFO] [listener]: I heard: [Hello World: 1]
[INFO] [listener]: I heard: [Hello World: 2]
```

#### Verification Checklist

- [ ] Both nodes run without errors
- [ ] Listener receives messages from talker
- [ ] `ros2 topic list` shows `/chatter`
- [ ] `ros2 topic echo /chatter` displays messages

#### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Nodes can't find each other | Different ROS_DOMAIN_ID | Ensure same domain ID in both terminals |
| No output from listener | Topic mismatch | Verify both use the same topic with `ros2 topic list` |
| Permission denied | Cyclone DDS permissions | Check `/dev/shm` permissions |

---

## Summary

In this chapter, you learned:

- **ROS 2 is essential for humanoid robotics** due to its real-time capabilities, distributed architecture, and safety features
- **Core concepts** include nodes (computation units), topics (pub/sub), services (request/response), and actions (long-running tasks)
- **DDS middleware** provides Quality of Service, discovery, and security features
- **ROS 2 Humble** is the recommended LTS distribution for production humanoid development
- **Workspaces and packages** organize your ROS 2 code in a modular, reusable structure

### Key Commands Reference

| Command | Description |
|---------|-------------|
| `ros2 run <pkg> <node>` | Run a node from a package |
| `ros2 topic list` | List active topics |
| `ros2 topic echo <topic>` | Print topic messages |
| `ros2 node list` | List active nodes |
| `ros2 pkg list` | List installed packages |
| `colcon build` | Build all packages in workspace |

---

## References

[1] S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, "Robot Operating System 2: Design, architecture, and uses in the wild," *Science Robotics*, vol. 7, no. 66, 2022.

[2] Open Robotics, "ROS 2 Documentation: Humble Hawksbill," [Online]. Available: https://docs.ros.org/en/humble/. [Accessed: Dec. 2024].

[3] Object Management Group, "Data Distribution Service (DDS) Specification," Version 1.4, 2015.

[4] M. Quigley et al., "ROS: an open-source Robot Operating System," in *ICRA Workshop on Open Source Software*, 2009.
