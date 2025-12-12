---
id: index
title: "Module 1: The Robotic Nervous System (ROS 2)"
sidebar_position: 1
---

# Module 1: The Robotic Nervous System (ROS 2)

**Estimated Time**: 20 hours | **Chapters**: 4 | **Exercises**: 13+

## Overview

Just as the human nervous system coordinates signals between the brain, sensors, and muscles, the Robot Operating System (ROS 2) provides the communication infrastructure that connects every component of a modern robot. In this module, you'll learn how ROS 2 enables humanoid robots to perceive their environment, make decisions, and execute coordinated movements.

## Learning Outcomes

By completing this module, you will be able to:

- **Understand ROS 2 architecture** and explain how nodes, topics, services, and actions work together
- **Model humanoid robots** using URDF to describe kinematic chains, joints, and links
- **Build communication systems** using publishers, subscribers, and services
- **Integrate Python AI agents** with rclpy for intelligent robot behavior
- **Create launch files** for complex multi-node systems

## Prerequisites

Before starting this module, ensure you have:

- [ ] Python programming proficiency
- [ ] Ubuntu 22.04 installed (native, WSL2, or VM)
- [ ] Completed the [Prerequisites Checklist](/docs/prerequisites)

## Chapters

### Chapter 1: Introduction to ROS 2 for Humanoids

Learn the fundamentals of ROS 2 and why it's the foundation for modern robotics. Set up your development environment and create your first ROS 2 package.

**Topics**: ROS 2 history, architecture overview, workspace setup, package creation

### Chapter 2: Nodes, Topics, Services & Launch Systems

Master the communication patterns that allow robot components to exchange data and coordinate actions in real-time.

**Topics**: Publishers, subscribers, services, actions, launch files, parameter servers

### Chapter 3: URDF & Modeling Humanoid Kinematics

Create mathematical models of humanoid robots that describe their physical structure, joint constraints, and coordinate frames.

**Topics**: URDF syntax, links and joints, visual/collision geometry, RViz2 visualization

### Chapter 4: Integrating Python AI Agents with rclpy

Connect AI decision-making systems to ROS 2 using Python, enabling robots to exhibit intelligent behavior.

**Topics**: rclpy fundamentals, behavior trees, external API integration, real-time control

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Node** | An independent process that performs computation |
| **Topic** | A named bus for publishing/subscribing messages |
| **Service** | A synchronous request/response communication pattern |
| **Action** | An asynchronous goal-oriented communication pattern |
| **URDF** | XML format for describing robot physical structure |

## Software Requirements

| Software | Version | Installation |
|----------|---------|--------------|
| ROS 2 | Humble | Chapter 1 |
| Python | 3.10+ | Pre-installed |
| colcon | Latest | Chapter 1 |
| RViz2 | Humble | With ROS 2 |

## What's Next

After completing this module, you'll have a solid foundation in ROS 2 and humanoid robot modeling. In **Module 2: The Digital Twin**, you'll bring your robot models to life in physics simulators like Gazebo and Unity.

---

_Ready to begin? Start with [Chapter 1: Introduction to ROS 2 for Humanoids](./ch01-intro-ros2)._
