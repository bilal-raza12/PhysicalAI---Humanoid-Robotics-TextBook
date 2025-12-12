---
id: glossary
title: Glossary
sidebar_position: 4
---

# Glossary

This glossary defines key terms used throughout the textbook. Terms are organized alphabetically.

---

## A

### Action (ROS 2)
A communication pattern in ROS 2 for long-running tasks that provides feedback during execution and can be canceled. Actions are used for tasks like navigation where progress updates are needed.

### Affordance
The possible actions that an object offers to an agent. In robotics, understanding affordances helps robots determine how to interact with objects (e.g., a handle affords grasping).

---

## B

### Behavior Tree
A hierarchical structure used to model complex robot behaviors through composition of simpler actions. Commonly used for decision-making in autonomous systems.

### Bipedal Locomotion
Movement using two legs, as seen in humanoid robots. Requires complex balance control and gait planning.

---

## C

### Closed-Loop Control
A control system where sensor feedback is continuously used to adjust outputs. Contrast with open-loop control which operates without feedback.

### Collision Detection
The computational process of determining when two or more objects intersect in simulation or the real world.

### Costmap
A grid representation used in navigation that assigns traversal costs to different areas, helping robots plan optimal paths.

---

## D

### Degrees of Freedom (DoF)
The number of independent parameters that define a system's configuration. A humanoid robot typically has 20-40+ DoF.

### Digital Twin
A virtual replica of a physical system that mirrors its behavior in real-time or through simulation.

### Domain Randomization
A technique for generating diverse training data by randomly varying simulation parameters (lighting, textures, physics) to improve model generalization.

---

## E

### End Effector
The device at the end of a robotic arm that interacts with the environment, such as a gripper or tool.

### Embodied AI
Artificial intelligence that operates in and learns from a physical or simulated body, as opposed to disembodied AI that only processes text or images.

---

## F

### Forward Kinematics
Computing the position and orientation of a robot's end effector given the joint angles.

### Frame (Coordinate)
A coordinate system attached to a point in space, used to describe positions and orientations in robotics.

---

## G

### Gait
The pattern of leg movements used for walking. Humanoid robots require sophisticated gait generation for stable bipedal locomotion.

### Gazebo
An open-source robotics simulator that provides accurate physics simulation and sensor modeling.

### Grounding (Language)
The process of connecting abstract language concepts to physical actions or objects in the real world.

---

## H

### Homogeneous Coordinates
A coordinate system that uses an extra dimension to enable translation to be represented as matrix multiplication, simplifying 3D transformations.

### Human-Robot Interaction (HRI)
The study and design of interactions between humans and robots, encompassing communication, collaboration, and safety.

---

## I

### IMU (Inertial Measurement Unit)
A sensor that measures acceleration, angular velocity, and sometimes magnetic field to track orientation and motion.

### Inverse Kinematics (IK)
Computing the joint angles needed to achieve a desired end effector position and orientation.

### Isaac Sim
NVIDIA's photorealistic robotics simulator built on the Omniverse platform.

---

## J

### Joint
A connection between two links in a robot that allows relative motion. Types include revolute (rotation), prismatic (sliding), and fixed.

---

## K

### Kinematic Chain
A series of links connected by joints, forming the mechanical structure of a robot.

---

## L

### LiDAR (Light Detection and Ranging)
A sensor that uses laser pulses to measure distances and create 3D maps of the environment.

### Link
A rigid body component of a robot connected to other links via joints.

### LLM (Large Language Model)
AI models trained on vast text data that can understand and generate human language, used in robotics for task planning and natural language interfaces.

### Localization
Determining the position and orientation of a robot within a known environment.

---

## M

### Manipulation
The use of robot arms and end effectors to interact with objects in the environment.

### Mapping
Creating a representation of the environment from sensor data.

### MoveIt
A ROS-based motion planning framework commonly used for robotic manipulation.

---

## N

### Nav2
The ROS 2 Navigation Stack, providing path planning, localization, and obstacle avoidance capabilities.

### Node (ROS 2)
A process that performs computation in ROS 2. Nodes communicate via topics, services, and actions.

---

## O

### Odometry
Estimating a robot's position change over time using sensor data, typically from wheel encoders or visual features.

### Omniverse
NVIDIA's platform for 3D simulation and collaboration, the foundation for Isaac Sim.

---

## P

### Perception
The ability of a robot to interpret sensor data to understand its environment.

### Physics Engine
Software that simulates physical interactions including gravity, collisions, friction, and dynamics.

### Publisher (ROS 2)
A node that sends messages to a topic for other nodes to receive.

---

## Q

### Quaternion
A mathematical representation of 3D rotation that avoids gimbal lock, commonly used in robotics.

---

## R

### ROS (Robot Operating System)
A middleware framework providing tools, libraries, and conventions for building robot applications.

### ROS 2
The second generation of ROS, designed for real-time, secure, and distributed robotics applications.

---

## S

### SDF (Simulation Description Format)
An XML format for describing robots and environments in Gazebo simulations.

### SLAM (Simultaneous Localization and Mapping)
Building a map of an unknown environment while simultaneously tracking the robot's position within it.

### Subscriber (ROS 2)
A node that receives messages from a topic.

### Synthetic Data
Computer-generated data used to train machine learning models, often from simulation.

---

## T

### tf2
The ROS 2 transform library that maintains relationships between coordinate frames over time.

### Topic (ROS 2)
A named channel for publishing and subscribing to messages between nodes.

### Transformation Matrix
A 4x4 matrix representing rotation and translation in 3D space.

---

## U

### URDF (Unified Robot Description Format)
An XML format for describing the physical structure of a robot including links, joints, and visual properties.

---

## V

### Visual Odometry
Estimating motion from camera images by tracking visual features.

### VLA (Vision-Language-Action)
Models that process visual input, understand natural language, and generate robot actions.

### VSLAM (Visual SLAM)
SLAM using camera images as the primary sensor input.

---

## W

### Whisper
OpenAI's automatic speech recognition model, used for voice command interfaces.

### World Frame
A fixed reference coordinate frame, typically representing the global environment.

### Workspace
In ROS 2, a directory containing packages for building and running robot applications.

---

## Z

### Zero Moment Point (ZMP)
A point where the sum of all moments due to gravity and inertia is zero, critical for bipedal balance control.

---

_For additional terms, consult the IEEE Standard Glossary of Software Engineering Terminology and ROS 2 documentation._
