---
id: ch03-urdf-kinematics
title: "Chapter 3: URDF & Modeling Humanoid Kinematics"
sidebar_position: 4
---

# Chapter 3: URDF & Modeling Humanoid Kinematics

**Estimated Time**: 5-6 hours | **Exercises**: 3

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand** the URDF format and its role in robot description
2. **Define** links, joints, and their properties for humanoid robots
3. **Model** kinematic chains for arms, legs, and torso
4. **Create** visual and collision geometries for robot components
5. **Visualize** URDF models in RViz2

---

## 3.1 Introduction to URDF

The **Unified Robot Description Format (URDF)** is an XML specification for describing robot models. It defines:

- **Links**: Rigid bodies with visual and physical properties
- **Joints**: Connections between links with motion constraints
- **Transmissions**: Actuator-to-joint mappings
- **Sensors**: Cameras, LiDAR, IMUs (via Gazebo plugins)

### Why URDF for Humanoids?

Humanoid robots have complex kinematic structures:

```
                    HEAD
                     │
              ┌──────┴──────┐
              │    TORSO    │
              └──────┬──────┘
         ┌───────────┼───────────┐
    LEFT ARM         │      RIGHT ARM
         │           │           │
    ┌────┴────┐      │     ┌────┴────┐
    │Shoulder │      │     │Shoulder │
    ├─────────┤      │     ├─────────┤
    │  Elbow  │      │     │  Elbow  │
    ├─────────┤      │     ├─────────┤
    │  Wrist  │      │     │  Wrist  │
    └─────────┘      │     └─────────┘
                     │
              ┌──────┴──────┐
              │    PELVIS   │
              └──────┬──────┘
         ┌───────────┼───────────┐
    LEFT LEG         │      RIGHT LEG
         │           │           │
    ┌────┴────┐      │     ┌────┴────┐
    │   Hip   │      │     │   Hip   │
    ├─────────┤      │     ├─────────┤
    │  Knee   │      │     │  Knee   │
    ├─────────┤      │     ├─────────┤
    │  Ankle  │      │     │  Ankle  │
    └─────────┘      │     └─────────┘
```

---

## 3.2 URDF Fundamentals

### Basic Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <!-- Visual, collision, and inertial properties -->
  </link>

  <!-- Joints connect links -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
  </joint>

  <link name="torso">
    <!-- Torso properties -->
  </link>
</robot>
```

### Link Definition

A link has three main components:

```xml
<link name="upper_arm">
  <!-- Visual geometry (for rendering) -->
  <visual>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.04" length="0.30"/>
    </geometry>
    <material name="skin">
      <color rgba="0.9 0.7 0.6 1"/>
    </material>
  </visual>

  <!-- Collision geometry (for physics) -->
  <collision>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.045" length="0.30"/>
    </geometry>
  </collision>

  <!-- Inertial properties (for dynamics) -->
  <inertial>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <mass value="2.5"/>
    <inertia ixx="0.02" ixy="0" ixz="0"
             iyy="0.02" iyz="0"
             izz="0.005"/>
  </inertial>
</link>
```

### Joint Types

| Type | Description | DoF | Use Case |
|------|-------------|-----|----------|
| `revolute` | Rotation with limits | 1 | Elbow, knee |
| `continuous` | Unlimited rotation | 1 | Wheels |
| `prismatic` | Linear translation | 1 | Telescoping |
| `fixed` | No motion | 0 | Sensor mounts |
| `floating` | 6-DoF free motion | 6 | Base link |
| `planar` | 2D plane motion | 3 | Mobile bases |

### Joint Definition

```xml
<joint name="left_elbow" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_forearm"/>

  <!-- Joint origin relative to parent -->
  <origin xyz="0 0 0.30" rpy="0 0 0"/>

  <!-- Rotation/translation axis -->
  <axis xyz="0 1 0"/>

  <!-- Joint limits -->
  <limit lower="0.0" upper="2.5"
         effort="50.0" velocity="2.0"/>

  <!-- Dynamics (damping and friction) -->
  <dynamics damping="0.5" friction="0.1"/>
</joint>
```

---

## 3.3 Humanoid Kinematic Chains

### Torso and Head

```xml
<!-- Torso -->
<link name="torso">
  <visual>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <geometry>
      <box size="0.3 0.2 0.4"/>
    </geometry>
    <material name="body"/>
  </visual>
  <collision>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <geometry>
      <box size="0.3 0.2 0.4"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="15.0"/>
    <inertia ixx="0.5" ixy="0" ixz="0"
             iyy="0.4" iyz="0" izz="0.3"/>
  </inertial>
</link>

<!-- Neck joint -->
<joint name="neck_pan" type="revolute">
  <parent link="torso"/>
  <child link="head"/>
  <origin xyz="0 0 0.4" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="20" velocity="1.5"/>
</joint>

<!-- Head -->
<link name="head">
  <visual>
    <geometry>
      <sphere radius="0.12"/>
    </geometry>
    <material name="head"/>
  </visual>
</link>
```

### Arm Kinematic Chain

A typical humanoid arm has 6-7 degrees of freedom:

```xml
<!-- Shoulder pitch -->
<joint name="left_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_shoulder_link"/>
  <origin xyz="0.15 0.1 0.35" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-3.14" upper="1.0" effort="80" velocity="2.0"/>
</joint>

<!-- Shoulder roll -->
<joint name="left_shoulder_roll" type="revolute">
  <parent link="left_shoulder_link"/>
  <child link="left_upper_arm"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-0.5" upper="3.14" effort="80" velocity="2.0"/>
</joint>

<!-- Shoulder yaw -->
<joint name="left_shoulder_yaw" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_upper_arm_yaw"/>
  <origin xyz="0 0 0.15" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="60" velocity="2.0"/>
</joint>

<!-- Elbow -->
<joint name="left_elbow" type="revolute">
  <parent link="left_upper_arm_yaw"/>
  <child link="left_forearm"/>
  <origin xyz="0 0 0.15" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0" upper="2.5" effort="50" velocity="2.5"/>
</joint>

<!-- Wrist pitch -->
<joint name="left_wrist_pitch" type="revolute">
  <parent link="left_forearm"/>
  <child link="left_wrist_pitch_link"/>
  <origin xyz="0 0 0.25" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="20" velocity="3.0"/>
</joint>

<!-- Wrist roll -->
<joint name="left_wrist_roll" type="revolute">
  <parent link="left_wrist_pitch_link"/>
  <child link="left_hand"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-3.14" upper="3.14" effort="15" velocity="4.0"/>
</joint>
```

### Leg Kinematic Chain

Each leg typically has 6 degrees of freedom:

```xml
<!-- Hip yaw -->
<joint name="left_hip_yaw" type="revolute">
  <parent link="pelvis"/>
  <child link="left_hip_yaw_link"/>
  <origin xyz="0 0.1 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-0.5" upper="0.5" effort="100" velocity="1.5"/>
</joint>

<!-- Hip roll -->
<joint name="left_hip_roll" type="revolute">
  <parent link="left_hip_yaw_link"/>
  <child link="left_hip_roll_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-0.5" upper="0.5" effort="100" velocity="1.5"/>
</joint>

<!-- Hip pitch -->
<joint name="left_hip_pitch" type="revolute">
  <parent link="left_hip_roll_link"/>
  <child link="left_thigh"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.5" upper="0.5" effort="150" velocity="2.0"/>
</joint>

<!-- Knee -->
<joint name="left_knee" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_shin"/>
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0" upper="2.5" effort="150" velocity="2.5"/>
</joint>

<!-- Ankle pitch -->
<joint name="left_ankle_pitch" type="revolute">
  <parent link="left_shin"/>
  <child link="left_ankle_pitch_link"/>
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.0" upper="0.7" effort="80" velocity="2.0"/>
</joint>

<!-- Ankle roll -->
<joint name="left_ankle_roll" type="revolute">
  <parent link="left_ankle_pitch_link"/>
  <child link="left_foot"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-0.5" upper="0.5" effort="60" velocity="2.0"/>
</joint>
```

---

## 3.4 Xacro for Modular URDFs

Xacro (XML Macros) makes URDFs more maintainable:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid">

  <!-- Properties -->
  <xacro:property name="arm_length" value="0.30"/>
  <xacro:property name="arm_radius" value="0.04"/>

  <!-- Arm macro -->
  <xacro:macro name="arm" params="prefix reflect">
    <link name="${prefix}_upper_arm">
      <visual>
        <geometry>
          <cylinder radius="${arm_radius}" length="${arm_length}"/>
        </geometry>
      </visual>
    </link>

    <joint name="${prefix}_shoulder" type="revolute">
      <parent link="torso"/>
      <child link="${prefix}_upper_arm"/>
      <origin xyz="${reflect * 0.15} 0 0.3"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Instantiate left and right arms -->
  <xacro:arm prefix="left" reflect="1"/>
  <xacro:arm prefix="right" reflect="-1"/>

</robot>
```

### Processing Xacro Files

```bash
# Convert xacro to URDF
xacro humanoid.urdf.xacro > humanoid.urdf

# Or use in launch file
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command

robot_description = ParameterValue(
    Command(['xacro ', 'path/to/humanoid.urdf.xacro']),
    value_type=str
)
```

---

## 3.5 Visualization in RViz2

### Robot State Publisher

The `robot_state_publisher` broadcasts TF transforms:

```python
# Launch file snippet
from launch_ros.actions import Node

robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'robot_description': robot_description}]
)
```

### Joint State Publisher GUI

For interactive joint control:

```bash
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

### RViz2 Configuration

```yaml
# config/humanoid.rviz
Panels:
  - Class: rviz_common/Displays
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Description Topic: /robot_description
      TF Prefix: ""
    - Class: rviz_default_plugins/TF
      Name: TF
      Show Names: true
  Global Options:
    Fixed Frame: base_link
```

---

## Exercises

### Exercise 3.1: Create a Simple Robot Arm URDF

**Objective**: Create a 3-DoF robot arm with shoulder, elbow, and wrist joints.

**Difficulty**: Beginner | **Estimated Time**: 45 minutes

#### Instructions

1. Create `simple_arm.urdf` with:
   - Base link (fixed)
   - Upper arm (revolute shoulder)
   - Forearm (revolute elbow)
   - Hand (revolute wrist)
2. Use cylinder geometry for arm segments
3. Set appropriate joint limits

#### Expected Outcome

The arm should be visualizable in RViz2 with movable joints.

#### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Links not visible | Missing visual geometry | Add `<visual>` elements |
| TF errors | Missing joint connections | Verify parent/child links |
| Joints don't move | Wrong joint type | Use `revolute` not `fixed` |

---

### Exercise 3.2: Build a Humanoid Torso

**Objective**: Create a torso with head and both arms.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Create `humanoid_torso.urdf.xacro`
2. Define torso, neck, and head links
3. Use Xacro macros for left/right arms
4. Add proper inertial properties

#### Verification

```bash
# Check URDF validity
check_urdf humanoid_torso.urdf

# Expected output:
robot name is: humanoid_torso
---------- Successfully Parsed XML ---------------
```

---

### Exercise 3.3: Visualize in RViz2

**Objective**: Create a complete visualization setup for the humanoid model.

**Difficulty**: Beginner | **Estimated Time**: 30 minutes

#### Instructions

1. Create a launch file that starts:
   - robot_state_publisher
   - joint_state_publisher_gui
   - RViz2 with saved config
2. Verify all joints are controllable
3. Save the RViz configuration

#### Verification Checklist

- [ ] Robot model appears in RViz2
- [ ] TF tree shows all links
- [ ] Joint sliders control the model
- [ ] No TF errors in terminal

---

## Summary

In this chapter, you learned:

- **URDF** is the standard format for describing robot models in ROS 2
- **Links** define rigid bodies with visual, collision, and inertial properties
- **Joints** connect links and define motion constraints
- **Humanoid kinematic chains** typically include 6-7 DoF arms and 6 DoF legs
- **Xacro** enables modular, reusable URDF components
- **RViz2** provides 3D visualization of robot models

---

## References

[1] Open Robotics, "URDF Specification," [Online]. Available: http://wiki.ros.org/urdf/XML.

[2] Open Robotics, "Xacro Documentation," [Online]. Available: http://wiki.ros.org/xacro.

[3] S. Chitta et al., "MoveIt!: An Introduction," in *Robot Operating System (ROS): The Complete Reference*, Springer, 2016.

[4] B. Siciliano and O. Khatib, Eds., *Springer Handbook of Robotics*, 2nd ed. Springer, 2016.
