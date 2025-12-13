---
id: ch06-gazebo-twin
title: "Chapter 6: Building a Humanoid Digital Twin in Gazebo"
sidebar_position: 3
---

# Chapter 6: Building a Humanoid Digital Twin in Gazebo

**Estimated Time**: 6-7 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Convert** URDF models to SDF for Gazebo simulation
2. **Create** custom Gazebo worlds for humanoid testing
3. **Configure** the ROS 2-Gazebo bridge for communication
4. **Spawn** and control humanoid robots in simulation
5. **Debug** common simulation issues

---

## 6.1 URDF to SDF Conversion

Gazebo uses SDF (Simulation Description Format) internally. While it can load URDF files, understanding SDF enables more control.

### Automatic Conversion

```bash
# Convert URDF to SDF
gz sdf -p humanoid.urdf > humanoid.sdf
```

### Key Differences

| Feature | URDF | SDF |
|---------|------|-----|
| Sensors | Limited (Gazebo plugins) | Native support |
| Worlds | Not supported | Full world definition |
| Physics | Basic inertia | Advanced physics config |
| Nested models | Not supported | Fully supported |

### SDF Model Structure

```xml
<?xml version="1.0"?>
<sdf version="1.9">
  <model name="humanoid">
    <static>false</static>

    <link name="torso">
      <pose>0 0 1.0 0 0 0</pose>

      <inertial>
        <mass>15.0</mass>
        <inertia>
          <ixx>0.5</ixx>
          <iyy>0.4</iyy>
          <izz>0.3</izz>
        </inertia>
      </inertial>

      <visual name="torso_visual">
        <geometry>
          <box><size>0.3 0.2 0.4</size></box>
        </geometry>
        <material>
          <ambient>0.3 0.3 0.8 1</ambient>
          <diffuse>0.5 0.5 0.9 1</diffuse>
        </material>
      </visual>

      <collision name="torso_collision">
        <geometry>
          <box><size>0.3 0.2 0.4</size></box>
        </geometry>
      </collision>
    </link>

    <joint name="left_shoulder" type="revolute">
      <parent>torso</parent>
      <child>left_upper_arm</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-3.14</lower>
          <upper>1.0</upper>
          <effort>80</effort>
          <velocity>2.0</velocity>
        </limit>
      </axis>
    </joint>

  </model>
</sdf>
```

---

## 6.2 Creating Gazebo Worlds

### World File Structure

```xml
<?xml version="1.0"?>
<sdf version="1.9">
  <world name="humanoid_world">

    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="ground">
        <collision name="collision">
          <geometry>
            <plane><normal>0 0 1</normal></plane>
          </geometry>
          <surface>
            <friction>
              <ode><mu>1.0</mu><mu2>1.0</mu2></ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane><normal>0 0 1</normal><size>100 100</size></plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
          </material>
        </visual>
      </link>
    </model>

    <!-- Include humanoid robot -->
    <include>
      <uri>model://humanoid</uri>
      <pose>0 0 1.0 0 0 0</pose>
    </include>

  </world>
</sdf>
```

### Adding Objects

```xml
<!-- Table -->
<model name="table">
  <static>true</static>
  <pose>1.5 0 0 0 0 0</pose>
  <link name="table_link">
    <collision name="collision">
      <geometry>
        <box><size>1.0 0.6 0.8</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>1.0 0.6 0.8</size></box>
      </geometry>
      <material>
        <ambient>0.6 0.4 0.2 1</ambient>
      </material>
    </visual>
  </link>
</model>

<!-- Graspable object -->
<model name="red_cube">
  <pose>1.5 0 0.85 0 0 0</pose>
  <link name="cube_link">
    <inertial>
      <mass>0.1</mass>
    </inertial>
    <collision name="collision">
      <geometry>
        <box><size>0.05 0.05 0.05</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>0.05 0.05 0.05</size></box>
      </geometry>
      <material>
        <ambient>0.8 0.1 0.1 1</ambient>
      </material>
    </visual>
  </link>
</model>
```

---

## 6.3 ROS 2-Gazebo Bridge

The `ros_gz_bridge` package connects Gazebo topics to ROS 2.

### Bridge Configuration

```yaml
# bridge_config.yaml
- topic_name: "/joint_states"
  ros_type_name: "sensor_msgs/msg/JointState"
  gz_type_name: "gz.msgs.Model"
  direction: GZ_TO_ROS

- topic_name: "/cmd_vel"
  ros_type_name: "geometry_msgs/msg/Twist"
  gz_type_name: "gz.msgs.Twist"
  direction: ROS_TO_GZ

- topic_name: "/camera/image_raw"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "gz.msgs.Image"
  direction: GZ_TO_ROS
```

### Launch File Integration

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('humanoid_sim')

    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={'gz_args': '-r humanoid_world.sdf'}.items()
    )

    # Spawn robot
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'humanoid',
            '-file', os.path.join(pkg_share, 'models', 'humanoid.sdf'),
            '-x', '0', '-y', '0', '-z', '1.0'
        ]
    )

    # ROS-Gazebo bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    return LaunchDescription([
        gazebo,
        spawn,
        bridge,
        robot_state_pub,
    ])
```

---

## 6.4 Joint Control in Gazebo

### Position Control Plugin

```xml
<plugin filename="gz-sim-joint-position-controller-system"
        name="gz::sim::systems::JointPositionController">
  <joint_name>left_elbow</joint_name>
  <topic>/humanoid/left_elbow/cmd_pos</topic>
  <p_gain>100</p_gain>
  <i_gain>0.1</i_gain>
  <d_gain>10</d_gain>
</plugin>
```

### Effort Control

```python
#!/usr/bin/env python3
"""
joint_controller.py
Send joint commands to Gazebo via ROS 2.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')

        # Create publishers for each joint
        self.joint_pubs = {}
        joint_names = ['left_shoulder', 'left_elbow', 'left_wrist']

        for joint in joint_names:
            topic = f'/humanoid/{joint}/cmd_pos'
            self.joint_pubs[joint] = self.create_publisher(Float64, topic, 10)

        self.timer = self.create_timer(0.1, self.control_callback)
        self.t = 0.0

    def control_callback(self):
        import math

        # Sinusoidal motion
        pos = Float64()
        pos.data = 0.5 * math.sin(self.t)

        for pub in self.joint_pubs.values():
            pub.publish(pos)

        self.t += 0.1

def main(args=None):
    rclpy.init(args=args)
    node = JointController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Exercises

### Exercise 6.1: Convert URDF to SDF

**Objective**: Convert your humanoid URDF to SDF format.

**Difficulty**: Beginner | **Estimated Time**: 30 minutes

#### Instructions

1. Use `gz sdf -p` to convert your URDF
2. Verify the SDF structure
3. Add Gazebo-specific elements (materials, friction)
4. Test loading in Gazebo

#### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Conversion fails | Invalid URDF | Run `check_urdf` first |
| Missing materials | URDF materials not converted | Add SDF materials manually |

---

### Exercise 6.2: Create Custom World

**Objective**: Build a test environment with obstacles.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

---

### Exercise 6.3: Configure ros_gz_bridge

**Objective**: Set up bidirectional communication between ROS 2 and Gazebo.

**Difficulty**: Intermediate | **Estimated Time**: 30 minutes

---

### Exercise 6.4: Spawn and Control Humanoid

**Objective**: Launch humanoid in Gazebo and control joint positions.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

---

## Summary

In this chapter, you learned:

- **SDF format** provides more control than URDF for Gazebo
- **World files** define environments, physics, and lighting
- **ros_gz_bridge** connects Gazebo and ROS 2 topics
- **Joint controllers** enable position/velocity/effort control
- **Launch files** orchestrate the complete simulation stack

---

## References

[1] Open Robotics, "Gazebo Sim Documentation," [Online]. Available: https://gazebosim.org/docs.

[2] Open Robotics, "ros_gz Integration," [Online]. Available: https://github.com/gazebosim/ros_gz.

[3] SDF Format Specification, [Online]. Available: http://sdformat.org/spec.
