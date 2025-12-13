---
id: ch02-nodes-topics
title: "Chapter 2: Nodes, Topics, Services & Launch Systems"
sidebar_position: 3
---

# Chapter 2: Nodes, Topics, Services & Launch Systems

**Estimated Time**: 5-6 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Create** custom publisher and subscriber nodes using rclpy
2. **Implement** ROS 2 services for synchronous request/response patterns
3. **Design** action servers and clients for long-running humanoid tasks
4. **Build** launch files to orchestrate multi-node systems
5. **Debug** node communication using ROS 2 CLI tools

---

## 2.1 Publisher/Subscriber Patterns

The publish/subscribe pattern is the foundation of ROS 2 communication. Publishers send messages to topics, and subscribers receive messages from those topics.

### Creating a Publisher Node

```python
#!/usr/bin/env python3
"""
joint_state_publisher.py
Publishes simulated joint states for a humanoid robot.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Create publisher with QoS depth of 10
        self.publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Timer callback at 50 Hz
        self.timer = self.create_timer(0.02, self.publish_joint_states)
        self.time = 0.0

        # Define humanoid joints
        self.joint_names = [
            'head_pan', 'head_tilt',
            'left_shoulder_pitch', 'left_shoulder_roll',
            'left_elbow', 'left_wrist',
            'right_shoulder_pitch', 'right_shoulder_roll',
            'right_elbow', 'right_wrist',
        ]

        self.get_logger().info('Joint state publisher initialized')

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Generate sinusoidal joint positions
        msg.position = [
            0.5 * math.sin(self.time + i * 0.5)
            for i in range(len(self.joint_names))
        ]
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)

        self.publisher.publish(msg)
        self.time += 0.02

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Subscriber Node

```python
#!/usr/bin/env python3
"""
joint_state_subscriber.py
Subscribes to joint states and logs them.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')

        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.get_logger().info('Joint state subscriber initialized')

    def joint_state_callback(self, msg):
        # Log first 3 joint positions
        positions = msg.position[:3]
        self.get_logger().info(
            f'Received joints: {[f"{p:.2f}" for p in positions]}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = JointStateSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Quality of Service (QoS) Profiles

QoS policies control message delivery behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Sensor data: best-effort, volatile (drop if slow)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    depth=5
)

# Control commands: reliable, transient local
control_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=10
)
```

---

## 2.2 Services and Actions

### Implementing a Service Server

Services are ideal for configuration changes or state queries:

```python
#!/usr/bin/env python3
"""
pose_service.py
Service to set the humanoid's target pose.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from geometry_msgs.msg import Pose

class PoseService(Node):
    def __init__(self):
        super().__init__('pose_service')

        self.srv = self.create_service(
            SetBool,
            'set_standing_pose',
            self.set_pose_callback
        )

        self.is_standing = False
        self.get_logger().info('Pose service ready')

    def set_pose_callback(self, request, response):
        if request.data:
            self.is_standing = True
            response.success = True
            response.message = 'Robot is now standing'
            self.get_logger().info('Standing pose activated')
        else:
            self.is_standing = False
            response.success = True
            response.message = 'Robot is now crouching'
            self.get_logger().info('Crouching pose activated')

        return response

def main(args=None):
    rclpy.init(args=args)
    node = PoseService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Implementing a Service Client

```python
#!/usr/bin/env python3
"""
pose_client.py
Client to request pose changes.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class PoseClient(Node):
    def __init__(self):
        super().__init__('pose_client')
        self.client = self.create_client(SetBool, 'set_standing_pose')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for pose service...')

    def send_request(self, stand: bool):
        request = SetBool.Request()
        request.data = stand

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

def main(args=None):
    rclpy.init(args=args)
    client = PoseClient()

    response = client.send_request(True)
    print(f'Response: {response.message}')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Actions for Long-Running Tasks

Actions are perfect for tasks like walking to a location:

```python
#!/usr/bin/env python3
"""
walk_action_server.py
Action server for walking commands.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci  # Placeholder action type
import time

class WalkActionServer(Node):
    def __init__(self):
        super().__init__('walk_action_server')

        self._action_server = ActionServer(
            self,
            Fibonacci,  # Replace with custom WalkToGoal action
            'walk_to_goal',
            self.execute_callback
        )

        self.get_logger().info('Walk action server ready')

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing walk goal...')

        feedback_msg = Fibonacci.Feedback()

        # Simulate walking progress
        for i in range(10):
            feedback_msg.partial_sequence = [i]
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Walking progress: {(i+1)*10}%')
            time.sleep(0.5)

        goal_handle.succeed()

        result = Fibonacci.Result()
        result.sequence = [0, 1, 1, 2, 3, 5, 8]
        return result

def main(args=None):
    rclpy.init(args=args)
    node = WalkActionServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 2.3 Launch Files

Launch files start multiple nodes with a single command.

### Python Launch File

```python
# launch/humanoid_bringup.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    use_sim = DeclareLaunchArgument(
        'use_sim',
        default_value='true',
        description='Use simulation time'
    )

    # Joint state publisher node
    joint_publisher = Node(
        package='humanoid_basics',
        executable='joint_state_publisher',
        name='joint_publisher',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim')
        }],
        output='screen'
    )

    # Joint state subscriber node
    joint_subscriber = Node(
        package='humanoid_basics',
        executable='joint_state_subscriber',
        name='joint_subscriber',
        output='screen'
    )

    # Robot state publisher for TF
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': '<robot name="humanoid"/>'
        }]
    )

    return LaunchDescription([
        use_sim,
        joint_publisher,
        joint_subscriber,
        robot_state_publisher,
    ])
```

### Running Launch Files

```bash
# Run from workspace root
ros2 launch humanoid_basics humanoid_bringup.launch.py

# With arguments
ros2 launch humanoid_basics humanoid_bringup.launch.py use_sim:=false
```

---

## 2.4 Debugging with ROS 2 CLI

### Essential Commands

```bash
# List all active nodes
ros2 node list

# Get node info
ros2 node info /joint_publisher

# List all topics
ros2 topic list

# Echo topic messages
ros2 topic echo /joint_states

# Get topic info (type, publishers, subscribers)
ros2 topic info /joint_states

# Publish a single message
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5}}"

# List all services
ros2 service list

# Call a service
ros2 service call /set_standing_pose std_srvs/srv/SetBool "{data: true}"

# View computation graph
ros2 run rqt_graph rqt_graph
```

---

## Exercises

### Exercise 2.1: Create a Custom Publisher Node

**Objective**: Create a publisher that sends humanoid arm positions.

**Difficulty**: Beginner | **Estimated Time**: 30 minutes

#### Instructions

1. Create a new Python file `arm_publisher.py`
2. Publish `Float64MultiArray` messages to `/arm_positions`
3. Send 6 joint values (shoulder, elbow, wrist for each arm)
4. Publish at 20 Hz

#### Expected Outcome

```bash
$ ros2 topic echo /arm_positions
data: [0.1, 0.2, 0.0, -0.1, -0.2, 0.0]
---
```

#### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No messages published | Timer not created | Verify `create_timer()` call |
| Wrong message type | Import error | Check `from std_msgs.msg import Float64MultiArray` |

---

### Exercise 2.2: Implement a Service Server/Client

**Objective**: Create a service to toggle the humanoid's arm control mode.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Create a service server that accepts `SetBool` requests
2. `True` = position control mode, `False` = velocity control mode
3. Create a client that toggles the mode
4. Log the current mode in the server

#### Verification

```bash
$ ros2 service call /set_control_mode std_srvs/srv/SetBool "{data: true}"
requester: making request
response: success=True message='Position control mode activated'
```

---

### Exercise 2.3: Build a Multi-Node Launch File

**Objective**: Create a launch file that starts all humanoid control nodes.

**Difficulty**: Intermediate | **Estimated Time**: 30 minutes

#### Instructions

1. Create `humanoid_control.launch.py`
2. Start the joint publisher, subscriber, and pose service
3. Add a launch argument for robot name
4. Verify all nodes start correctly

#### Expected Outcome

```bash
$ ros2 launch humanoid_basics humanoid_control.launch.py
[joint_publisher]: Joint state publisher initialized
[joint_subscriber]: Joint state subscriber initialized
[pose_service]: Pose service ready
```

---

### Exercise 2.4: Debug Node Communication

**Objective**: Use ROS 2 CLI tools to inspect and debug a running system.

**Difficulty**: Beginner | **Estimated Time**: 20 minutes

#### Instructions

1. Start the humanoid launch file
2. List all nodes and verify they're running
3. Echo the `/joint_states` topic
4. Use `rqt_graph` to visualize the node graph
5. Call the pose service and verify response

#### Verification Checklist

- [ ] All expected nodes appear in `ros2 node list`
- [ ] Topic messages are visible with `ros2 topic echo`
- [ ] rqt_graph shows correct connections
- [ ] Service calls return expected responses

---

## Summary

In this chapter, you learned:

- **Publishers and subscribers** form the backbone of ROS 2 communication
- **QoS profiles** control reliability, durability, and other delivery guarantees
- **Services** provide synchronous request/response for configuration and queries
- **Actions** handle long-running tasks with feedback and cancellation
- **Launch files** orchestrate multi-node systems with configurable parameters
- **CLI tools** enable powerful debugging and introspection

### Communication Pattern Selection Guide

| Pattern | Use When | Example |
|---------|----------|---------|
| Topic | Continuous data streams | Sensor readings, joint states |
| Service | Quick configuration/query | Set mode, get status |
| Action | Long tasks with feedback | Walk to goal, pick object |

---

## References

[1] Open Robotics, "ROS 2 Concepts: Nodes," [Online]. Available: https://docs.ros.org/en/humble/Concepts/About-Nodes.html.

[2] Open Robotics, "ROS 2 Tutorials: Writing a Simple Publisher and Subscriber," [Online]. Available: https://docs.ros.org/en/humble/Tutorials/.

[3] Open Robotics, "ROS 2 Launch System," [Online]. Available: https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/.

[4] Object Management Group, "DDS Quality of Service Policies," DDS Specification v1.4, 2015.
