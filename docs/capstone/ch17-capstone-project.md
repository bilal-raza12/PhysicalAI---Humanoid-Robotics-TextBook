---
id: ch17-capstone-project
title: "Chapter 17: Capstone Project - Building an AI-Powered Humanoid Assistant"
sidebar_position: 2
---

# Chapter 17: Capstone Project - Building an AI-Powered Humanoid Assistant

**Estimated Time**: 20-30 hours | **Project Duration**: 2-4 weeks

## Project Overview

In this capstone project, you will integrate all concepts from the textbook to build a complete AI-powered humanoid assistant capable of natural language interaction, autonomous navigation, object manipulation, and adaptive learning.

### Project Goals

By completing this project, you will demonstrate mastery of:

1. **ROS 2 system architecture** for humanoid control
2. **Digital twin simulation** using Gazebo and Isaac Sim
3. **Reinforcement learning** for locomotion policies
4. **Vision-Language-Action models** for task understanding
5. **System integration** and real-world deployment

---

## Project Specification

### The Humanoid Assistant

You will build "ARIA" (Adaptive Robotic Intelligent Assistant), a humanoid robot capable of:

```
┌─────────────────────────────────────────────────────────┐
│                    ARIA Capabilities                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Natural Interaction         Autonomous Operation       │
│   ───────────────────         ────────────────────      │
│   • Voice commands            • Navigate environments   │
│   • Gesture recognition       • Avoid obstacles         │
│   • Conversational AI         • Find objects            │
│   • Emotion awareness         • Plan efficient paths    │
│                                                          │
│   Object Manipulation         Learning & Adaptation     │
│   ───────────────────         ─────────────────────     │
│   • Pick and place            • Learn from demos        │
│   • Handover to humans        • Improve with feedback   │
│   • Tool use                  • Remember preferences    │
│   • Careful handling          • Transfer to new tasks   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Scenario: Home Assistant

ARIA operates in a home environment helping with daily tasks:

1. **Morning Routine**: Wake up, navigate to kitchen, prepare items for breakfast
2. **Object Retrieval**: "ARIA, can you bring me my book from the living room?"
3. **Visitor Greeting**: Detect visitors, navigate to door, greet appropriately
4. **Cleanup Tasks**: Identify misplaced items, return them to proper locations
5. **Human Assistance**: Assist elderly or disabled users with reaching items

---

## Phase 1: Foundation Setup (Week 1)

### 1.1 ROS 2 Workspace Setup

Create the project workspace:

```bash
# Create workspace
mkdir -p ~/aria_ws/src
cd ~/aria_ws/src

# Create packages
ros2 pkg create --build-type ament_python aria_bringup
ros2 pkg create --build-type ament_python aria_perception
ros2 pkg create --build-type ament_python aria_navigation
ros2 pkg create --build-type ament_python aria_manipulation
ros2 pkg create --build-type ament_python aria_planning
ros2 pkg create --build-type ament_python aria_interfaces

# Build
cd ~/aria_ws
colcon build
source install/setup.bash
```

### 1.2 Robot Description

Create the humanoid URDF:

```xml
<!-- aria_description/urdf/aria.urdf.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="aria">

  <!-- Properties -->
  <xacro:property name="body_mass" value="50.0"/>
  <xacro:property name="arm_length" value="0.6"/>
  <xacro:property name="leg_length" value="0.8"/>

  <!-- Materials -->
  <material name="white">
    <color rgba="0.9 0.9 0.9 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.2 0.4 0.8 1.0"/>
  </material>

  <!-- Base/Torso -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${body_mass * 0.4}"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="head_pan" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
  </joint>

  <!-- Include arms and legs (defined in separate xacro files) -->
  <xacro:include filename="$(find aria_description)/urdf/arm.urdf.xacro"/>
  <xacro:include filename="$(find aria_description)/urdf/leg.urdf.xacro"/>

  <xacro:aria_arm prefix="left_" parent="base_link" reflect="1"/>
  <xacro:aria_arm prefix="right_" parent="base_link" reflect="-1"/>
  <xacro:aria_leg prefix="left_" parent="base_link" reflect="1"/>
  <xacro:aria_leg prefix="right_" parent="base_link" reflect="-1"/>

  <!-- Sensors -->
  <!-- RGB-D Camera -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.1 0.03"/>
      </geometry>
      <material name="blue"/>
    </visual>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>

  <!-- IMU -->
  <link name="imu_link"/>
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <robot_sim_type>gazebo_ros2_control/GazeboSystem</robot_sim_type>
    </plugin>
  </gazebo>

</robot>
```

### 1.3 Custom Interfaces

Define custom ROS 2 messages and services:

```python
# aria_interfaces/msg/HumanCommand.msg
string command_text
string[] detected_objects
geometry_msgs/Pose human_pose
float32 confidence

# aria_interfaces/msg/TaskStatus.msg
string task_id
string task_description
uint8 status  # 0=pending, 1=running, 2=success, 3=failed
string current_step
float32 progress

# aria_interfaces/srv/ExecuteTask.srv
string task_description
string[] constraints
---
bool accepted
string task_id
string message

# aria_interfaces/action/NavigateAndManipulate.action
# Goal
string target_object
string target_location
bool return_to_start
---
# Result
bool success
string message
float32 total_time
---
# Feedback
string current_phase
float32 progress
geometry_msgs/Pose current_pose
```

### Milestone 1 Checklist

- [ ] ROS 2 workspace compiles without errors
- [ ] URDF displays correctly in RViz2
- [ ] All custom interfaces build successfully
- [ ] Launch file starts basic nodes

---

## Phase 2: Perception System (Week 1-2)

### 2.1 Visual Perception Node

```python
#!/usr/bin/env python3
"""
aria_perception/visual_perception_node.py
Main visual perception node for ARIA.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np

class VisualPerceptionNode(Node):
    def __init__(self):
        super().__init__('visual_perception_node')

        # Parameters
        self.declare_parameter('model_name', 'google/owlvit-base-patch32')
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('target_objects', ['cup', 'book', 'phone', 'remote'])

        model_name = self.get_parameter('model_name').value
        self.threshold = self.get_parameter('confidence_threshold').value
        self.target_objects = self.get_parameter('target_objects').value

        # Load model
        self.get_logger().info(f'Loading model: {model_name}')
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.get_logger().info('Using CUDA')

        self.bridge = CvBridge()
        self.camera_info = None

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw',
            self.image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw',
            self.depth_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self.info_callback, 10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/aria/detections', 10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, '/aria/object_pose', 10
        )

        self.depth_image = None
        self.get_logger().info('Visual perception node initialized')

    def info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def image_callback(self, msg):
        # Convert image
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

        # Detect objects
        detections = self.detect_objects(cv_image)

        # Publish detections
        det_msg = Detection2DArray()
        det_msg.header = msg.header

        for det in detections:
            d = Detection2D()
            d.bbox.center.position.x = float(det['center'][0])
            d.bbox.center.position.y = float(det['center'][1])
            d.bbox.size_x = float(det['bbox'][2] - det['bbox'][0])
            d.bbox.size_y = float(det['bbox'][3] - det['bbox'][1])

            # Add hypothesis
            from vision_msgs.msg import ObjectHypothesisWithPose
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = det['label']
            hyp.hypothesis.score = det['score']
            d.results.append(hyp)

            det_msg.detections.append(d)

            # Publish 3D pose if depth available
            if self.depth_image is not None and self.camera_info is not None:
                pose = self.compute_3d_pose(det, msg.header)
                if pose:
                    self.pose_pub.publish(pose)

        self.detection_pub.publish(det_msg)

    def detect_objects(self, image):
        """Run object detection."""
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)

        inputs = self.processor(
            text=self.target_objects,
            images=pil_image,
            return_tensors='pt'
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([pil_image.size[::-1]])
        if torch.cuda.is_available():
            target_sizes = target_sizes.cuda()

        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]

        detections = []
        for score, label, box in zip(
            results['scores'], results['labels'], results['boxes']
        ):
            x1, y1, x2, y2 = box.int().tolist()
            detections.append({
                'label': self.target_objects[label],
                'score': score.item(),
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) / 2, (y1 + y2) / 2)
            })

        return detections

    def compute_3d_pose(self, detection, header):
        """Compute 3D pose from 2D detection and depth."""
        cx, cy = detection['center']
        cx, cy = int(cx), int(cy)

        if cy >= self.depth_image.shape[0] or cx >= self.depth_image.shape[1]:
            return None

        depth = self.depth_image[cy, cx]
        if depth <= 0 or np.isnan(depth):
            return None

        # Convert to meters if needed
        if depth > 100:
            depth = depth / 1000.0

        # Project to 3D
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        ppx = self.camera_info.k[2]
        ppy = self.camera_info.k[5]

        x = (cx - ppx) * depth / fx
        y = (cy - ppy) * depth / fy
        z = depth

        pose = PoseStamped()
        pose.header = header
        pose.header.frame_id = 'camera_link'
        pose.pose.position.x = z  # Camera Z -> Robot X
        pose.pose.position.y = -x  # Camera X -> Robot -Y
        pose.pose.position.z = -y  # Camera Y -> Robot -Z
        pose.pose.orientation.w = 1.0

        return pose


def main(args=None):
    rclpy.init(args=args)
    node = VisualPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2.2 Speech Interface

```python
#!/usr/bin/env python3
"""
aria_perception/speech_interface_node.py
Speech recognition and synthesis for ARIA.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from aria_interfaces.msg import HumanCommand
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import queue
import threading

class SpeechInterfaceNode(Node):
    def __init__(self):
        super().__init__('speech_interface_node')

        # Parameters
        self.declare_parameter('model_name', 'openai/whisper-base')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_duration', 3.0)

        model_name = self.get_parameter('model_name').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.chunk_duration = self.get_parameter('chunk_duration').value

        # Load Whisper model
        self.get_logger().info(f'Loading Whisper model: {model_name}')
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Audio queue
        self.audio_queue = queue.Queue()
        self.is_listening = False

        # Publishers
        self.command_pub = self.create_publisher(
            HumanCommand, '/aria/human_command', 10
        )
        self.transcript_pub = self.create_publisher(
            String, '/aria/speech_transcript', 10
        )

        # Subscribers
        self.speak_sub = self.create_subscription(
            String, '/aria/speak',
            self.speak_callback, 10
        )

        # Timer for processing audio
        self.timer = self.create_timer(0.1, self.process_audio)

        # Start listening
        self.start_listening()

        self.get_logger().info('Speech interface initialized')

    def start_listening(self):
        """Start audio capture."""
        self.is_listening = True

        def audio_callback(indata, frames, time, status):
            if status:
                self.get_logger().warn(f'Audio status: {status}')
            self.audio_queue.put(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=int(self.sample_rate * 0.1)
        )
        self.stream.start()

    def process_audio(self):
        """Process accumulated audio."""
        if self.audio_queue.qsize() < int(self.chunk_duration * 10):
            return

        # Collect audio chunks
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())

        if not audio_chunks:
            return

        audio = np.concatenate(audio_chunks).flatten()

        # Check for voice activity (simple energy threshold)
        energy = np.mean(audio ** 2)
        if energy < 0.001:
            return

        # Transcribe
        text = self.transcribe(audio)

        if text and len(text.strip()) > 2:
            self.get_logger().info(f'Heard: {text}')

            # Publish transcript
            msg = String()
            msg.data = text
            self.transcript_pub.publish(msg)

            # Check for command
            if self._is_command(text):
                cmd = HumanCommand()
                cmd.command_text = text
                cmd.confidence = 0.9
                self.command_pub.publish(cmd)

    def transcribe(self, audio):
        """Transcribe audio to text."""
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors='pt'
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(inputs['input_features'])

        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return text.strip()

    def _is_command(self, text):
        """Check if text is a command for ARIA."""
        triggers = ['aria', 'hey aria', 'robot', 'please', 'can you', 'could you']
        text_lower = text.lower()
        return any(t in text_lower for t in triggers)

    def speak_callback(self, msg):
        """Synthesize and speak text."""
        # Using system TTS for simplicity
        import subprocess
        subprocess.run(['espeak', msg.data], capture_output=True)

    def destroy_node(self):
        self.is_listening = False
        if hasattr(self, 'stream'):
            self.stream.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeechInterfaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Milestone 2 Checklist

- [ ] Visual perception detects target objects
- [ ] 3D pose estimation works with depth camera
- [ ] Speech recognition transcribes commands
- [ ] System responds to "ARIA" wake word

---

## Phase 3: Navigation and Manipulation (Week 2-3)

### 3.1 Navigation System

```python
#!/usr/bin/env python3
"""
aria_navigation/navigation_node.py
Navigation system for ARIA using Nav2.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from aria_interfaces.srv import ExecuteTask
import yaml

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Load known locations
        self.declare_parameter('locations_file', '')
        locations_file = self.get_parameter('locations_file').value

        self.locations = {}
        if locations_file:
            with open(locations_file, 'r') as f:
                self.locations = yaml.safe_load(f)

        # Nav2 action client
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

        # Service
        self.navigate_srv = self.create_service(
            ExecuteTask, '/aria/navigate_to',
            self.navigate_callback
        )

        self.get_logger().info('Navigation node initialized')
        self.get_logger().info(f'Known locations: {list(self.locations.keys())}')

    def navigate_callback(self, request, response):
        """Handle navigation request."""
        location_name = request.task_description

        if location_name in self.locations:
            loc = self.locations[location_name]
            success = self.navigate_to_pose(
                loc['x'], loc['y'], loc['theta']
            )

            response.accepted = True
            response.task_id = f'nav_{location_name}'
            response.message = 'Navigation started' if success else 'Navigation failed'
        else:
            response.accepted = False
            response.message = f'Unknown location: {location_name}'

        return response

    def navigate_to_pose(self, x, y, theta):
        """Navigate to a specific pose."""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 server not available')
            return False

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y

        # Convert theta to quaternion
        import math
        goal.pose.pose.orientation.z = math.sin(theta / 2)
        goal.pose.pose.orientation.w = math.cos(theta / 2)

        self.get_logger().info(f'Navigating to ({x}, {y}, {theta})')

        future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        return future.result() is not None

    def navigate_to_object(self, object_pose):
        """Navigate to near an object."""
        # Compute approach pose (offset from object)
        approach_distance = 0.5

        x = object_pose.pose.position.x - approach_distance
        y = object_pose.pose.position.y
        theta = 0.0  # Face the object

        return self.navigate_to_pose(x, y, theta)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3.2 Manipulation Controller

```python
#!/usr/bin/env python3
"""
aria_manipulation/manipulation_node.py
Manipulation control for ARIA.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from aria_interfaces.action import NavigateAndManipulate
from aria_interfaces.srv import ExecuteTask
import numpy as np

class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation_node')

        # Joint names for arms
        self.left_arm_joints = [
            'left_shoulder_pitch', 'left_shoulder_roll',
            'left_elbow', 'left_wrist_pitch', 'left_wrist_roll'
        ]
        self.right_arm_joints = [
            'right_shoulder_pitch', 'right_shoulder_roll',
            'right_elbow', 'right_wrist_pitch', 'right_wrist_roll'
        ]

        # Trajectory publisher
        self.traj_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10
        )

        # Services
        self.pick_srv = self.create_service(
            ExecuteTask, '/aria/pick_object',
            self.pick_callback
        )
        self.place_srv = self.create_service(
            ExecuteTask, '/aria/place_object',
            self.place_callback
        )

        # Current state
        self.holding_object = None

        self.get_logger().info('Manipulation node initialized')

    def pick_callback(self, request, response):
        """Handle pick request."""
        object_name = request.task_description

        self.get_logger().info(f'Attempting to pick: {object_name}')

        # Execute pick sequence
        success = self.execute_pick_sequence()

        if success:
            self.holding_object = object_name
            response.accepted = True
            response.task_id = f'pick_{object_name}'
            response.message = f'Successfully picked {object_name}'
        else:
            response.accepted = False
            response.message = f'Failed to pick {object_name}'

        return response

    def place_callback(self, request, response):
        """Handle place request."""
        location = request.task_description

        if self.holding_object is None:
            response.accepted = False
            response.message = 'Not holding any object'
            return response

        self.get_logger().info(f'Placing {self.holding_object} at {location}')

        success = self.execute_place_sequence()

        if success:
            response.accepted = True
            response.task_id = f'place_{self.holding_object}'
            response.message = f'Successfully placed {self.holding_object}'
            self.holding_object = None
        else:
            response.accepted = False
            response.message = 'Failed to place object'

        return response

    def execute_pick_sequence(self):
        """Execute pick motion sequence."""
        # Pre-grasp
        self.move_to_pose('pre_grasp')
        self.wait(1.0)

        # Approach
        self.move_to_pose('grasp')
        self.wait(0.5)

        # Close gripper
        self.close_gripper()
        self.wait(0.5)

        # Lift
        self.move_to_pose('post_grasp')
        self.wait(0.5)

        return True

    def execute_place_sequence(self):
        """Execute place motion sequence."""
        # Pre-place
        self.move_to_pose('pre_place')
        self.wait(1.0)

        # Place
        self.move_to_pose('place')
        self.wait(0.5)

        # Open gripper
        self.open_gripper()
        self.wait(0.5)

        # Retract
        self.move_to_pose('post_place')
        self.wait(0.5)

        return True

    def move_to_pose(self, pose_name):
        """Move arm to named pose."""
        poses = {
            'home': [0.0, 0.0, 0.0, 0.0, 0.0],
            'pre_grasp': [0.5, 0.0, -0.5, 0.0, 0.0],
            'grasp': [0.7, 0.0, -0.7, 0.0, 0.0],
            'post_grasp': [0.5, 0.0, -0.5, 0.0, 0.0],
            'pre_place': [0.5, 0.3, -0.5, 0.0, 0.0],
            'place': [0.7, 0.3, -0.7, 0.0, 0.0],
            'post_place': [0.5, 0.3, -0.5, 0.0, 0.0],
        }

        if pose_name not in poses:
            return

        target = poses[pose_name]

        # Create trajectory
        traj = JointTrajectory()
        traj.joint_names = self.right_arm_joints

        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start.sec = 2

        traj.points.append(point)

        self.traj_pub.publish(traj)

    def close_gripper(self):
        """Close gripper."""
        self.get_logger().info('Closing gripper')
        # Publish to gripper controller

    def open_gripper(self):
        """Open gripper."""
        self.get_logger().info('Opening gripper')
        # Publish to gripper controller

    def wait(self, duration):
        """Wait for specified duration."""
        import time
        time.sleep(duration)


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Milestone 3 Checklist

- [ ] Navigation to named locations works
- [ ] Basic pick and place sequences execute
- [ ] Obstacle avoidance during navigation
- [ ] Arm trajectories are smooth and safe

---

## Phase 4: Task Planning Integration (Week 3-4)

### 4.1 LLM Task Planner

```python
#!/usr/bin/env python3
"""
aria_planning/task_planner_node.py
LLM-based task planning for ARIA.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from aria_interfaces.msg import HumanCommand, TaskStatus
from aria_interfaces.srv import ExecuteTask
import json

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # LLM client (configure based on your setup)
        self.declare_parameter('llm_endpoint', 'http://localhost:8000/v1/chat/completions')
        self.llm_endpoint = self.get_parameter('llm_endpoint').value

        # Available skills
        self.skills = [
            'navigate_to(location)',
            'pick_object(object_name)',
            'place_object(location)',
            'speak(message)',
            'look_at(target)',
            'wait(seconds)'
        ]

        # Known locations and objects
        self.locations = ['kitchen', 'living_room', 'bedroom', 'entrance']
        self.objects = ['cup', 'book', 'phone', 'remote', 'keys']

        # Subscribers
        self.command_sub = self.create_subscription(
            HumanCommand, '/aria/human_command',
            self.command_callback, 10
        )

        # Publishers
        self.status_pub = self.create_publisher(
            TaskStatus, '/aria/task_status', 10
        )
        self.speak_pub = self.create_publisher(
            String, '/aria/speak', 10
        )

        # Service clients
        self.nav_client = self.create_client(
            ExecuteTask, '/aria/navigate_to'
        )
        self.pick_client = self.create_client(
            ExecuteTask, '/aria/pick_object'
        )
        self.place_client = self.create_client(
            ExecuteTask, '/aria/place_object'
        )

        self.current_task_id = 0
        self.get_logger().info('Task planner initialized')

    def command_callback(self, msg):
        """Handle human command."""
        command = msg.command_text
        self.get_logger().info(f'Received command: {command}')

        # Generate plan
        plan = self.generate_plan(command)

        if plan:
            self.current_task_id += 1
            task_id = f'task_{self.current_task_id}'

            # Acknowledge
            speak_msg = String()
            speak_msg.data = f"I'll {command.lower()}"
            self.speak_pub.publish(speak_msg)

            # Execute plan
            self.execute_plan(plan, task_id)
        else:
            speak_msg = String()
            speak_msg.data = "I'm sorry, I don't understand that request."
            self.speak_pub.publish(speak_msg)

    def generate_plan(self, command):
        """Generate task plan using LLM."""
        prompt = self._build_planning_prompt(command)

        # Call LLM (simplified - implement actual API call)
        response = self._call_llm(prompt)

        # Parse plan
        try:
            plan = self._parse_plan(response)
            return plan
        except Exception as e:
            self.get_logger().error(f'Failed to parse plan: {e}')
            return None

    def _build_planning_prompt(self, command):
        return f"""You are a task planner for a humanoid robot assistant named ARIA.

Available skills:
{chr(10).join(['- ' + s for s in self.skills])}

Known locations: {', '.join(self.locations)}
Known objects: {', '.join(self.objects)}

User command: "{command}"

Generate a step-by-step plan using only the available skills.
Output as JSON array:
[
  {{"skill": "skill_name", "args": {{"arg1": "value1"}}}},
  ...
]

Plan:"""

    def _call_llm(self, prompt):
        """Call LLM API."""
        import requests

        try:
            response = requests.post(
                self.llm_endpoint,
                json={
                    'model': 'gpt-4',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.1
                },
                timeout=30
            )
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            self.get_logger().error(f'LLM call failed: {e}')
            return None

    def _parse_plan(self, response):
        """Parse LLM response into plan."""
        # Find JSON array in response
        start = response.find('[')
        end = response.rfind(']') + 1

        if start == -1 or end == 0:
            return None

        json_str = response[start:end]
        return json.loads(json_str)

    def execute_plan(self, plan, task_id):
        """Execute generated plan."""
        total_steps = len(plan)

        for i, step in enumerate(plan):
            skill = step['skill']
            args = step.get('args', {})

            # Update status
            status = TaskStatus()
            status.task_id = task_id
            status.status = 1  # Running
            status.current_step = f'{skill}({args})'
            status.progress = float(i) / total_steps
            self.status_pub.publish(status)

            # Execute skill
            success = self._execute_skill(skill, args)

            if not success:
                status.status = 3  # Failed
                self.status_pub.publish(status)
                return

        # Complete
        status = TaskStatus()
        status.task_id = task_id
        status.status = 2  # Success
        status.progress = 1.0
        self.status_pub.publish(status)

        speak_msg = String()
        speak_msg.data = "Task completed."
        self.speak_pub.publish(speak_msg)

    def _execute_skill(self, skill, args):
        """Execute a single skill."""
        if skill == 'navigate_to':
            return self._call_navigate(args.get('location'))
        elif skill == 'pick_object':
            return self._call_pick(args.get('object_name'))
        elif skill == 'place_object':
            return self._call_place(args.get('location'))
        elif skill == 'speak':
            msg = String()
            msg.data = args.get('message', '')
            self.speak_pub.publish(msg)
            return True
        elif skill == 'wait':
            import time
            time.sleep(float(args.get('seconds', 1)))
            return True
        else:
            self.get_logger().warn(f'Unknown skill: {skill}')
            return False

    def _call_navigate(self, location):
        """Call navigation service."""
        if not self.nav_client.wait_for_service(timeout_sec=5.0):
            return False

        request = ExecuteTask.Request()
        request.task_description = location

        future = self.nav_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result().accepted

    def _call_pick(self, object_name):
        """Call pick service."""
        if not self.pick_client.wait_for_service(timeout_sec=5.0):
            return False

        request = ExecuteTask.Request()
        request.task_description = object_name

        future = self.pick_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result().accepted

    def _call_place(self, location):
        """Call place service."""
        if not self.place_client.wait_for_service(timeout_sec=5.0):
            return False

        request = ExecuteTask.Request()
        request.task_description = location

        future = self.place_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result().accepted


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Milestone 4 Checklist

- [ ] LLM generates valid plans from commands
- [ ] Plans execute correctly through skill calls
- [ ] Error handling and recovery works
- [ ] ARIA responds naturally to voice commands

---

## Phase 5: Integration and Testing (Week 4)

### 5.1 Complete Launch File

```python
# aria_bringup/launch/aria_complete.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    aria_description = get_package_share_directory('aria_description')
    aria_bringup = get_package_share_directory('aria_bringup')

    return LaunchDescription([
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': open(
                    os.path.join(aria_description, 'urdf', 'aria.urdf')
                ).read()
            }]
        ),

        # Perception
        Node(
            package='aria_perception',
            executable='visual_perception_node',
            parameters=[{
                'target_objects': ['cup', 'book', 'phone', 'remote', 'keys']
            }]
        ),
        Node(
            package='aria_perception',
            executable='speech_interface_node',
        ),

        # Navigation
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('nav2_bringup'),
                            'launch', 'navigation_launch.py')
            ])
        ),
        Node(
            package='aria_navigation',
            executable='navigation_node',
            parameters=[{
                'locations_file': os.path.join(aria_bringup, 'config', 'locations.yaml')
            }]
        ),

        # Manipulation
        Node(
            package='aria_manipulation',
            executable='manipulation_node',
        ),

        # Planning
        Node(
            package='aria_planning',
            executable='task_planner_node',
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(aria_bringup, 'rviz', 'aria.rviz')]
        ),
    ])
```

### 5.2 Test Scenarios

```python
# aria_bringup/test/test_scenarios.py
"""Test scenarios for ARIA."""

import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from aria_interfaces.msg import HumanCommand, TaskStatus
import time

class ARIATestNode(Node):
    def __init__(self):
        super().__init__('aria_test_node')

        self.command_pub = self.create_publisher(
            HumanCommand, '/aria/human_command', 10
        )

        self.task_status = None
        self.status_sub = self.create_subscription(
            TaskStatus, '/aria/task_status',
            self.status_callback, 10
        )

    def status_callback(self, msg):
        self.task_status = msg

    def send_command(self, text):
        msg = HumanCommand()
        msg.command_text = text
        msg.confidence = 1.0
        self.command_pub.publish(msg)

    def wait_for_completion(self, timeout=60.0):
        start = time.time()
        while time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.task_status and self.task_status.status in [2, 3]:
                return self.task_status.status == 2
        return False


class TestObjectRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = ARIATestNode()

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_fetch_object_from_location(self):
        """Test: 'ARIA, bring me the book from the living room'"""
        self.node.send_command(
            "ARIA, bring me the book from the living room"
        )
        success = self.node.wait_for_completion()
        self.assertTrue(success)

    def test_navigate_to_location(self):
        """Test: 'ARIA, go to the kitchen'"""
        self.node.send_command("ARIA, go to the kitchen")
        success = self.node.wait_for_completion()
        self.assertTrue(success)

    def test_pick_and_place(self):
        """Test: 'ARIA, pick up the cup and put it on the table'"""
        self.node.send_command(
            "ARIA, pick up the cup and put it on the table"
        )
        success = self.node.wait_for_completion()
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
```

---

## Evaluation Criteria

### Technical Requirements (60%)

| Requirement | Points | Description |
|-------------|--------|-------------|
| ROS 2 Architecture | 15 | Clean modular design, proper interfaces |
| Perception | 15 | Accurate detection, 3D localization |
| Navigation | 10 | Reliable path planning, obstacle avoidance |
| Manipulation | 10 | Smooth trajectories, grasp success |
| Planning | 10 | Correct plan generation, execution |

### Integration Quality (20%)

| Criterion | Points | Description |
|-----------|--------|-------------|
| End-to-end flow | 10 | Seamless component integration |
| Error handling | 5 | Graceful failure recovery |
| Performance | 5 | Responsive real-time operation |

### Documentation (10%)

| Item | Points | Description |
|------|--------|-------------|
| README | 3 | Setup and usage instructions |
| Architecture | 4 | System design documentation |
| API docs | 3 | Interface documentation |

### Demo & Presentation (10%)

| Element | Points | Description |
|---------|--------|-------------|
| Live demo | 5 | Working demonstration |
| Presentation | 5 | Clear explanation of approach |

---

## Submission Requirements

1. **GitHub Repository** with all code
2. **README.md** with setup instructions
3. **Demo Video** (3-5 minutes) showing:
   - Object retrieval task
   - Navigation with obstacle avoidance
   - Voice command interaction
4. **Architecture Document** explaining design decisions
5. **Reflection** (1-2 pages) on challenges and learnings

---

## Extensions (Extra Credit)

- **Multi-person interaction**: Handle multiple humans
- **Learning from demonstration**: Teach new tasks
- **Emotion recognition**: Adapt behavior to human mood
- **Mobile app control**: Remote monitoring and control
- **Simulation validation**: Isaac Sim digital twin

---

## Resources

- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)
- [Nav2 Documentation](https://navigation.ros.org/)
- [MoveIt 2 Documentation](https://moveit.ros.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)

Good luck with your capstone project!
