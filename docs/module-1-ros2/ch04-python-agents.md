---
id: ch04-python-agents
title: "Chapter 4: Integrating Python AI Agents with rclpy"
sidebar_position: 5
---

# Chapter 4: Integrating Python AI Agents with rclpy

**Estimated Time**: 4-5 hours | **Exercises**: 3

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Master** advanced rclpy patterns for building intelligent nodes
2. **Implement** behavior trees for humanoid decision-making
3. **Integrate** external AI services (LLMs, vision APIs) with ROS 2
4. **Design** reactive and deliberative agent architectures
5. **Handle** real-time constraints in AI-robotics systems

---

## 4.1 Advanced rclpy Patterns

### Multi-Threaded Executors

For AI agents that need to handle multiple callbacks concurrently:

```python
#!/usr/bin/env python3
"""
multi_threaded_agent.py
Demonstrates multi-threaded execution for AI processing.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import String
import threading

class AIAgent(Node):
    def __init__(self):
        super().__init__('ai_agent')

        # Separate callback groups for parallelism
        self.sensor_group = ReentrantCallbackGroup()
        self.ai_group = MutuallyExclusiveCallbackGroup()

        # High-frequency sensor subscription
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10,
            callback_group=self.sensor_group
        )

        # AI inference (can run in parallel)
        self.inference_timer = self.create_timer(
            0.1,  # 10 Hz inference
            self.run_inference,
            callback_group=self.ai_group
        )

        # Decision publisher
        self.decision_pub = self.create_publisher(String, '/agent/decision', 10)

        self.latest_image = None
        self.lock = threading.Lock()

        self.get_logger().info('AI Agent initialized with multi-threading')

    def camera_callback(self, msg):
        with self.lock:
            self.latest_image = msg

    def run_inference(self):
        with self.lock:
            if self.latest_image is None:
                return
            image = self.latest_image

        # Simulate AI inference (replace with actual model)
        decision = self.process_image(image)

        msg = String()
        msg.data = decision
        self.decision_pub.publish(msg)

    def process_image(self, image):
        # Placeholder for actual AI inference
        return "move_forward"

def main(args=None):
    rclpy.init(args=args)
    agent = AIAgent()

    # Use multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(agent)

    try:
        executor.spin()
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Async/Await Patterns

```python
import asyncio
from rclpy.node import Node

class AsyncAgent(Node):
    def __init__(self):
        super().__init__('async_agent')
        self.timer = self.create_timer(1.0, self.timer_callback)

    async def async_ai_call(self, prompt):
        """Async call to AI service."""
        await asyncio.sleep(0.5)  # Simulate API call
        return f"Response to: {prompt}"

    def timer_callback(self):
        # Run async code from sync callback
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.async_ai_call("What should the robot do?")
        )
        self.get_logger().info(f'AI Response: {result}')
```

---

## 4.2 Behavior Trees for Humanoids

Behavior trees provide a modular, hierarchical approach to robot decision-making.

### Core Concepts

```
       [Root]
          │
     [Sequence]
      /   |   \
   [Check] [Move] [Grasp]
   Battery  To     Object
            Goal
```

**Node Types**:
- **Sequence**: Execute children in order; fail if any fails
- **Selector**: Try children until one succeeds
- **Decorator**: Modify child behavior (repeat, invert, timeout)
- **Action**: Execute robot behaviors
- **Condition**: Check world state

### Implementation with py_trees

```python
#!/usr/bin/env python3
"""
humanoid_behavior_tree.py
Behavior tree for humanoid task execution.
"""

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CheckBattery(Behaviour):
    """Condition: Check if battery is sufficient."""

    def __init__(self, name, threshold=20.0):
        super().__init__(name)
        self.threshold = threshold
        self.battery_level = 100.0  # Simulated

    def update(self):
        if self.battery_level > self.threshold:
            return Status.SUCCESS
        return Status.FAILURE

class MoveToGoal(Behaviour):
    """Action: Move humanoid to target location."""

    def __init__(self, name, node, target_x, target_y):
        super().__init__(name)
        self.node = node
        self.target = (target_x, target_y)
        self.cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)
        self.distance = 10.0  # Simulated

    def initialise(self):
        self.node.get_logger().info(f'Moving to {self.target}')

    def update(self):
        if self.distance < 0.1:
            return Status.SUCCESS

        # Publish movement command
        cmd = Twist()
        cmd.linear.x = 0.5
        self.cmd_pub.publish(cmd)
        self.distance -= 0.5

        return Status.RUNNING

    def terminate(self, new_status):
        # Stop movement
        cmd = Twist()
        self.cmd_pub.publish(cmd)

class GraspObject(Behaviour):
    """Action: Grasp detected object."""

    def __init__(self, name, node):
        super().__init__(name)
        self.node = node
        self.grasp_progress = 0

    def update(self):
        self.grasp_progress += 1

        if self.grasp_progress < 5:
            self.node.get_logger().info(f'Grasping... {self.grasp_progress}/5')
            return Status.RUNNING

        self.node.get_logger().info('Object grasped!')
        return Status.SUCCESS

def create_humanoid_tree(node):
    """Create the behavior tree for humanoid tasks."""

    # Build tree structure
    root = py_trees.composites.Sequence("HumanoidTask", memory=True)

    # Check prerequisites
    check_battery = CheckBattery("CheckBattery", threshold=20.0)

    # Move to object
    move_to_object = MoveToGoal("MoveToObject", node, 2.0, 1.0)

    # Grasp object
    grasp = GraspObject("GraspObject", node)

    # Move to destination
    move_to_dest = MoveToGoal("MoveToDestination", node, 0.0, 0.0)

    # Assemble tree
    root.add_children([check_battery, move_to_object, grasp, move_to_dest])

    return root

class BehaviorTreeNode(Node):
    def __init__(self):
        super().__init__('behavior_tree_node')

        self.tree = create_humanoid_tree(self)
        self.tree.setup_with_descendants()

        # Tick the tree at 10 Hz
        self.timer = self.create_timer(0.1, self.tick_tree)

        self.get_logger().info('Behavior tree initialized')

    def tick_tree(self):
        self.tree.tick_once()

        if self.tree.status == Status.SUCCESS:
            self.get_logger().info('Task completed successfully!')
            self.timer.cancel()
        elif self.tree.status == Status.FAILURE:
            self.get_logger().error('Task failed!')
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = BehaviorTreeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 4.3 External AI Service Integration

### LLM Integration for Task Planning

```python
#!/usr/bin/env python3
"""
llm_planner.py
Integrates LLM for high-level task planning.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import os

# For OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# For local LLM (Ollama)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

class LLMPlanner(Node):
    def __init__(self):
        super().__init__('llm_planner')

        # Parameters
        self.declare_parameter('use_local_llm', True)
        self.declare_parameter('ollama_model', 'llama2')
        self.declare_parameter('ollama_url', 'http://localhost:11434')

        self.use_local = self.get_parameter('use_local_llm').value

        # Subscribers and publishers
        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self.plan_pub = self.create_publisher(String, '/task_plan', 10)

        self.get_logger().info('LLM Planner initialized')

    def command_callback(self, msg):
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Generate plan from LLM
        plan = self.generate_plan(command)

        # Publish plan
        plan_msg = String()
        plan_msg.data = json.dumps(plan)
        self.plan_pub.publish(plan_msg)

    def generate_plan(self, command):
        """Generate action plan from natural language command."""

        prompt = f"""You are a humanoid robot task planner. Given a command,
output a JSON list of actions. Each action has:
- "action": one of [move_to, grasp, release, look_at, speak]
- "parameters": action-specific parameters

Command: {command}

Output only valid JSON, no explanation."""

        if self.use_local:
            response = self.call_ollama(prompt)
        else:
            response = self.call_openai(prompt)

        try:
            plan = json.loads(response)
        except json.JSONDecodeError:
            plan = [{"action": "speak", "parameters": {"text": "I didn't understand"}}]

        return plan

    def call_ollama(self, prompt):
        """Call local Ollama LLM."""
        url = self.get_parameter('ollama_url').value + '/api/generate'
        model = self.get_parameter('ollama_model').value

        try:
            response = requests.post(url, json={
                'model': model,
                'prompt': prompt,
                'stream': False
            }, timeout=30)

            return response.json().get('response', '[]')
        except Exception as e:
            self.get_logger().error(f'Ollama error: {e}')
            return '[]'

    def call_openai(self, prompt):
        """Call OpenAI API."""
        if not HAS_OPENAI:
            return '[]'

        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            self.get_logger().error(f'OpenAI error: {e}')
            return '[]'

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Vision API Integration

```python
#!/usr/bin/env python3
"""
vision_agent.py
Integrates computer vision for object detection.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge
import numpy as np

class VisionAgent(Node):
    def __init__(self):
        super().__init__('vision_agent')

        self.bridge = CvBridge()

        # Camera subscription
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Detection publisher
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detections',
            10
        )

        self.get_logger().info('Vision Agent initialized')

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Run detection (placeholder - use actual model)
        detections = self.detect_objects(cv_image)

        # Publish detections
        det_msg = Detection2DArray()
        det_msg.header = msg.header
        det_msg.detections = detections
        self.detection_pub.publish(det_msg)

    def detect_objects(self, image):
        """Run object detection on image."""
        # Placeholder - integrate YOLO, Detectron2, etc.
        detections = []

        # Simulated detection
        det = Detection2D()
        det.results = []
        detections.append(det)

        return detections

def main(args=None):
    rclpy.init(args=args)
    node = VisionAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 4.4 Real-Time Considerations

### Deadline Handling

```python
from rclpy.qos import QoSProfile, DeadlineQoSPolicy
from rclpy.duration import Duration

# QoS with deadline
deadline_qos = QoSProfile(
    depth=10,
    deadline=DeadlineQoSPolicy(period=Duration(seconds=0.1))
)

# Publisher with deadline
self.create_publisher(Twist, '/cmd_vel', deadline_qos)
```

### Lifecycle Nodes

```python
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn

class ManagedAgent(LifecycleNode):
    def __init__(self):
        super().__init__('managed_agent')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring...')
        # Initialize resources
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating...')
        # Start processing
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating...')
        # Pause processing
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up...')
        # Release resources
        return TransitionCallbackReturn.SUCCESS
```

---

## Exercises

### Exercise 4.1: Create an rclpy Decision Node

**Objective**: Build a node that makes decisions based on sensor data.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Subscribe to `/joint_states` and `/scan` topics
2. Implement decision logic based on joint positions and obstacles
3. Publish decisions to `/agent/action`
4. Use proper callback groups for thread safety

#### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Callbacks blocking | Single-threaded executor | Use MultiThreadedExecutor |
| Race conditions | Shared state access | Use threading.Lock |

---

### Exercise 4.2: Implement a Behavior Tree

**Objective**: Create a behavior tree for a pick-and-place task.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Install py_trees: `pip install py_trees`
2. Create behaviors: Detect, Approach, Grasp, Move, Release
3. Compose into a sequence tree
4. Integrate with ROS 2 publishers/subscribers

---

### Exercise 4.3: Connect to External API

**Objective**: Integrate an external AI service for decision-making.

**Difficulty**: Advanced | **Estimated Time**: 45 minutes

#### Instructions

1. Set up Ollama locally with a small model
2. Create a ROS 2 node that sends prompts
3. Parse responses into robot actions
4. Handle API errors gracefully

---

## Summary

In this chapter, you learned:

- **Multi-threaded executors** enable parallel AI processing
- **Behavior trees** provide modular, hierarchical decision-making
- **LLM integration** enables natural language task planning
- **Vision APIs** provide object detection for manipulation
- **Lifecycle nodes** manage agent state transitions

---

## References

[1] M. Colledanchise and P. Ögren, *Behavior Trees in Robotics and AI*, CRC Press, 2018.

[2] Open Robotics, "rclpy API Documentation," [Online]. Available: https://docs.ros.org/en/humble/p/rclpy/.

[3] py_trees Documentation, [Online]. Available: https://py-trees.readthedocs.io/.

[4] OpenAI, "GPT-4 Technical Report," arXiv:2303.08774, 2023.
