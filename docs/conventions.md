---
id: conventions
title: Conventions Used
sidebar_position: 3
---

# Conventions Used in This Book

This page explains the formatting conventions, notation, and symbols used throughout the textbook.

## Typography

### Code and Commands

**Inline code** appears in `monospace font`:

- File names: `robot_description.urdf`
- Package names: `rclpy`
- Commands: `ros2 run`
- Variables: `joint_angle`

**Code blocks** show complete examples:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
```

### Terminal Commands

Commands you type in a terminal are shown with a `$` prompt:

```bash
$ ros2 run my_package my_node
```

Output from commands appears without a prompt:

```
[INFO] [my_node]: Node started successfully
```

### File Paths

- **Absolute paths**: `/opt/ros/humble/share/`
- **Relative paths**: `src/my_package/`
- **Home directory**: `~/ros2_ws/`

---

## Admonitions

Special callout boxes highlight important information:

:::note
General information or tips that enhance understanding.
:::

:::tip Best Practice
Recommended approaches and industry best practices.
:::

:::info
Additional context or background information.
:::

:::caution
Important warnings about potential issues or pitfalls.
:::

:::danger
Critical safety or data loss warnings. Do not ignore these.
:::

---

## Mathematical Notation

### Variables and Symbols

| Symbol | Meaning |
|--------|---------|
| θ (theta) | Joint angle (radians) |
| ω (omega) | Angular velocity (rad/s) |
| **p** | Position vector (bold for vectors) |
| **R** | Rotation matrix |
| **T** | Transformation matrix |
| ẋ | Time derivative of x |

### Coordinate Frames

- **World frame**: Fixed reference frame, denoted as `{W}`
- **Base frame**: Robot base, denoted as `{B}`
- **End-effector frame**: Tool/hand, denoted as `{E}`

### Units

All measurements use SI units unless otherwise noted:

| Quantity | Unit | Symbol |
|----------|------|--------|
| Length | meters | m |
| Angle | radians | rad |
| Time | seconds | s |
| Mass | kilograms | kg |
| Force | newtons | N |
| Torque | newton-meters | N·m |

---

## Diagrams and Figures

### Diagram Types

- **Architecture diagrams**: System component relationships
- **Sequence diagrams**: Message flow over time
- **Node graphs**: ROS 2 node/topic connections
- **Kinematic diagrams**: Robot joint and link structure

### Figure Captions

All figures include:
1. **Figure number**: Sequential within each chapter (e.g., Figure 3.2)
2. **Title**: Brief description
3. **Caption**: Detailed explanation

Example:
> _Figure 2.1: ROS 2 Publisher-Subscriber Pattern. Messages flow from publisher nodes through topics to subscriber nodes._

---

## Code Examples

### Structure

Each code example includes:

1. **Title**: Descriptive name
2. **Language indicator**: Python, C++, YAML, etc.
3. **Line numbers**: For reference in text
4. **Highlighted lines**: Key lines to focus on (shown with yellow background)
5. **Expected output**: When applicable

### Example Format

```python title="minimal_publisher.py" showLineNumbers
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        # highlight-next-line
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, World!'
        self.publisher_.publish(msg)
```

**Expected Output:**
```
[INFO] [minimal_publisher]: Publishing: "Hello, World!"
```

### Downloadable Code

Complete code examples are available in the repository:
- Location: `static/code/module-N/chNN/`
- Each chapter's code can be downloaded and run directly

---

## Exercises

### Exercise Format

Each exercise follows this structure:

```
### Exercise N.M: Title

**Objective**: What you will accomplish

**Difficulty**: Beginner | Intermediate | Advanced

**Estimated Time**: X minutes

#### Instructions
1. Step-by-step guide
2. ...

#### Expected Outcome
What success looks like

#### Verification
How to confirm completion

#### Troubleshooting
| Issue | Cause | Solution |
|-------|-------|----------|
| ... | ... | ... |
```

### Difficulty Levels

| Level | Description |
|-------|-------------|
| **Beginner** | Follows provided code with minor modifications |
| **Intermediate** | Requires understanding concepts and writing some code |
| **Advanced** | Requires problem-solving and integration of multiple concepts |

---

## References and Citations

### Citation Style

This book uses **IEEE citation style**:

- In-text: "...as shown in [1]..."
- Reference list: Numbered, at end of each chapter

### Reference Format

```
[1] A. Author and B. Author, "Title of Article," Journal Name,
    vol. X, no. Y, pp. 1-10, Year.

[2] C. Author, Book Title, Xth ed. City: Publisher, Year.

[3] "Web Page Title," Site Name. [Online]. Available:
    https://example.com. [Accessed: Month Day, Year].
```

---

## Abbreviations

Common abbreviations used throughout the book:

| Abbreviation | Full Form |
|--------------|-----------|
| ROS | Robot Operating System |
| URDF | Unified Robot Description Format |
| SDF | Simulation Description Format |
| VSLAM | Visual Simultaneous Localization and Mapping |
| VLA | Vision-Language-Action |
| LLM | Large Language Model |
| HRI | Human-Robot Interaction |
| Nav2 | Navigation 2 |
| IMU | Inertial Measurement Unit |
| LiDAR | Light Detection and Ranging |
| DoF | Degrees of Freedom |

---

## Keyboard Shortcuts

When keyboard shortcuts are referenced:

- **Ctrl+C**: Hold Control, press C
- **Ctrl+Shift+T**: Hold Control and Shift, press T
- Platform-specific notes indicate macOS differences (⌘ for Ctrl)

---

## Version Information

This book targets:

| Software | Version |
|----------|---------|
| Ubuntu | 22.04 LTS |
| ROS 2 | Humble Hawksbill |
| Gazebo | Fortress |
| Isaac Sim | 2023.1+ |
| Unity | 2022.3 LTS |
| Python | 3.10+ |

:::caution Version Compatibility
Commands and configurations are tested with these specific versions. Other versions may require modifications.
:::
