---
id: ch07-sensor-sim
title: "Chapter 7: Sensor Simulation - LiDAR, Depth & IMU"
sidebar_position: 4
---

# Chapter 7: Sensor Simulation - LiDAR, Depth & IMU

**Estimated Time**: 5-6 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Configure** simulated LiDAR sensors for obstacle detection
2. **Implement** depth camera simulation for 3D perception
3. **Set up** IMU sensors for orientation and motion tracking
4. **Integrate** sensor data with ROS 2 navigation stacks
5. **Add noise models** for realistic sensor simulation

---

## 7.1 LiDAR Simulation

LiDAR (Light Detection and Ranging) sensors are essential for humanoid robot navigation and obstacle avoidance.

### LiDAR Sensor Types

| Type | Characteristics | Use Case |
|------|-----------------|----------|
| 2D LiDAR | Single plane scan | Indoor navigation |
| 3D LiDAR | Multi-plane point cloud | Outdoor mapping |
| Solid-state | No moving parts | Compact robots |

### Gazebo LiDAR Configuration

```xml
<!-- 2D LiDAR sensor -->
<sensor name="lidar" type="gpu_lidar">
  <pose>0 0 0.1 0 0 0</pose>
  <topic>/scan</topic>
  <update_rate>10</update_rate>

  <lidar>
    <scan>
      <horizontal>
        <samples>640</samples>
        <resolution>1</resolution>
        <min_angle>-2.35619</min_angle>
        <max_angle>2.35619</max_angle>
      </horizontal>
      <vertical>
        <samples>1</samples>
        <resolution>1</resolution>
        <min_angle>0</min_angle>
        <max_angle>0</max_angle>
      </vertical>
    </scan>
    <range>
      <min>0.08</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </lidar>

  <visualize>true</visualize>
</sensor>
```

### 3D LiDAR (Point Cloud)

```xml
<!-- 3D LiDAR with multiple scan planes -->
<sensor name="lidar_3d" type="gpu_lidar">
  <pose>0 0 1.5 0 0 0</pose>
  <topic>/points</topic>
  <update_rate>10</update_rate>

  <lidar>
    <scan>
      <horizontal>
        <samples>1800</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.26</min_angle>
        <max_angle>0.26</max_angle>
      </vertical>
    </scan>
    <range>
      <min>0.3</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </lidar>
</sensor>
```

### ROS 2 LiDAR Processing

```python
#!/usr/bin/env python3
"""
lidar_processor.py
Process LiDAR data for obstacle detection.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
import numpy as np

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        # Subscribe to LaserScan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publish filtered scan
        self.filtered_pub = self.create_publisher(
            LaserScan,
            '/scan_filtered',
            10
        )

        # Parameters
        self.declare_parameter('min_range', 0.1)
        self.declare_parameter('max_range', 8.0)

        self.get_logger().info('LiDAR processor initialized')

    def scan_callback(self, msg):
        # Filter out invalid readings
        ranges = np.array(msg.ranges)

        min_range = self.get_parameter('min_range').value
        max_range = self.get_parameter('max_range').value

        # Replace invalid values
        ranges[ranges < min_range] = float('inf')
        ranges[ranges > max_range] = float('inf')
        ranges[np.isnan(ranges)] = float('inf')

        # Create filtered message
        filtered_msg = LaserScan()
        filtered_msg.header = msg.header
        filtered_msg.angle_min = msg.angle_min
        filtered_msg.angle_max = msg.angle_max
        filtered_msg.angle_increment = msg.angle_increment
        filtered_msg.time_increment = msg.time_increment
        filtered_msg.scan_time = msg.scan_time
        filtered_msg.range_min = min_range
        filtered_msg.range_max = max_range
        filtered_msg.ranges = ranges.tolist()
        filtered_msg.intensities = msg.intensities

        self.filtered_pub.publish(filtered_msg)

        # Detect closest obstacle
        valid_ranges = ranges[np.isfinite(ranges)]
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            if min_distance < 0.5:
                self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')

def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 7.2 Depth Camera Simulation

Depth cameras provide 3D perception for manipulation and navigation.

### Depth Camera Types

| Camera | Technology | Range | Use Case |
|--------|------------|-------|----------|
| Intel RealSense D435 | Stereo + IR | 0.3-3m | Manipulation |
| Intel RealSense L515 | LiDAR | 0.25-9m | Mapping |
| Azure Kinect | ToF | 0.5-5.46m | Body tracking |
| ZED 2 | Stereo | 0.3-20m | Outdoor |

### Gazebo Depth Camera

```xml
<!-- Depth camera sensor -->
<sensor name="depth_camera" type="depth_camera">
  <pose>0.1 0 1.5 0 0 0</pose>
  <topic>/camera/depth</topic>
  <update_rate>30</update_rate>

  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R_FLOAT32</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.005</stddev>
    </noise>
  </camera>
</sensor>

<!-- RGB camera (co-located) -->
<sensor name="rgb_camera" type="camera">
  <pose>0.1 0 1.5 0 0 0</pose>
  <topic>/camera/image_raw</topic>
  <update_rate>30</update_rate>

  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
  </camera>
</sensor>
```

### Point Cloud Generation

```python
#!/usr/bin/env python3
"""
depth_to_pointcloud.py
Convert depth images to point clouds.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import struct

class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud')

        self.bridge = CvBridge()
        self.camera_info = None

        # Subscribers
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.info_callback,
            10
        )

        # Publisher
        self.pc_pub = self.create_publisher(
            PointCloud2,
            '/camera/points',
            10
        )

        self.get_logger().info('Depth to PointCloud converter initialized')

    def info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        if self.camera_info is None:
            return

        # Convert depth image
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

        # Camera intrinsics
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # Generate point cloud
        height, width = depth_image.shape

        # Create coordinate grids
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)

        # Calculate 3D coordinates
        z = depth_image
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Filter invalid points
        valid = (z > 0.1) & (z < 10.0) & np.isfinite(z)

        points = np.stack([x[valid], y[valid], z[valid]], axis=-1)

        # Create PointCloud2 message
        pc_msg = self.create_pointcloud2(msg.header, points)
        self.pc_pub.publish(pc_msg)

    def create_pointcloud2(self, header, points):
        """Create a PointCloud2 message from numpy array."""
        msg = PointCloud2()
        msg.header = header

        msg.height = 1
        msg.width = len(points)

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True

        msg.data = points.astype(np.float32).tobytes()

        return msg

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 7.3 IMU Simulation

Inertial Measurement Units (IMUs) provide orientation and motion data critical for humanoid balance.

### IMU Components

| Component | Measures | Units |
|-----------|----------|-------|
| Accelerometer | Linear acceleration | m/s^2 |
| Gyroscope | Angular velocity | rad/s |
| Magnetometer | Magnetic field | Tesla |

### Gazebo IMU Configuration

```xml
<!-- IMU sensor -->
<sensor name="imu" type="imu">
  <pose>0 0 0.5 0 0 0</pose>
  <topic>/imu/data</topic>
  <update_rate>200</update_rate>

  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0002</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0002</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.0002</stddev>
        </noise>
      </z>
    </angular_velocity>

    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

### IMU Data Processing

```python
#!/usr/bin/env python3
"""
imu_processor.py
Process IMU data for orientation estimation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
import numpy as np
from scipy.spatial.transform import Rotation

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publish processed orientation
        self.euler_pub = self.create_publisher(
            Vector3Stamped,
            '/imu/euler',
            10
        )

        # State
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion
        self.gyro_bias = np.zeros(3)
        self.last_time = None

        # Complementary filter parameters
        self.alpha = 0.98  # Gyro weight

        self.get_logger().info('IMU processor initialized')

    def imu_callback(self, msg):
        # Extract data
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]) - self.gyro_bias

        quat = np.array([
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ])

        # Use quaternion from sensor if available
        if np.linalg.norm(quat) > 0.9:
            self.orientation = quat

        # Convert to Euler angles
        r = Rotation.from_quat([
            self.orientation[1],
            self.orientation[2],
            self.orientation[3],
            self.orientation[0]
        ])
        euler = r.as_euler('xyz', degrees=True)

        # Publish Euler angles
        euler_msg = Vector3Stamped()
        euler_msg.header = msg.header
        euler_msg.vector.x = euler[0]  # Roll
        euler_msg.vector.y = euler[1]  # Pitch
        euler_msg.vector.z = euler[2]  # Yaw

        self.euler_pub.publish(euler_msg)

        # Check for tilt warning
        if abs(euler[0]) > 30 or abs(euler[1]) > 30:
            self.get_logger().warn(
                f'High tilt detected: Roll={euler[0]:.1f}, Pitch={euler[1]:.1f}'
            )

def main(args=None):
    rclpy.init(args=args)
    node = IMUProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 7.4 Sensor Fusion

Combining multiple sensors for robust state estimation.

### Extended Kalman Filter

```python
#!/usr/bin/env python3
"""
sensor_fusion.py
Fuse IMU and odometry using Extended Kalman Filter.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

class SensorFusion(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # State: [x, y, z, roll, pitch, yaw, vx, vy, vz]
        self.state = np.zeros(9)
        self.covariance = np.eye(9) * 0.1

        # Process noise
        self.Q = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.1, 0.1, 0.1])

        # Measurement noise
        self.R_imu = np.diag([0.01, 0.01, 0.01])  # Orientation
        self.R_odom = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # Pose

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Publisher
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/robot_pose', 10
        )

        self.last_time = None

        self.get_logger().info('Sensor fusion initialized')

    def predict(self, dt):
        """EKF prediction step."""
        # State transition (simple kinematic model)
        F = np.eye(9)
        F[0, 6] = dt  # x += vx * dt
        F[1, 7] = dt  # y += vy * dt
        F[2, 8] = dt  # z += vz * dt

        # Predict state
        self.state = F @ self.state

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q * dt

    def update_imu(self, orientation):
        """EKF update with IMU orientation."""
        # Measurement matrix (orientation only)
        H = np.zeros((3, 9))
        H[0, 3] = 1  # roll
        H[1, 4] = 1  # pitch
        H[2, 5] = 1  # yaw

        # Innovation
        z = orientation
        y = z - H @ self.state

        # Kalman gain
        S = H @ self.covariance @ H.T + self.R_imu
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.eye(9) - K @ H) @ self.covariance

    def imu_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds * 1e-9

        if self.last_time is not None:
            dt = current_time - self.last_time
            self.predict(dt)

        self.last_time = current_time

        # Extract orientation (simplified - use proper quaternion conversion)
        orientation = np.array([0.0, 0.0, 0.0])  # Placeholder
        self.update_imu(orientation)

        self.publish_pose(msg.header)

    def odom_callback(self, msg):
        # Update with odometry measurement
        pass

    def publish_pose(self, header):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'odom'

        pose_msg.pose.pose.position.x = self.state[0]
        pose_msg.pose.pose.position.y = self.state[1]
        pose_msg.pose.pose.position.z = self.state[2]

        # Set covariance (6x6 for pose)
        cov = np.zeros((6, 6))
        cov[:3, :3] = self.covariance[:3, :3]
        cov[3:, 3:] = self.covariance[3:6, 3:6]
        pose_msg.pose.covariance = cov.flatten().tolist()

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 7.5 Sensor Noise Models

Realistic sensor simulation requires appropriate noise models.

### Noise Types

| Type | Description | Parameters |
|------|-------------|------------|
| Gaussian | Normal distribution | mean, stddev |
| Uniform | Equal probability | min, max |
| Bias | Constant offset | value |
| Drift | Time-varying bias | rate |

### Configuring Noise in Gazebo

```xml
<!-- Comprehensive noise model -->
<sensor name="realistic_lidar" type="gpu_lidar">
  <lidar>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.02</stddev>
    </noise>
  </lidar>
</sensor>

<!-- IMU with bias and noise -->
<sensor name="realistic_imu" type="imu">
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0001</mean>  <!-- Bias -->
          <stddev>0.0002</stddev>
          <bias_mean>0.0001</bias_mean>
          <bias_stddev>0.00001</bias_stddev>
        </noise>
      </x>
    </angular_velocity>
  </imu>
</sensor>
```

---

## Exercises

### Exercise 7.1: Configure LiDAR Sensor

**Objective**: Add a 2D LiDAR to your humanoid robot.

**Difficulty**: Beginner | **Estimated Time**: 30 minutes

#### Instructions

1. Add a LiDAR sensor to the humanoid URDF/SDF
2. Configure range parameters for indoor navigation
3. Add Gaussian noise model
4. Verify scan data in RViz2

#### Expected Outcome

LaserScan messages published on `/scan` topic, visible in RViz2.

---

### Exercise 7.2: Implement Depth Camera

**Objective**: Add RGB-D camera and process depth images.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Add depth camera sensor to robot head
2. Configure image resolution and frame rate
3. Implement depth-to-pointcloud converter
4. Visualize in RViz2

---

### Exercise 7.3: IMU Integration

**Objective**: Use IMU data for orientation tracking.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Add IMU sensor to robot torso
2. Subscribe to IMU messages
3. Implement complementary filter
4. Display orientation in RViz2

---

### Exercise 7.4: Multi-Sensor Fusion

**Objective**: Fuse LiDAR, depth, and IMU data.

**Difficulty**: Advanced | **Estimated Time**: 60 minutes

#### Instructions

1. Set up robot_localization package
2. Configure EKF for sensor fusion
3. Integrate IMU and odometry
4. Validate fused pose output

---

## Summary

In this chapter, you learned:

- **LiDAR sensors** provide 2D/3D range measurements for navigation
- **Depth cameras** enable 3D perception for manipulation
- **IMU sensors** measure orientation and acceleration
- **Noise models** make simulation realistic
- **Sensor fusion** combines multiple sources for robust estimation

---

## References

[1] S. Thrun, W. Burgard, and D. Fox, *Probabilistic Robotics*, MIT Press, 2005.

[2] Open Robotics, "Gazebo Sensors," [Online]. Available: https://gazebosim.org/docs.

[3] Intel, "RealSense SDK 2.0," [Online]. Available: https://github.com/IntelRealSense/librealsense.

[4] T. Moore and D. Stouch, "A Generalized Extended Kalman Filter Implementation for the Robot Operating System," in *IO*, 2014.
