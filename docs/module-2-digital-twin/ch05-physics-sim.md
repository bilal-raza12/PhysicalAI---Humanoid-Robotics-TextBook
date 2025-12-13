---
id: ch05-physics-sim
title: "Chapter 5: Introduction to Physics Simulation"
sidebar_position: 2
---

# Chapter 5: Introduction to Physics Simulation

**Estimated Time**: 5-6 hours | **Exercises**: 3

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand** the fundamentals of physics engines and their role in robotics
2. **Configure** physics parameters for realistic humanoid simulation
3. **Implement** collision detection and response systems
4. **Optimize** simulation performance for real-time applications
5. **Validate** simulation accuracy against real-world behavior

---

## 5.1 Physics Engine Fundamentals

Physics simulation is the backbone of digital twin technology. It allows us to test robot behaviors in a virtual environment before deploying to physical hardware.

### Why Physics Simulation Matters

| Benefit | Description |
|---------|-------------|
| **Safety** | Test dangerous scenarios without risking hardware |
| **Speed** | Run thousands of trials faster than real-time |
| **Cost** | No wear and tear on physical robots |
| **Repeatability** | Perfect reproduction of test conditions |
| **Debugging** | Full observability of internal states |

### Common Physics Engines

| Engine | Type | Strengths | Use Case |
|--------|------|-----------|----------|
| **ODE** | Rigid body | Stable, well-tested | General robotics |
| **Bullet** | Rigid body | Fast, GPU support | Gaming, VR |
| **DART** | Rigid body | Accurate dynamics | Research |
| **MuJoCo** | Rigid body | Fast, contact-rich | RL, control |
| **PhysX** | Rigid body | Real-time, GPU | NVIDIA Isaac |

### The Simulation Loop

```
┌─────────────────────────────────────────────┐
│              SIMULATION LOOP                 │
├─────────────────────────────────────────────┤
│                                              │
│  1. Read sensor data                         │
│         │                                    │
│         ▼                                    │
│  2. Apply control inputs                     │
│         │                                    │
│         ▼                                    │
│  3. Collision detection                      │
│         │                                    │
│         ▼                                    │
│  4. Solve constraints                        │
│         │                                    │
│         ▼                                    │
│  5. Integrate dynamics                       │
│         │                                    │
│         ▼                                    │
│  6. Update world state                       │
│         │                                    │
│         └──────────────────────────┐        │
│                                    │        │
└────────────────────────────────────┘        │
         ▲                                     │
         └─────────────────────────────────────┘
```

---

## 5.2 Rigid Body Dynamics

### Newton-Euler Equations

The motion of a rigid body is governed by:

**Linear motion**: F = ma

**Angular motion**: τ = Iα

Where:
- F = Force vector
- m = Mass
- a = Linear acceleration
- τ = Torque vector
- I = Inertia tensor
- α = Angular acceleration

### Inertia Tensor

For a humanoid robot, each link has an inertia tensor:

```xml
<inertial>
  <mass value="5.0"/>
  <inertia ixx="0.1" ixy="0" ixz="0"
           iyy="0.1" iyz="0"
           izz="0.05"/>
</inertial>
```

### Common Shapes and Inertias

| Shape | Ixx | Iyy | Izz |
|-------|-----|-----|-----|
| Solid sphere | (2/5)mr² | (2/5)mr² | (2/5)mr² |
| Solid cylinder (z-axis) | (1/12)m(3r²+h²) | (1/12)m(3r²+h²) | (1/2)mr² |
| Solid box | (1/12)m(y²+z²) | (1/12)m(x²+z²) | (1/12)m(x²+y²) |

---

## 5.3 Collision Detection

### Collision Geometry Types

```xml
<!-- Primitive shapes (fast) -->
<collision>
  <geometry>
    <box size="0.1 0.1 0.3"/>
  </geometry>
</collision>

<!-- Mesh (accurate but slow) -->
<collision>
  <geometry>
    <mesh filename="package://robot/meshes/arm.stl"/>
  </geometry>
</collision>
```

### Collision Detection Pipeline

1. **Broad phase**: Quick rejection of distant objects (AABB, spatial hashing)
2. **Narrow phase**: Precise intersection tests (GJK, SAT algorithms)
3. **Contact generation**: Compute contact points, normals, depths

### Contact Properties

```xml
<gazebo reference="foot">
  <mu1>1.0</mu1>           <!-- Friction coefficient 1 -->
  <mu2>1.0</mu2>           <!-- Friction coefficient 2 -->
  <kp>1000000.0</kp>       <!-- Contact stiffness -->
  <kd>100.0</kd>           <!-- Contact damping -->
  <minDepth>0.001</minDepth>
  <maxVel>1.0</maxVel>
</gazebo>
```

---

## 5.4 Simulation Parameters

### Time Step Selection

```python
# Typical values for humanoid simulation
physics_config = {
    'time_step': 0.001,      # 1ms (1000 Hz)
    'real_time_factor': 1.0,  # Real-time
    'max_step_size': 0.002,   # Maximum step
    'solver_iterations': 50,  # Constraint solver
}
```

**Guidelines**:
- Smaller time steps = more accurate but slower
- Humanoid balance: 500-1000 Hz recommended
- Manipulation: 100-500 Hz typically sufficient

### Solver Configuration

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `iterations` | Constraint solver passes | 50-100 |
| `sor` | Successive over-relaxation | 1.3 |
| `precon_iters` | Preconditioner iterations | 0-4 |
| `iters` | Total iterations | 50-200 |

---

## 5.5 Simulation Accuracy

### Sources of Error

1. **Discretization**: Finite time steps approximate continuous motion
2. **Constraint drift**: Accumulated errors in joint constraints
3. **Contact modeling**: Simplified friction and collision models
4. **Numerical precision**: Floating-point limitations

### Validation Techniques

```python
def validate_simulation(sim_data, real_data):
    """Compare simulation to real-world data."""

    # Compute RMSE for joint positions
    rmse_pos = np.sqrt(np.mean(
        (sim_data['positions'] - real_data['positions'])**2
    ))

    # Compute correlation
    correlation = np.corrcoef(
        sim_data['velocities'].flatten(),
        real_data['velocities'].flatten()
    )[0, 1]

    return {
        'rmse_position': rmse_pos,
        'velocity_correlation': correlation
    }
```

---

## Exercises

### Exercise 5.1: Configure Physics Parameters

**Objective**: Set up physics parameters for stable humanoid simulation.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Create a Gazebo world file with custom physics settings
2. Configure time step, solver iterations, and gravity
3. Test stability with a simple falling object
4. Measure simulation performance (RTF)

#### Expected Outcome

Physics configuration that maintains real-time factor > 0.9.

#### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Simulation unstable | Time step too large | Reduce to 0.001s |
| Objects exploding | Contact stiffness too high | Reduce kp value |
| Slow simulation | Too many solver iterations | Reduce or use GPU |

---

### Exercise 5.2: Test Collision Detection

**Objective**: Implement and test collision detection for humanoid limbs.

**Difficulty**: Intermediate | **Estimated Time**: 30 minutes

#### Instructions

1. Add collision geometry to a humanoid URDF
2. Spawn the robot in Gazebo
3. Test self-collision detection
4. Verify contact forces are published

---

### Exercise 5.3: Optimize Performance

**Objective**: Achieve real-time simulation for a full humanoid.

**Difficulty**: Advanced | **Estimated Time**: 45 minutes

#### Instructions

1. Profile simulation performance with `gz stats`
2. Simplify collision meshes where possible
3. Adjust solver parameters for speed
4. Enable GPU acceleration if available

---

## Summary

In this chapter, you learned:

- **Physics engines** simulate rigid body dynamics for robotics
- **Collision detection** uses broad/narrow phase algorithms
- **Time step selection** balances accuracy and performance
- **Contact parameters** affect friction and stability
- **Validation** ensures simulation matches reality

---

## References

[1] R. Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2008.

[2] E. Todorov, T. Erez, and Y. Tassa, "MuJoCo: A physics engine for model-based control," in *IROS*, 2012.

[3] Open Robotics, "Gazebo Physics," [Online]. Available: https://gazebosim.org/docs.

[4] N. Koenig and A. Howard, "Design and use paradigms for Gazebo," in *IROS*, 2004.
