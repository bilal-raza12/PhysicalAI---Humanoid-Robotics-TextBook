---
id: ch11-reinforcement-learning
title: "Chapter 11: Reinforcement Learning for Locomotion"
sidebar_position: 4
---

# Chapter 11: Reinforcement Learning for Locomotion

**Estimated Time**: 6-7 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand** RL fundamentals for robot control
2. **Configure** Isaac Gym for parallel training
3. **Design** reward functions for humanoid locomotion
4. **Train** walking and balancing policies
5. **Transfer** learned policies to simulation and hardware

---

## 11.1 Reinforcement Learning Fundamentals

RL enables robots to learn complex behaviors through trial and error.

### RL Framework

```
┌─────────────────────────────────────────────────────────┐
│                    RL Loop                               │
│                                                          │
│   ┌─────────┐    action     ┌─────────────┐             │
│   │  Agent  │──────────────▶│ Environment │             │
│   │ (Policy)│               │  (Isaac Gym)│             │
│   └─────────┘◀──────────────└─────────────┘             │
│        ▲      state, reward                             │
│        │                                                 │
│   ┌─────────┐                                           │
│   │  Value  │  estimates future rewards                 │
│   │Function │                                           │
│   └─────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Description | Humanoid Example |
|---------|-------------|------------------|
| State | Observation of environment | Joint angles, velocities, IMU |
| Action | Control output | Joint torques or positions |
| Reward | Scalar feedback signal | Forward velocity, stability |
| Policy | State-to-action mapping | Neural network |
| Value | Expected cumulative reward | Long-term benefit estimate |

### Policy Gradient Methods

```python
# PPO pseudocode
def ppo_update(policy, old_policy, states, actions, advantages, clip_epsilon=0.2):
    """Proximal Policy Optimization update."""
    # Compute probability ratio
    pi = policy(states)
    pi_old = old_policy(states)

    ratio = pi.log_prob(actions).exp() / pi_old.log_prob(actions).exp()

    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    objective = torch.min(ratio * advantages, clipped_ratio * advantages)

    # Update policy
    loss = -objective.mean()
    loss.backward()
    optimizer.step()
```

---

## 11.2 Isaac Gym Setup

Isaac Gym provides massively parallel physics simulation for RL.

### Installation

```bash
# Download Isaac Gym from NVIDIA Developer
# https://developer.nvidia.com/isaac-gym

# Extract and install
cd isaacgym/python
pip install -e .

# Verify installation
python -c "import isaacgym; print('Isaac Gym installed successfully')"
```

### Basic Environment

```python
# basic_env.py
from isaacgym import gymapi, gymtorch
import torch

class BasicHumanoidEnv:
    """Basic humanoid environment for Isaac Gym."""

    def __init__(self, num_envs=1024, device="cuda"):
        self.num_envs = num_envs
        self.device = device

        # Initialize gym
        self.gym = gymapi.acquire_gym()

        # Simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        # PhysX parameters
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.rest_offset = 0.01

        # Create simulator
        self.sim = self.gym.create_sim(
            0, 0,  # GPU indices
            gymapi.SIM_PHYSX,
            sim_params
        )

        # Create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # Load humanoid asset
        self.load_humanoid()

        # Create environments
        self.create_envs()

        # Prepare tensors
        self.gym.prepare_sim(self.sim)
        self.acquire_tensors()

    def load_humanoid(self):
        """Load humanoid URDF asset."""
        asset_root = "assets/"
        asset_file = "humanoid.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        self.humanoid_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        self.num_dof = self.gym.get_asset_dof_count(self.humanoid_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.humanoid_asset)

    def create_envs(self):
        """Create parallel environments."""
        env_spacing = 2.0
        envs_per_row = int(self.num_envs ** 0.5)

        lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        self.envs = []
        self.actors = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, envs_per_row)
            self.envs.append(env)

            # Spawn humanoid
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            actor = self.gym.create_actor(
                env, self.humanoid_asset, pose, "humanoid", i, 1
            )
            self.actors.append(actor)

            # Set DOF properties
            dof_props = self.gym.get_actor_dof_properties(env, actor)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"].fill(1000.0)
            dof_props["damping"].fill(100.0)
            self.gym.set_actor_dof_properties(env, actor, dof_props)

    def acquire_tensors(self):
        """Acquire GPU tensors for fast access."""
        # Root state tensor (position, orientation, velocities)
        self.root_states = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        )

        # DOF state tensor (positions, velocities)
        self.dof_states = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        )

        # Contact force tensor
        self.contact_forces = gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim)
        )

        # Reshape tensors
        self.root_pos = self.root_states[:, :3]
        self.root_orient = self.root_states[:, 3:7]
        self.root_lin_vel = self.root_states[:, 7:10]
        self.root_ang_vel = self.root_states[:, 10:13]

        self.dof_pos = self.dof_states[:, :, 0]
        self.dof_vel = self.dof_states[:, :, 1]

    def step(self, actions):
        """Execute one simulation step."""
        # Apply actions
        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(actions)
        )

        # Simulate
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Compute observations and rewards
        obs = self.compute_observations()
        rewards = self.compute_rewards()
        dones = self.compute_dones()

        return obs, rewards, dones, {}

    def compute_observations(self):
        """Compute observation tensor."""
        obs = torch.cat([
            self.root_pos,
            self.root_orient,
            self.root_lin_vel,
            self.root_ang_vel,
            self.dof_pos,
            self.dof_vel,
        ], dim=-1)
        return obs

    def compute_rewards(self):
        """Compute reward tensor."""
        # Forward velocity reward
        forward_vel = self.root_lin_vel[:, 0]
        reward = forward_vel

        # Penalty for falling
        height = self.root_pos[:, 2]
        reward -= 10.0 * (height < 0.5).float()

        # Penalty for excessive torso rotation
        # (orientation close to upright)
        up_vec = torch.tensor([0, 0, 1], device=self.device)
        # Simplified - should use proper quaternion math

        return reward

    def compute_dones(self):
        """Compute done flags."""
        height = self.root_pos[:, 2]
        return height < 0.3

    def reset(self, env_ids=None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset root state
        self.root_states[env_ids, 2] = 1.0  # Height
        self.root_states[env_ids, 7:] = 0.0  # Velocities

        # Reset DOF state
        self.dof_states[env_ids] = 0.0

        # Apply resets
        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states)
        )
        self.gym.set_dof_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states)
        )

        return self.compute_observations()
```

---

## 11.3 Reward Function Design

Reward shaping is critical for learning humanoid locomotion.

### Reward Components

```python
# reward_functions.py
import torch

class HumanoidRewards:
    """Reward functions for humanoid locomotion."""

    def __init__(self, env, config):
        self.env = env
        self.config = config

        # Reward weights
        self.w_forward = config.get("forward_weight", 1.0)
        self.w_alive = config.get("alive_weight", 0.5)
        self.w_energy = config.get("energy_weight", -0.01)
        self.w_posture = config.get("posture_weight", 0.2)
        self.w_smooth = config.get("smooth_weight", -0.1)

    def compute_total_reward(self, obs, actions, prev_actions):
        """Compute total reward."""
        rewards = {}

        # Forward velocity reward
        rewards["forward"] = self.forward_velocity_reward()

        # Alive bonus
        rewards["alive"] = self.alive_reward()

        # Energy penalty
        rewards["energy"] = self.energy_penalty(actions)

        # Posture reward
        rewards["posture"] = self.posture_reward()

        # Action smoothness
        rewards["smooth"] = self.smoothness_penalty(actions, prev_actions)

        # Total
        total = (
            self.w_forward * rewards["forward"] +
            self.w_alive * rewards["alive"] +
            self.w_energy * rewards["energy"] +
            self.w_posture * rewards["posture"] +
            self.w_smooth * rewards["smooth"]
        )

        return total, rewards

    def forward_velocity_reward(self):
        """Reward forward movement."""
        target_vel = self.config.get("target_velocity", 1.0)
        forward_vel = self.env.root_lin_vel[:, 0]

        # Reward for matching target velocity
        vel_error = torch.abs(forward_vel - target_vel)
        reward = torch.exp(-vel_error)

        return reward

    def alive_reward(self):
        """Bonus for staying upright."""
        height = self.env.root_pos[:, 2]
        min_height = self.config.get("min_height", 0.8)

        alive = (height > min_height).float()
        return alive

    def energy_penalty(self, actions):
        """Penalty for high torque usage."""
        # L2 norm of actions
        energy = torch.sum(actions ** 2, dim=-1)
        return energy

    def posture_reward(self):
        """Reward upright posture."""
        # Get torso orientation
        quat = self.env.root_orient

        # Compute up vector in body frame
        # Simplified - proper implementation needs quaternion rotation
        # Reward when torso z-axis aligns with world z-axis

        return torch.ones(self.env.num_envs, device=self.env.device)

    def smoothness_penalty(self, actions, prev_actions):
        """Penalty for jerky actions."""
        if prev_actions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        action_diff = actions - prev_actions
        jerk = torch.sum(action_diff ** 2, dim=-1)
        return jerk

    def foot_contact_reward(self):
        """Reward proper foot contacts during gait."""
        # Get foot contact forces
        left_foot_contact = self.env.contact_forces[:, self.env.left_foot_idx]
        right_foot_contact = self.env.contact_forces[:, self.env.right_foot_idx]

        # Encourage alternating foot contacts
        left_contact = torch.norm(left_foot_contact, dim=-1) > 1.0
        right_contact = torch.norm(right_foot_contact, dim=-1) > 1.0

        # At least one foot should be in contact
        any_contact = (left_contact | right_contact).float()

        return any_contact

    def gait_symmetry_reward(self):
        """Reward symmetric gait patterns."""
        # Compare left and right leg joint angles
        # Simplified - need proper joint indexing
        return torch.zeros(self.env.num_envs, device=self.env.device)
```

### Curriculum Learning

```python
# curriculum.py
class LocomotionCurriculum:
    """Curriculum for progressive difficulty."""

    def __init__(self, env):
        self.env = env
        self.level = 0
        self.max_level = 5

        self.curricula = {
            0: {"target_vel": 0.5, "terrain": "flat"},
            1: {"target_vel": 1.0, "terrain": "flat"},
            2: {"target_vel": 1.5, "terrain": "flat"},
            3: {"target_vel": 1.5, "terrain": "rough"},
            4: {"target_vel": 2.0, "terrain": "rough"},
            5: {"target_vel": 2.0, "terrain": "stairs"},
        }

    def get_config(self):
        """Get current curriculum configuration."""
        return self.curricula[self.level]

    def update(self, success_rate):
        """Update curriculum based on performance."""
        if success_rate > 0.8 and self.level < self.max_level:
            self.level += 1
            print(f"Curriculum advanced to level {self.level}")
            self.apply_curriculum()
        elif success_rate < 0.3 and self.level > 0:
            self.level -= 1
            print(f"Curriculum decreased to level {self.level}")
            self.apply_curriculum()

    def apply_curriculum(self):
        """Apply current curriculum settings."""
        config = self.get_config()

        # Update reward function
        self.env.reward_fn.config["target_velocity"] = config["target_vel"]

        # Update terrain
        if config["terrain"] == "rough":
            self.env.add_terrain_roughness(0.05)
        elif config["terrain"] == "stairs":
            self.env.add_stairs()
```

---

## 11.4 Training Pipeline

### PPO Implementation

```python
# ppo_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.backbone(obs)
        return features

    def get_action(self, obs, deterministic=False):
        features = self.forward(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_std)

        if deterministic:
            return mean

        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob

    def evaluate(self, obs, actions):
        features = self.forward(obs)

        # Actor
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        # Critic
        value = self.critic(features).squeeze(-1)

        return log_prob, value, entropy


class PPOTrainer:
    """PPO training loop."""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = config.get("device", "cuda")

        # Create policy
        obs_dim = env.compute_observations().shape[-1]
        action_dim = env.num_dof

        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.get("learning_rate", 3e-4)
        )

        # PPO parameters
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.epochs = config.get("epochs", 5)
        self.batch_size = config.get("batch_size", 4096)

        # Storage
        self.storage = RolloutStorage(
            config.get("horizon", 24),
            env.num_envs,
            obs_dim,
            action_dim,
            self.device
        )

    def train(self, num_iterations):
        """Main training loop."""
        obs = self.env.reset()
        prev_actions = None

        for iteration in range(num_iterations):
            # Collect rollouts
            for step in range(self.config.get("horizon", 24)):
                with torch.no_grad():
                    actions, log_probs = self.policy.get_action(obs)
                    values = self.policy.critic(
                        self.policy.forward(obs)
                    ).squeeze(-1)

                # Step environment
                next_obs, rewards, dones, _ = self.env.step(actions)

                # Store transition
                self.storage.add(
                    obs, actions, rewards, dones,
                    log_probs, values
                )

                obs = next_obs
                prev_actions = actions

                # Reset done environments
                done_envs = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(done_envs) > 0:
                    obs[done_envs] = self.env.reset(done_envs)[done_envs]

            # Compute returns and advantages
            with torch.no_grad():
                last_values = self.policy.critic(
                    self.policy.forward(obs)
                ).squeeze(-1)
            self.storage.compute_returns(last_values, self.gamma, self.gae_lambda)

            # PPO update
            policy_loss, value_loss, entropy = self.ppo_update()

            # Log progress
            if iteration % 10 == 0:
                mean_reward = self.storage.rewards.mean().item()
                print(f"Iter {iteration}: reward={mean_reward:.3f}, "
                      f"policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")

            # Clear storage
            self.storage.clear()

        return self.policy

    def ppo_update(self):
        """Perform PPO update."""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for epoch in range(self.epochs):
            for batch in self.storage.get_batches(self.batch_size):
                obs, actions, old_log_probs, returns, advantages = batch

                # Evaluate current policy
                log_probs, values, entropy = self.policy.evaluate(obs, actions)

                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (returns - values).pow(2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.get("value_coef", 0.5) * value_loss +
                    self.config.get("entropy_coef", 0.01) * entropy_loss
                )

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.get("max_grad_norm", 1.0)
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = self.epochs * (self.storage.size // self.batch_size)
        return (
            total_policy_loss / n_updates,
            total_value_loss / n_updates,
            total_entropy / n_updates
        )


class RolloutStorage:
    """Store rollout data for PPO."""

    def __init__(self, horizon, num_envs, obs_dim, action_dim, device):
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = device
        self.size = horizon * num_envs

        # Allocate buffers
        self.obs = torch.zeros(horizon, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device)
        self.log_probs = torch.zeros(horizon, num_envs, device=device)
        self.values = torch.zeros(horizon, num_envs, device=device)
        self.returns = torch.zeros(horizon, num_envs, device=device)
        self.advantages = torch.zeros(horizon, num_envs, device=device)

        self.step = 0

    def add(self, obs, actions, rewards, dones, log_probs, values):
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.log_probs[self.step] = log_probs
        self.values[self.step] = values
        self.step += 1

    def compute_returns(self, last_values, gamma, gae_lambda):
        """Compute GAE returns and advantages."""
        gae = 0
        for step in reversed(range(self.horizon)):
            if step == self.horizon - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step] +
                gamma * next_values * (1 - self.dones[step]) -
                self.values[step]
            )
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_batches(self, batch_size):
        """Generate random batches."""
        indices = torch.randperm(self.size, device=self.device)

        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # Flatten and index
            flat_obs = self.obs.view(-1, self.obs.shape[-1])
            flat_actions = self.actions.view(-1, self.actions.shape[-1])
            flat_log_probs = self.log_probs.view(-1)
            flat_returns = self.returns.view(-1)
            flat_advantages = self.advantages.view(-1)

            yield (
                flat_obs[batch_indices],
                flat_actions[batch_indices],
                flat_log_probs[batch_indices],
                flat_returns[batch_indices],
                flat_advantages[batch_indices],
            )

    def clear(self):
        self.step = 0
```

---

## 11.5 Policy Deployment

### Export to ONNX

```python
# export_policy.py
import torch

def export_to_onnx(policy, obs_dim, output_path):
    """Export trained policy to ONNX."""
    policy.eval()

    # Dummy input
    dummy_input = torch.randn(1, obs_dim, device="cuda")

    # Export
    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"}
        }
    )

    print(f"Policy exported to {output_path}")
```

### ROS 2 Policy Node

```python
#!/usr/bin/env python3
"""
policy_node.py
Deploy RL policy as ROS 2 node.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import torch
import numpy as np

class PolicyNode(Node):
    def __init__(self):
        super().__init__('rl_policy_node')

        # Load policy
        self.policy = torch.jit.load('humanoid_policy.pt')
        self.policy.eval()

        # State subscriber
        self.state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10
        )

        # Command publisher
        self.cmd_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )

        # State buffer
        self.last_obs = None

        self.get_logger().info('RL Policy node started')

    def state_callback(self, msg):
        # Build observation
        obs = self.build_observation(msg)

        # Get action from policy
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.policy(obs_tensor).squeeze(0).numpy()

        # Publish command
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = list(msg.name)
        cmd.position = action.tolist()

        self.cmd_pub.publish(cmd)

    def build_observation(self, joint_state):
        # Simplified - actual implementation needs full state
        positions = np.array(joint_state.position)
        velocities = np.array(joint_state.velocity)

        return np.concatenate([positions, velocities])

def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## Exercises

### Exercise 11.1: Setup Isaac Gym

**Objective**: Install and run Isaac Gym examples.

**Difficulty**: Beginner | **Estimated Time**: 45 minutes

#### Instructions

1. Download and install Isaac Gym
2. Run the humanoid example
3. Observe training progress
4. Visualize learned behavior

---

### Exercise 11.2: Design Custom Reward

**Objective**: Create a reward function for standing balance.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Implement posture reward component
2. Add penalty for excessive motion
3. Include foot contact reward
4. Train and evaluate policy

---

### Exercise 11.3: Train Walking Policy

**Objective**: Train humanoid to walk forward.

**Difficulty**: Intermediate | **Estimated Time**: 90 minutes

#### Instructions

1. Configure walking reward function
2. Set up curriculum learning
3. Train for 1000 iterations
4. Evaluate walking quality

---

### Exercise 11.4: Deploy Policy

**Objective**: Deploy trained policy to Isaac Sim.

**Difficulty**: Advanced | **Estimated Time**: 60 minutes

#### Instructions

1. Export policy to TorchScript
2. Create ROS 2 policy node
3. Connect to Isaac Sim
4. Evaluate real-time performance

---

## Summary

In this chapter, you learned:

- **RL fundamentals** enable learning from experience
- **Isaac Gym** provides GPU-accelerated parallel training
- **Reward design** shapes learned behaviors
- **PPO** is effective for continuous control
- **Policy deployment** bridges sim-to-real

---

## References

[1] J. Schulman et al., "Proximal Policy Optimization Algorithms," *arXiv:1707.06347*, 2017.

[2] NVIDIA, "Isaac Gym," [Online]. Available: https://developer.nvidia.com/isaac-gym.

[3] X. B. Peng et al., "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills," *ACM Trans. Graph.*, 2018.

[4] T. Haarnoja et al., "Learning to Walk via Deep Reinforcement Learning," in *RSS*, 2019.
