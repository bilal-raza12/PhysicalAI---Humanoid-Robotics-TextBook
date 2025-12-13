---
id: ch12-sim2real
title: "Chapter 12: Sim-to-Real Transfer Strategies"
sidebar_position: 5
---

# Chapter 12: Sim-to-Real Transfer Strategies

**Estimated Time**: 5-6 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Understand** the sim-to-real gap and its causes
2. **Apply** domain randomization techniques
3. **Implement** system identification for simulation accuracy
4. **Use** real-world fine-tuning approaches
5. **Evaluate** transfer quality and debug failures

---

## 12.1 The Sim-to-Real Gap

Policies trained in simulation often fail when deployed on real robots.

### Sources of Reality Gap

```
┌─────────────────────────────────────────────────────────┐
│                  Reality Gap Sources                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Physics         Sensors          Actuators            │
│   ────────        ───────          ─────────            │
│   • Friction      • Noise          • Delays             │
│   • Mass/Inertia  • Latency        • Backlash           │
│   • Contact       • Calibration    • Torque limits      │
│   • Deformation   • Occlusion      • Temperature        │
│                                                          │
│   Environment     Perception       Timing               │
│   ───────────     ──────────       ──────               │
│   • Lighting      • Distortion     • Control rate       │
│   • Textures      • Color shift    • Computation        │
│   • Clutter       • Reflections    • Communication      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Gap Quantification

```python
# gap_analysis.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class SimRealComparison:
    """Compare simulation and real-world trajectories."""
    sim_trajectory: np.ndarray
    real_trajectory: np.ndarray
    timestamps: np.ndarray

def compute_trajectory_error(comparison: SimRealComparison) -> Dict:
    """Quantify sim-to-real trajectory error."""
    sim = comparison.sim_trajectory
    real = comparison.real_trajectory

    # Position error
    pos_error = np.linalg.norm(sim[:, :3] - real[:, :3], axis=1)

    # Velocity error
    vel_error = np.linalg.norm(sim[:, 3:6] - real[:, 3:6], axis=1)

    # Compute statistics
    results = {
        "position_error": {
            "mean": np.mean(pos_error),
            "std": np.std(pos_error),
            "max": np.max(pos_error),
            "rmse": np.sqrt(np.mean(pos_error ** 2))
        },
        "velocity_error": {
            "mean": np.mean(vel_error),
            "std": np.std(vel_error),
            "max": np.max(vel_error),
            "rmse": np.sqrt(np.mean(vel_error ** 2))
        }
    }

    return results

def identify_gap_sources(
    sim_data: np.ndarray,
    real_data: np.ndarray,
    joint_names: List[str]
) -> Dict:
    """Identify which components contribute most to gap."""
    gaps = {}

    for i, name in enumerate(joint_names):
        sim_joint = sim_data[:, i]
        real_joint = real_data[:, i]

        # Phase delay
        correlation = np.correlate(sim_joint, real_joint, mode='full')
        delay_samples = np.argmax(correlation) - len(sim_joint) + 1

        # Amplitude difference
        amplitude_ratio = np.std(real_joint) / (np.std(sim_joint) + 1e-6)

        # Bias
        bias = np.mean(real_joint) - np.mean(sim_joint)

        gaps[name] = {
            "delay_samples": delay_samples,
            "amplitude_ratio": amplitude_ratio,
            "bias": bias,
            "rmse": np.sqrt(np.mean((sim_joint - real_joint) ** 2))
        }

    return gaps
```

---

## 12.2 Domain Randomization

Randomize simulation parameters to create robust policies.

### Comprehensive Randomization

```python
# domain_randomization.py
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class RandomizationConfig:
    """Configuration for domain randomization."""
    # Physics
    friction_range: Tuple[float, float] = (0.5, 1.5)
    mass_scale_range: Tuple[float, float] = (0.8, 1.2)
    inertia_scale_range: Tuple[float, float] = (0.8, 1.2)

    # Actuators
    motor_strength_range: Tuple[float, float] = (0.8, 1.2)
    motor_delay_range: Tuple[float, float] = (0.0, 0.02)
    damping_range: Tuple[float, float] = (0.5, 2.0)

    # Sensors
    obs_noise_std: float = 0.01
    obs_delay_range: Tuple[int, int] = (0, 2)  # frames

    # External forces
    push_force_range: Tuple[float, float] = (0, 50)
    push_interval_range: Tuple[int, int] = (100, 500)  # steps


class DomainRandomizer:
    """Apply domain randomization during training."""

    def __init__(self, env, config: RandomizationConfig):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng()

        # Track randomized parameters
        self.current_params = {}

    def randomize_physics(self):
        """Randomize physics parameters."""
        # Friction
        friction = self.rng.uniform(*self.config.friction_range)
        self.env.set_friction(friction)

        # Mass
        mass_scale = self.rng.uniform(*self.config.mass_scale_range)
        self.env.scale_masses(mass_scale)

        # Inertia
        inertia_scale = self.rng.uniform(*self.config.inertia_scale_range)
        self.env.scale_inertias(inertia_scale)

        self.current_params.update({
            "friction": friction,
            "mass_scale": mass_scale,
            "inertia_scale": inertia_scale
        })

    def randomize_actuators(self):
        """Randomize actuator properties."""
        num_joints = self.env.num_dof

        # Motor strength per joint
        strengths = self.rng.uniform(
            *self.config.motor_strength_range,
            size=num_joints
        )
        self.env.set_motor_strengths(strengths)

        # Damping per joint
        dampings = self.rng.uniform(
            *self.config.damping_range,
            size=num_joints
        )
        self.env.set_joint_dampings(dampings)

        # Delay (common)
        delay = self.rng.uniform(*self.config.motor_delay_range)
        self.env.set_action_delay(delay)

        self.current_params.update({
            "motor_strengths": strengths.tolist(),
            "dampings": dampings.tolist(),
            "action_delay": delay
        })

    def add_observation_noise(self, obs):
        """Add noise to observations."""
        noise = self.rng.normal(
            0,
            self.config.obs_noise_std,
            size=obs.shape
        )
        return obs + noise

    def add_observation_delay(self, obs_buffer):
        """Add delay to observations."""
        delay = self.rng.integers(*self.config.obs_delay_range)
        if delay > 0 and len(obs_buffer) >= delay:
            return obs_buffer[-delay]
        return obs_buffer[-1]

    def apply_random_push(self, step_count):
        """Apply random external forces."""
        if not hasattr(self, 'next_push_step'):
            self.next_push_step = self.rng.integers(
                *self.config.push_interval_range
            )

        if step_count >= self.next_push_step:
            # Apply push
            force_magnitude = self.rng.uniform(*self.config.push_force_range)
            force_direction = self.rng.uniform(-1, 1, size=3)
            force_direction /= np.linalg.norm(force_direction)

            force = force_magnitude * force_direction
            self.env.apply_external_force(force)

            # Schedule next push
            self.next_push_step = step_count + self.rng.integers(
                *self.config.push_interval_range
            )

    def on_reset(self):
        """Randomize on environment reset."""
        self.randomize_physics()
        self.randomize_actuators()
```

### Adaptive Domain Randomization

```python
# adaptive_randomization.py
import numpy as np
from collections import deque

class AdaptiveDomainRandomizer:
    """Adapt randomization based on policy performance."""

    def __init__(self, base_config: RandomizationConfig):
        self.config = base_config
        self.performance_history = deque(maxlen=100)
        self.difficulty = 0.5  # 0 = easy, 1 = hard

    def update_difficulty(self, episode_return: float, success: bool):
        """Update difficulty based on performance."""
        self.performance_history.append((episode_return, success))

        if len(self.performance_history) >= 20:
            recent_success_rate = np.mean(
                [s for _, s in list(self.performance_history)[-20:]]
            )

            # Increase difficulty if doing well
            if recent_success_rate > 0.8:
                self.difficulty = min(1.0, self.difficulty + 0.1)
            # Decrease if struggling
            elif recent_success_rate < 0.3:
                self.difficulty = max(0.0, self.difficulty - 0.1)

    def get_randomization_ranges(self):
        """Get ranges scaled by difficulty."""
        def scale_range(base_range, scale):
            center = (base_range[0] + base_range[1]) / 2
            width = (base_range[1] - base_range[0]) / 2
            new_width = width * scale
            return (center - new_width, center + new_width)

        scaled = RandomizationConfig(
            friction_range=scale_range(
                self.config.friction_range, self.difficulty
            ),
            mass_scale_range=scale_range(
                self.config.mass_scale_range, self.difficulty
            ),
            motor_strength_range=scale_range(
                self.config.motor_strength_range, self.difficulty
            ),
            obs_noise_std=self.config.obs_noise_std * self.difficulty,
            push_force_range=(
                self.config.push_force_range[0],
                self.config.push_force_range[1] * self.difficulty
            )
        )

        return scaled
```

---

## 12.3 System Identification

Improve simulation accuracy by identifying real system parameters.

### Parameter Estimation

```python
# system_identification.py
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Dict, List

class SystemIdentification:
    """Identify system parameters from real data."""

    def __init__(
        self,
        sim_function: Callable,
        real_data: np.ndarray,
        param_bounds: Dict[str, tuple]
    ):
        self.sim_function = sim_function
        self.real_data = real_data
        self.param_bounds = param_bounds

    def objective(self, params: np.ndarray) -> float:
        """Compute error between simulation and real data."""
        # Run simulation with parameters
        sim_data = self.sim_function(params)

        # Compute trajectory error
        error = np.mean((sim_data - self.real_data) ** 2)

        return error

    def identify(self, initial_guess: np.ndarray) -> Dict:
        """Run system identification optimization."""
        bounds = [
            self.param_bounds[name]
            for name in sorted(self.param_bounds.keys())
        ]

        result = minimize(
            self.objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )

        identified_params = {
            name: result.x[i]
            for i, name in enumerate(sorted(self.param_bounds.keys()))
        }

        return {
            "params": identified_params,
            "error": result.fun,
            "success": result.success
        }


class BayesianSystemID:
    """Bayesian approach to system identification."""

    def __init__(self, param_priors: Dict[str, tuple]):
        """
        param_priors: Dict of parameter name -> (mean, std)
        """
        self.priors = param_priors
        self.posterior_samples = []

    def update_posterior(
        self,
        real_trajectory: np.ndarray,
        sim_function: Callable,
        n_samples: int = 100
    ):
        """Update parameter posterior using MCMC."""
        import pymc3 as pm

        with pm.Model():
            # Define priors
            params = {}
            for name, (mean, std) in self.priors.items():
                params[name] = pm.Normal(name, mu=mean, sigma=std)

            # Likelihood based on trajectory matching
            # Simplified - actual implementation needs proper likelihood

            # Sample
            trace = pm.sample(n_samples, return_inferencedata=False)
            self.posterior_samples.append(trace)

    def get_posterior_mean(self) -> Dict[str, float]:
        """Get posterior mean estimates."""
        if not self.posterior_samples:
            return {}

        latest_trace = self.posterior_samples[-1]
        means = {}
        for name in self.priors.keys():
            means[name] = np.mean(latest_trace[name])

        return means
```

### Motor Dynamics Identification

```python
# motor_identification.py
import numpy as np
from scipy.signal import savgol_filter

class MotorModelIdentification:
    """Identify motor dynamics from data."""

    def __init__(self, dt: float):
        self.dt = dt

    def identify_from_step_response(
        self,
        command: np.ndarray,
        response: np.ndarray
    ) -> Dict:
        """Identify motor parameters from step response."""
        # Find step time
        step_idx = np.argmax(np.diff(command) > 0.1)

        # Extract response after step
        step_response = response[step_idx:]
        target = command[step_idx + 1]

        # Fit first-order model: tau * dx/dt + x = K * u
        # Find time constant (63.2% of final value)
        final_value = np.mean(step_response[-10:])
        threshold = step_response[0] + 0.632 * (final_value - step_response[0])
        tau_idx = np.argmax(step_response >= threshold)
        tau = tau_idx * self.dt

        # Gain
        K = final_value / target

        return {
            "time_constant": tau,
            "gain": K,
            "damping_estimate": tau  # Simplified relationship
        }

    def identify_friction(
        self,
        velocity: np.ndarray,
        torque: np.ndarray
    ) -> Dict:
        """Identify friction parameters."""
        # Filter velocity
        vel_filtered = savgol_filter(velocity, 11, 3)

        # Compute acceleration
        accel = np.gradient(vel_filtered, self.dt)

        # At steady state (low acceleration), torque = friction
        steady_mask = np.abs(accel) < 0.1

        if np.sum(steady_mask) < 10:
            return {"error": "Not enough steady state data"}

        steady_vel = vel_filtered[steady_mask]
        steady_torque = torque[steady_mask]

        # Linear regression: torque = coulomb + viscous * vel
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(steady_vel.reshape(-1, 1), steady_torque)

        return {
            "coulomb_friction": abs(model.intercept_),
            "viscous_friction": model.coef_[0]
        }
```

---

## 12.4 Real-World Fine-Tuning

Adapt policies using limited real-world experience.

### Online Adaptation

```python
# online_adaptation.py
import torch
import torch.nn as nn
from collections import deque

class OnlineAdapter:
    """Adapt policy online during deployment."""

    def __init__(
        self,
        policy: nn.Module,
        adaptation_lr: float = 1e-4,
        buffer_size: int = 1000
    ):
        self.policy = policy
        self.adaptation_lr = adaptation_lr

        # Replay buffer for recent experience
        self.buffer = deque(maxlen=buffer_size)

        # Separate optimizer for adaptation
        # Only adapt later layers
        adapt_params = list(policy.actor_mean.parameters())
        self.optimizer = torch.optim.Adam(
            adapt_params,
            lr=adaptation_lr
        )

    def add_experience(self, obs, action, reward, next_obs, done):
        """Store experience for adaptation."""
        self.buffer.append((obs, action, reward, next_obs, done))

    def adapt_step(self, batch_size: int = 64):
        """Perform one adaptation step."""
        if len(self.buffer) < batch_size:
            return None

        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]

        obs = torch.stack([b[0] for b in batch])
        actions = torch.stack([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch])

        # Simple behavior cloning objective
        # Increase likelihood of high-reward actions
        predicted_actions = self.policy.get_action(obs, deterministic=True)
        action_loss = ((predicted_actions - actions) ** 2).mean()

        # Weight by reward
        weights = torch.softmax(rewards, dim=0)
        weighted_loss = (action_loss * weights).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        return weighted_loss.item()


class MAML_Adapter:
    """Model-Agnostic Meta-Learning for quick adaptation."""

    def __init__(
        self,
        policy: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 1e-3
    ):
        self.policy = policy
        self.inner_lr = inner_lr
        self.outer_optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=outer_lr
        )

    def inner_loop(
        self,
        support_data,
        n_steps: int = 5
    ):
        """Fast adaptation on support set."""
        # Clone policy for adaptation
        adapted_policy = self.clone_policy()

        for _ in range(n_steps):
            loss = self.compute_loss(adapted_policy, support_data)

            # Manual gradient descent (not using optimizer)
            grads = torch.autograd.grad(
                loss,
                adapted_policy.parameters(),
                create_graph=True
            )

            # Update parameters
            for param, grad in zip(adapted_policy.parameters(), grads):
                param.data = param.data - self.inner_lr * grad

        return adapted_policy

    def outer_loop(self, tasks: List):
        """Meta-learning over multiple tasks."""
        meta_loss = 0

        for task in tasks:
            support_data, query_data = task

            # Adapt to task
            adapted_policy = self.inner_loop(support_data)

            # Evaluate on query set
            query_loss = self.compute_loss(adapted_policy, query_data)
            meta_loss += query_loss

        # Update meta-parameters
        self.outer_optimizer.zero_grad()
        meta_loss.backward()
        self.outer_optimizer.step()

        return meta_loss.item() / len(tasks)

    def clone_policy(self):
        """Create a differentiable copy of the policy."""
        import copy
        return copy.deepcopy(self.policy)

    def compute_loss(self, policy, data):
        """Compute policy loss on data."""
        obs, actions, rewards = data
        predicted = policy.get_action(obs, deterministic=True)
        return ((predicted - actions) ** 2).mean()
```

---

## 12.5 Transfer Evaluation

### Evaluation Framework

```python
# transfer_evaluation.py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TransferMetrics:
    """Metrics for evaluating sim-to-real transfer."""
    success_rate: float
    mean_episode_length: float
    mean_reward: float
    tracking_error: float
    stability_score: float

class TransferEvaluator:
    """Evaluate sim-to-real transfer quality."""

    def __init__(self, sim_env, real_env=None):
        self.sim_env = sim_env
        self.real_env = real_env

    def evaluate_in_sim(
        self,
        policy,
        n_episodes: int = 100,
        randomization: bool = True
    ) -> TransferMetrics:
        """Evaluate policy in simulation."""
        successes = []
        episode_lengths = []
        rewards = []
        tracking_errors = []

        for _ in range(n_episodes):
            if randomization:
                self.sim_env.randomize()

            obs = self.sim_env.reset()
            episode_reward = 0
            steps = 0

            while True:
                action = policy(obs)
                obs, reward, done, info = self.sim_env.step(action)

                episode_reward += reward
                steps += 1

                if 'tracking_error' in info:
                    tracking_errors.append(info['tracking_error'])

                if done:
                    successes.append(info.get('success', False))
                    break

            episode_lengths.append(steps)
            rewards.append(episode_reward)

        return TransferMetrics(
            success_rate=np.mean(successes),
            mean_episode_length=np.mean(episode_lengths),
            mean_reward=np.mean(rewards),
            tracking_error=np.mean(tracking_errors) if tracking_errors else 0,
            stability_score=self.compute_stability_score(rewards)
        )

    def compute_stability_score(self, rewards: List[float]) -> float:
        """Score based on reward consistency."""
        if len(rewards) < 2:
            return 1.0
        return 1.0 / (1.0 + np.std(rewards) / (np.mean(rewards) + 1e-6))

    def compare_sim_real(
        self,
        policy,
        sim_episodes: int = 50,
        real_episodes: int = 10
    ) -> Dict:
        """Compare performance in sim vs real."""
        sim_metrics = self.evaluate_in_sim(policy, sim_episodes)

        real_metrics = None
        if self.real_env is not None:
            real_metrics = self.evaluate_on_real(policy, real_episodes)

        return {
            "sim": sim_metrics,
            "real": real_metrics,
            "transfer_gap": self.compute_transfer_gap(
                sim_metrics, real_metrics
            ) if real_metrics else None
        }

    def compute_transfer_gap(
        self,
        sim_metrics: TransferMetrics,
        real_metrics: TransferMetrics
    ) -> Dict:
        """Quantify sim-to-real performance gap."""
        return {
            "success_rate_gap": sim_metrics.success_rate - real_metrics.success_rate,
            "reward_gap_ratio": (
                (sim_metrics.mean_reward - real_metrics.mean_reward) /
                (sim_metrics.mean_reward + 1e-6)
            ),
            "tracking_error_increase": (
                real_metrics.tracking_error - sim_metrics.tracking_error
            )
        }

    def diagnostic_rollout(
        self,
        policy,
        record_video: bool = True
    ) -> Dict:
        """Detailed diagnostic rollout with logging."""
        obs_history = []
        action_history = []
        state_history = []

        obs = self.sim_env.reset()
        done = False

        while not done:
            obs_history.append(obs.copy())

            action = policy(obs)
            action_history.append(action.copy())

            obs, reward, done, info = self.sim_env.step(action)

            if 'full_state' in info:
                state_history.append(info['full_state'])

        return {
            "observations": np.array(obs_history),
            "actions": np.array(action_history),
            "states": np.array(state_history) if state_history else None,
        }
```

### Debugging Transfer Failures

```python
# debug_transfer.py
import numpy as np
import matplotlib.pyplot as plt

class TransferDebugger:
    """Debug tools for sim-to-real transfer failures."""

    def __init__(self, sim_rollout: Dict, real_rollout: Dict):
        self.sim = sim_rollout
        self.real = real_rollout

    def plot_trajectory_comparison(self, joint_indices: List[int] = None):
        """Plot sim vs real trajectories."""
        sim_pos = self.sim['observations']
        real_pos = self.real['observations']

        if joint_indices is None:
            joint_indices = list(range(min(6, sim_pos.shape[1])))

        fig, axes = plt.subplots(len(joint_indices), 1, figsize=(12, 3 * len(joint_indices)))

        for i, idx in enumerate(joint_indices):
            ax = axes[i] if len(joint_indices) > 1 else axes

            ax.plot(sim_pos[:, idx], label='Simulation', alpha=0.8)
            ax.plot(real_pos[:, idx], label='Real', alpha=0.8)
            ax.set_ylabel(f'Joint {idx}')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('trajectory_comparison.png')
        plt.close()

    def analyze_action_distribution(self):
        """Compare action distributions."""
        sim_actions = self.sim['actions']
        real_actions = self.real['actions']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histograms
        axes[0].hist(sim_actions.flatten(), bins=50, alpha=0.5, label='Sim')
        axes[0].hist(real_actions.flatten(), bins=50, alpha=0.5, label='Real')
        axes[0].set_title('Action Distribution')
        axes[0].legend()

        # Action range usage
        sim_range = np.max(sim_actions, axis=0) - np.min(sim_actions, axis=0)
        real_range = np.max(real_actions, axis=0) - np.min(real_actions, axis=0)

        x = np.arange(len(sim_range))
        axes[1].bar(x - 0.2, sim_range, 0.4, label='Sim')
        axes[1].bar(x + 0.2, real_range, 0.4, label='Real')
        axes[1].set_title('Action Range per Joint')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('action_analysis.png')
        plt.close()

    def identify_failure_modes(self) -> List[str]:
        """Identify common failure patterns."""
        failures = []

        sim_obs = self.sim['observations']
        real_obs = self.real['observations']

        # Check for observation divergence
        obs_error = np.abs(sim_obs - real_obs[:len(sim_obs)])
        if np.max(obs_error) > 1.0:
            failures.append("Large observation divergence detected")

        # Check for oscillations
        real_vel = np.diff(real_obs, axis=0)
        sign_changes = np.sum(np.diff(np.sign(real_vel), axis=0) != 0, axis=0)
        if np.max(sign_changes) > len(real_obs) * 0.5:
            failures.append("Oscillatory behavior detected")

        # Check for saturation
        sim_actions = self.sim['actions']
        action_range = 1.0  # Assuming normalized
        saturation_rate = np.mean(np.abs(sim_actions) > 0.95 * action_range)
        if saturation_rate > 0.2:
            failures.append(f"Action saturation: {saturation_rate:.1%} of actions")

        return failures
```

---

## Exercises

### Exercise 12.1: Quantify Reality Gap

**Objective**: Measure sim-to-real gap for a simple system.

**Difficulty**: Beginner | **Estimated Time**: 45 minutes

#### Instructions

1. Collect trajectory data from simulation
2. Collect corresponding real data (or synthetic "real" with noise)
3. Compute trajectory error metrics
4. Visualize the gap

---

### Exercise 12.2: Implement Domain Randomization

**Objective**: Add physics randomization to training.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Implement friction randomization
2. Add mass/inertia scaling
3. Include observation noise
4. Train with and without DR, compare

---

### Exercise 12.3: System Identification

**Objective**: Identify motor parameters from data.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Record step response data
2. Implement parameter estimation
3. Update simulation with identified parameters
4. Verify improved matching

---

### Exercise 12.4: Evaluate Transfer Quality

**Objective**: Comprehensively evaluate policy transfer.

**Difficulty**: Advanced | **Estimated Time**: 60 minutes

#### Instructions

1. Set up evaluation framework
2. Run diagnostic rollouts
3. Analyze failure modes
4. Propose improvements

---

## Summary

In this chapter, you learned:

- **Reality gap** arises from physics, sensors, and actuator differences
- **Domain randomization** creates robust policies
- **System identification** improves simulation accuracy
- **Online adaptation** enables real-world fine-tuning
- **Evaluation frameworks** help debug transfer failures

---

## References

[1] J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," in *IROS*, 2017.

[2] W. Yu et al., "Sim-to-Real Transfer for Biped Locomotion," in *IROS*, 2019.

[3] OpenAI et al., "Learning Dexterous In-Hand Manipulation," *Int. J. Robot. Res.*, 2020.

[4] X. B. Peng et al., "Sim-to-Real Robot Learning from Pixels with Progressive Nets," in *CoRL*, 2017.
