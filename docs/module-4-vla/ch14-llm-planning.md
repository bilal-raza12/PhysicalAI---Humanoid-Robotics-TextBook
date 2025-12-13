---
id: ch14-llm-planning
title: "Chapter 14: LLM-Powered Task Planning"
sidebar_position: 3
---

{/* This file uses raw code blocks to prevent MDX from parsing Python f-strings */}

# Chapter 14: LLM-Powered Task Planning

**Estimated Time**: 5-6 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Integrate** LLMs for high-level robot task planning
2. **Design** prompts for effective robot control
3. **Implement** plan execution and monitoring
4. **Handle** failures and re-planning scenarios
5. **Ground** language commands in robot capabilities

---

## 14.1 LLMs as Robot Planners

Large Language Models can decompose complex tasks into executable robot actions.

### Planning Architecture

```
┌─────────────────────────────────────────────────────────┐
│                LLM Task Planning Pipeline                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   User Command                                           │
│       ↓                                                  │
│   ┌─────────────────┐                                   │
│   │  LLM Planner    │ ← Context (skills, state)         │
│   │  (GPT-4, etc.)  │                                   │
│   └────────┬────────┘                                   │
│            ↓                                             │
│   [Step 1] → [Step 2] → [Step 3] → ...                  │
│            ↓                                             │
│   ┌─────────────────┐                                   │
│   │ Skill Executor  │ ← Feedback                        │
│   └────────┬────────┘                                   │
│            ↓                                             │
│   Robot Actions                                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Basic LLM Planner

```python
# llm_planner.py
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

@dataclass
class PlanStep:
    """Single step in a task plan."""
    action: str
    parameters: Dict
    preconditions: List[str]
    expected_effects: List[str]

@dataclass
class TaskPlan:
    """Complete task plan."""
    goal: str
    steps: List[PlanStep]
    estimated_duration: float

class LLMPlanner:
    """Use LLM for task planning."""

    def __init__(self, llm_client, available_skills: List[str]):
        self.llm = llm_client
        self.skills = available_skills
        self.context_history = []

    def create_plan(
        self,
        user_command: str,
        current_state: Dict,
        max_steps: int = 10
    ) -> TaskPlan:
        """
        Generate a task plan from natural language command.

        Args:
            user_command: Natural language task description
            current_state: Current robot and environment state
            max_steps: Maximum number of plan steps

        Returns:
            TaskPlan with executable steps
        """
        prompt = self._build_planning_prompt(
            user_command, current_state, max_steps
        )

        response = self.llm.generate(prompt)
        plan = self._parse_plan_response(response, user_command)

        return plan

    def _build_planning_prompt(
        self,
        command: str,
        state: Dict,
        max_steps: int
    ) -> str:
        """Construct the planning prompt."""
        skills_description = self._format_skills()
        state_description = self._format_state(state)

        prompt = f"""You are a robot task planner. Given a command, decompose it into
executable steps using the available robot skills.

## Available Skills
{skills_description}

## Current State
{state_description}

## Command
{command}

## Instructions
1. Analyze the command and current state
2. Break down the task into {max_steps} or fewer steps
3. Each step must use an available skill
4. Consider preconditions and order dependencies
5. Output in JSON format

## Output Format
```json
{{
    "goal": "brief goal description",
    "steps": [
        {{
            "action": "skill_name",
            "parameters": {{"param1": "value1"}},
            "preconditions": ["condition1"],
            "expected_effects": ["effect1"]
        }}
    ],
    "estimated_duration": 30.0
}}
```

Generate the plan:"""

        return prompt

    def _format_skills(self) -> str:
        """Format available skills for the prompt."""
        return "\n".join(["- " + skill for skill in self.skills])

    def _format_state(self, state: Dict) -> str:
        """Format current state for the prompt."""
        return json.dumps(state, indent=2)

    def _parse_plan_response(self, response: str, goal: str) -> TaskPlan:
        """Parse LLM response into TaskPlan."""
        # Extract JSON from response
        try:
            # Find JSON block
            start = response.find("```json")
            end = response.find("```", start + 7)
            if start != -1 and end != -1:
                json_str = response[start + 7:end].strip()
            else:
                json_str = response

            plan_dict = json.loads(json_str)

            steps = [
                PlanStep(
                    action=step["action"],
                    parameters=step.get("parameters", {}),
                    preconditions=step.get("preconditions", []),
                    expected_effects=step.get("expected_effects", [])
                )
                for step in plan_dict["steps"]
            ]

            return TaskPlan(
                goal=plan_dict.get("goal", goal),
                steps=steps,
                estimated_duration=plan_dict.get("estimated_duration", 0.0)
            )

        except json.JSONDecodeError as e:
            print(f"Failed to parse plan: {e}")
            return TaskPlan(goal=goal, steps=[], estimated_duration=0.0)


# Example skill definitions
HUMANOID_SKILLS = [
    "navigate_to(location: str) - Move to a named location",
    "pick_up(object: str) - Grasp and pick up an object",
    "place_on(surface: str) - Place held object on surface",
    "open_door(door: str) - Open a door",
    "close_door(door: str) - Close a door",
    "say(message: str) - Speak a message",
    "wave() - Wave greeting gesture",
    "point_at(target: str) - Point at a target",
    "wait(seconds: float) - Wait for specified time",
    "look_at(target: str) - Turn head to look at target",
]
```

---

## 14.2 Prompt Engineering for Robotics

Effective prompts are crucial for reliable robot planning.

### Prompt Design Patterns

```python
# prompt_patterns.py
from typing import Dict, List

class RobotPromptTemplates:
    """Prompt templates for robot planning."""

    @staticmethod
    def chain_of_thought_planning(
        command: str,
        skills: List[str],
        state: Dict
    ) -> str:
        """Chain-of-thought prompting for complex planning."""
        return f"""Let's plan this robot task step by step.

Command: {command}

Available skills: {', '.join(skills)}

Current state:
- Robot location: {state.get('robot_location', 'unknown')}
- Held object: {state.get('held_object', 'none')}
- Visible objects: {state.get('visible_objects', [])}

Think through this carefully:

1. What is the goal of this command?
2. What are the preconditions that must be met?
3. What sequence of skills achieves the goal?
4. What could go wrong, and how to handle it?

Now provide the plan:"""

    @staticmethod
    def few_shot_planning(
        command: str,
        examples: List[Dict]
    ) -> str:
        """Few-shot prompting with examples."""
        examples_str = ""
        for ex in examples:
            examples_str += f"""
Example:
Command: {ex['command']}
Plan:
{ex['plan']}
---"""

        return f"""You are a robot planner. Given a command, output a plan.

{examples_str}

Now plan for:
Command: {command}
Plan:"""

    @staticmethod
    def constrained_planning(
        command: str,
        constraints: List[str]
    ) -> str:
        """Planning with explicit constraints."""
        constraints_str = "\n".join([f"- {c}" for c in constraints])

        return f"""Plan a robot task with the following constraints:

Command: {command}

Constraints (MUST be satisfied):
{constraints_str}

Generate a plan that satisfies ALL constraints:"""

    @staticmethod
    def error_recovery_prompt(
        original_plan: str,
        failed_step: str,
        error_message: str,
        current_state: Dict
    ) -> str:
        """Prompt for recovering from execution errors."""
        return f"""The robot encountered an error during plan execution.

Original plan:
{original_plan}

Failed at step: {failed_step}
Error: {error_message}

Current state after failure:
{current_state}

Please provide a recovery plan to either:
1. Retry the failed step with modifications
2. Find an alternative approach
3. Safely abort and explain why

Recovery plan:"""


class PromptOptimizer:
    """Optimize prompts based on execution feedback."""

    def __init__(self):
        self.prompt_history = []
        self.success_rates = {}

    def record_outcome(
        self,
        prompt_template: str,
        success: bool,
        execution_time: float
    ):
        """Record prompt execution outcome."""
        self.prompt_history.append({
            "template": prompt_template,
            "success": success,
            "time": execution_time
        })

        if prompt_template not in self.success_rates:
            self.success_rates[prompt_template] = {"success": 0, "total": 0}

        self.success_rates[prompt_template]["total"] += 1
        if success:
            self.success_rates[prompt_template]["success"] += 1

    def get_best_template(self) -> str:
        """Return the most successful prompt template."""
        best_rate = 0
        best_template = None

        for template, stats in self.success_rates.items():
            if stats["total"] >= 5:  # Minimum trials
                rate = stats["success"] / stats["total"]
                if rate > best_rate:
                    best_rate = rate
                    best_template = template

        return best_template
```

---

## 14.3 Skill Grounding

Ground language commands in actual robot capabilities.

### Skill Library

```python
# skill_library.py
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class SkillStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"
    PREEMPTED = "preempted"

@dataclass
class SkillResult:
    """Result of skill execution."""
    status: SkillStatus
    message: str
    data: Optional[Dict] = None

class Skill:
    """Base class for robot skills."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters = {}

    def check_preconditions(self, state: Dict) -> bool:
        """Check if skill can be executed."""
        raise NotImplementedError

    def execute(self, robot, **kwargs) -> SkillResult:
        """Execute the skill."""
        raise NotImplementedError

    def get_expected_effects(self, **kwargs) -> List[str]:
        """Return expected state changes."""
        raise NotImplementedError


class NavigateSkill(Skill):
    """Navigate to a location."""

    def __init__(self):
        super().__init__(
            name="navigate_to",
            description="Move robot to a named location"
        )
        self.parameters = {
            "location": "Target location name (str)"
        }

    def check_preconditions(self, state: Dict) -> bool:
        # Robot must not be holding fragile objects while walking
        held = state.get("held_object")
        if held and state.get("object_properties", {}).get(held, {}).get("fragile"):
            return False
        return True

    def execute(self, robot, location: str) -> SkillResult:
        """Execute navigation."""
        try:
            # Get location coordinates
            coords = robot.get_location_coords(location)
            if coords is None:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    message=f"Unknown location: {location}"
                )

            # Execute navigation
            success = robot.navigate(coords)

            if success:
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    message=f"Arrived at {location}",
                    data={"final_position": coords}
                )
            else:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    message="Navigation failed"
                )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                message=str(e)
            )

    def get_expected_effects(self, location: str) -> List[str]:
        return [f"robot_at({location})"]


class PickUpSkill(Skill):
    """Pick up an object."""

    def __init__(self):
        super().__init__(
            name="pick_up",
            description="Grasp and pick up an object"
        )
        self.parameters = {
            "object": "Target object name (str)"
        }

    def check_preconditions(self, state: Dict) -> bool:
        # Hand must be empty
        if state.get("held_object") is not None:
            return False
        return True

    def execute(self, robot, object: str) -> SkillResult:
        """Execute pick up."""
        try:
            # Detect object
            detection = robot.detect_object(object)
            if detection is None:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    message=f"Cannot find object: {object}"
                )

            # Plan grasp
            grasp_pose = robot.plan_grasp(detection)
            if grasp_pose is None:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    message="No valid grasp found"
                )

            # Execute grasp
            success = robot.execute_grasp(grasp_pose)

            if success:
                return SkillResult(
                    status=SkillStatus.SUCCESS,
                    message=f"Picked up {object}",
                    data={"grasped_object": object}
                )
            else:
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    message="Grasp execution failed"
                )

        except Exception as e:
            return SkillResult(
                status=SkillStatus.FAILURE,
                message=str(e)
            )

    def get_expected_effects(self, object: str) -> List[str]:
        return [f"holding({object})", f"not(on_surface({object}))"]


class SkillLibrary:
    """Registry of available robot skills."""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}

    def register(self, skill: Skill):
        """Register a skill."""
        self.skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        """Get skill by name."""
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """List all available skills."""
        return list(self.skills.keys())

    def get_skill_descriptions(self) -> str:
        """Get formatted skill descriptions for LLM."""
        descriptions = []
        for name, skill in self.skills.items():
            params = ", ".join([
                f"{k}: {v}" for k, v in skill.parameters.items()
            ])
            descriptions.append(f"- {name}({params}): {skill.description}")
        return "\n".join(descriptions)


# Create default skill library
def create_humanoid_skill_library() -> SkillLibrary:
    """Create skill library for humanoid robot."""
    library = SkillLibrary()

    library.register(NavigateSkill())
    library.register(PickUpSkill())
    # Add more skills...

    return library
```

---

## 14.4 Plan Execution and Monitoring

### Execution Engine

```python
# plan_executor.py
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    REPLANNING = "replanning"

@dataclass
class ExecutionState:
    """Current execution state."""
    plan: TaskPlan
    current_step: int
    status: ExecutionStatus
    step_results: List[SkillResult]
    start_time: float
    end_time: Optional[float] = None

class PlanExecutor:
    """Execute task plans with monitoring."""

    def __init__(
        self,
        robot,
        skill_library: SkillLibrary,
        planner: LLMPlanner,
        state_monitor: Callable
    ):
        self.robot = robot
        self.skills = skill_library
        self.planner = planner
        self.get_state = state_monitor

        self.execution_state: Optional[ExecutionState] = None
        self.max_retries = 3
        self.max_replans = 2

    def execute(
        self,
        plan: TaskPlan,
        on_step_complete: Optional[Callable] = None
    ) -> ExecutionState:
        """
        Execute a task plan.

        Args:
            plan: TaskPlan to execute
            on_step_complete: Callback after each step

        Returns:
            Final ExecutionState
        """
        self.execution_state = ExecutionState(
            plan=plan,
            current_step=0,
            status=ExecutionStatus.RUNNING,
            step_results=[],
            start_time=time.time()
        )

        replan_count = 0

        while self.execution_state.current_step < len(plan.steps):
            step = plan.steps[self.execution_state.current_step]

            # Check preconditions
            current_state = self.get_state()
            skill = self.skills.get(step.action)

            if skill is None:
                self._handle_error(f"Unknown skill: {step.action}")
                break

            if not skill.check_preconditions(current_state):
                # Try replanning
                if replan_count < self.max_replans:
                    new_plan = self._replan(
                        plan.goal,
                        current_state,
                        f"Preconditions not met for {step.action}"
                    )
                    if new_plan and new_plan.steps:
                        plan = new_plan
                        self.execution_state.plan = plan
                        self.execution_state.current_step = 0
                        replan_count += 1
                        continue

                self._handle_error(
                    f"Preconditions not met for {step.action}"
                )
                break

            # Execute step
            result = self._execute_step(skill, step)
            self.execution_state.step_results.append(result)

            if result.status == SkillStatus.SUCCESS:
                self.execution_state.current_step += 1

                if on_step_complete:
                    on_step_complete(step, result)

            else:
                # Handle failure
                if not self._handle_step_failure(step, result, replan_count):
                    break
                replan_count += 1

        # Finalize
        self.execution_state.end_time = time.time()

        if self.execution_state.current_step >= len(plan.steps):
            self.execution_state.status = ExecutionStatus.SUCCESS
        else:
            self.execution_state.status = ExecutionStatus.FAILURE

        return self.execution_state

    def _execute_step(self, skill: Skill, step: PlanStep) -> SkillResult:
        """Execute a single plan step with retries."""
        for attempt in range(self.max_retries):
            result = skill.execute(self.robot, **step.parameters)

            if result.status == SkillStatus.SUCCESS:
                return result

            print(f"Step failed (attempt {attempt + 1}): {result.message}")
            time.sleep(0.5)

        return result

    def _handle_step_failure(
        self,
        step: PlanStep,
        result: SkillResult,
        replan_count: int
    ) -> bool:
        """Handle step failure, return True if recovered."""
        if replan_count >= self.max_replans:
            self._handle_error(f"Max replans reached: {result.message}")
            return False

        # Attempt replanning
        current_state = self.get_state()
        new_plan = self._replan(
            self.execution_state.plan.goal,
            current_state,
            f"Step '{step.action}' failed: {result.message}"
        )

        if new_plan and new_plan.steps:
            self.execution_state.plan = new_plan
            self.execution_state.current_step = 0
            self.execution_state.status = ExecutionStatus.REPLANNING
            return True

        return False

    def _replan(
        self,
        goal: str,
        current_state: Dict,
        failure_reason: str
    ) -> Optional[TaskPlan]:
        """Request new plan from LLM."""
        print(f"Replanning due to: {failure_reason}")

        # Add failure context to state
        current_state["failure_reason"] = failure_reason
        current_state["attempted_steps"] = [
            r.message for r in self.execution_state.step_results
        ]

        try:
            new_plan = self.planner.create_plan(goal, current_state)
            return new_plan
        except Exception as e:
            print(f"Replanning failed: {e}")
            return None

    def _handle_error(self, message: str):
        """Handle execution error."""
        print(f"Execution error: {message}")
        self.execution_state.status = ExecutionStatus.FAILURE


class ExecutionMonitor:
    """Monitor plan execution and environment."""

    def __init__(self, robot, expected_duration: float):
        self.robot = robot
        self.expected_duration = expected_duration
        self.start_time = None
        self.anomalies = []

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()

    def check(self) -> List[str]:
        """Check for anomalies."""
        warnings = []

        # Check timeout
        elapsed = time.time() - self.start_time
        if elapsed > self.expected_duration * 1.5:
            warnings.append(f"Execution taking longer than expected: {elapsed:.1f}s")

        # Check robot health
        if not self.robot.is_healthy():
            warnings.append("Robot health check failed")

        # Check for obstacles
        if self.robot.emergency_stop_triggered():
            warnings.append("Emergency stop triggered")

        self.anomalies.extend(warnings)
        return warnings
```

---

## 14.5 Error Handling and Recovery

### Recovery Strategies

```python
# error_recovery.py
from typing import Dict, List, Optional
from enum import Enum

class RecoveryStrategy(Enum):
    RETRY = "retry"
    REPLAN = "replan"
    FALLBACK = "fallback"
    ABORT = "abort"
    HUMAN_HELP = "human_help"

class ErrorClassifier:
    """Classify errors and suggest recovery strategies."""

    def __init__(self):
        self.error_patterns = {
            "object_not_found": {
                "patterns": ["cannot find", "not detected", "not visible"],
                "strategy": RecoveryStrategy.REPLAN,
                "suggestion": "Search for object in different locations"
            },
            "grasp_failed": {
                "patterns": ["grasp failed", "cannot grasp", "slip"],
                "strategy": RecoveryStrategy.RETRY,
                "suggestion": "Try different grasp pose"
            },
            "navigation_blocked": {
                "patterns": ["path blocked", "collision", "cannot reach"],
                "strategy": RecoveryStrategy.REPLAN,
                "suggestion": "Find alternative path"
            },
            "hardware_error": {
                "patterns": ["motor", "sensor", "communication"],
                "strategy": RecoveryStrategy.ABORT,
                "suggestion": "Hardware issue requires maintenance"
            },
            "timeout": {
                "patterns": ["timeout", "too long", "deadline"],
                "strategy": RecoveryStrategy.FALLBACK,
                "suggestion": "Simplify task or skip step"
            }
        }

    def classify(self, error_message: str) -> Dict:
        """Classify error and return recovery info."""
        error_lower = error_message.lower()

        for error_type, info in self.error_patterns.items():
            for pattern in info["patterns"]:
                if pattern in error_lower:
                    return {
                        "type": error_type,
                        "strategy": info["strategy"],
                        "suggestion": info["suggestion"]
                    }

        # Default
        return {
            "type": "unknown",
            "strategy": RecoveryStrategy.HUMAN_HELP,
            "suggestion": "Error not recognized, requesting human assistance"
        }


class RecoveryPlanner:
    """Plan recovery from execution failures."""

    def __init__(self, llm_planner: LLMPlanner):
        self.planner = llm_planner
        self.classifier = ErrorClassifier()

    def plan_recovery(
        self,
        failed_step: PlanStep,
        error_message: str,
        current_state: Dict,
        original_goal: str
    ) -> Optional[TaskPlan]:
        """
        Plan recovery from a failure.

        Returns:
            Recovery plan or None if cannot recover
        """
        # Classify error
        error_info = self.classifier.classify(error_message)
        strategy = error_info["strategy"]

        if strategy == RecoveryStrategy.ABORT:
            return None

        if strategy == RecoveryStrategy.HUMAN_HELP:
            self._request_human_help(error_message)
            return None

        if strategy == RecoveryStrategy.RETRY:
            # Create single-step retry plan
            return TaskPlan(
                goal=f"Retry: {failed_step.action}",
                steps=[failed_step],
                estimated_duration=10.0
            )

        if strategy == RecoveryStrategy.REPLAN:
            # Use LLM to create alternative plan
            recovery_prompt = f"""
The robot was trying to: {original_goal}
It failed at step: {failed_step.action}({failed_step.parameters})
Error: {error_message}
Suggestion: {error_info['suggestion']}

Current state: {current_state}

Create an alternative plan to achieve the original goal,
avoiding the approach that failed.
"""
            return self.planner.create_plan(
                recovery_prompt,
                current_state
            )

        if strategy == RecoveryStrategy.FALLBACK:
            # Create simplified fallback plan
            return self._create_fallback_plan(original_goal, current_state)

        return None

    def _request_human_help(self, error_message: str):
        """Request human intervention."""
        print(f"\n[HUMAN HELP NEEDED]\n{error_message}")
        # Could integrate with UI or notification system

    def _create_fallback_plan(
        self,
        goal: str,
        state: Dict
    ) -> Optional[TaskPlan]:
        """Create simplified fallback plan."""
        # LLM creates simpler version
        fallback_prompt = f"""
Create a SIMPLIFIED version of this goal that is more likely to succeed:
Goal: {goal}

The simplified plan should:
1. Skip non-essential steps
2. Use more robust/reliable skills
3. Accept partial success
"""
        return self.planner.create_plan(fallback_prompt, state)
```

---

## Exercises

### Exercise 14.1: Build LLM Planner

**Objective**: Create a basic LLM-based task planner.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Set up OpenAI or local LLM client
2. Implement planning prompt template
3. Parse LLM response to task plan
4. Test with sample commands

---

### Exercise 14.2: Design Skill Library

**Objective**: Create a library of robot skills.

**Difficulty**: Intermediate | **Estimated Time**: 45 minutes

#### Instructions

1. Define 5+ robot skills
2. Implement precondition checking
3. Add expected effects
4. Test skill execution

---

### Exercise 14.3: Implement Plan Execution

**Objective**: Build plan executor with monitoring.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Create execution engine
2. Add step-by-step execution
3. Implement progress monitoring
4. Handle timeouts

---

### Exercise 14.4: Error Recovery System

**Objective**: Add error handling and recovery.

**Difficulty**: Advanced | **Estimated Time**: 60 minutes

#### Instructions

1. Implement error classifier
2. Create recovery strategies
3. Add replanning capability
4. Test failure scenarios

---

## Summary

In this chapter, you learned:

- **LLM planners** decompose complex tasks into executable steps
- **Prompt engineering** is critical for reliable planning
- **Skill grounding** connects language to robot capabilities
- **Plan execution** requires monitoring and error handling
- **Recovery strategies** enable robust task completion

---

## References

[1] M. Ahn et al., "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances," in *CoRL*, 2022.

[2] W. Huang et al., "Language Models as Zero-Shot Planners," in *ICML*, 2022.

[3] B. Liu et al., "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency," *arXiv*, 2023.

[4] Y. Ding et al., "Task and Motion Planning with Large Language Models for Object Rearrangement," in *IROS*, 2023.
