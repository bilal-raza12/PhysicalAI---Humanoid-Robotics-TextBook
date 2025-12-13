---
id: ch16-embodied-agents
title: "Chapter 16: Building Embodied AI Agents"
sidebar_position: 5
---

# Chapter 16: Building Embodied AI Agents

**Estimated Time**: 5-6 hours | **Exercises**: 4

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Design** end-to-end embodied AI agents
2. **Integrate** perception, planning, and control
3. **Implement** memory and learning from experience
4. **Handle** human interaction naturally
5. **Deploy** agents in real-world scenarios

---

## 16.1 Embodied Agent Architecture

An embodied AI agent combines perception, reasoning, and action in a unified system.

### Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Embodied AI Agent Architecture              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│                    ┌───────────┐                        │
│                    │   Human   │                        │
│                    │Interaction│                        │
│                    └─────┬─────┘                        │
│                          ↓                              │
│   ┌────────────────────────────────────────────────┐   │
│   │              Cognitive Layer                     │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐        │   │
│   │  │ Memory  │  │ Planner │  │Reasoner │        │   │
│   │  └────┬────┘  └────┬────┘  └────┬────┘        │   │
│   │       └────────────┼────────────┘              │   │
│   └────────────────────┼────────────────────────────┘   │
│                        ↓                                │
│   ┌────────────────────────────────────────────────┐   │
│   │             Perception Layer                     │   │
│   │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  │   │
│   │  │Vision │  │ Audio │  │Tactile│  │Proprio│  │   │
│   │  └───────┘  └───────┘  └───────┘  └───────┘  │   │
│   └────────────────────┼────────────────────────────┘   │
│                        ↓                                │
│   ┌────────────────────────────────────────────────┐   │
│   │              Action Layer                        │   │
│   │  ┌─────────┐  ┌──────────┐  ┌─────────┐       │   │
│   │  │ Motion  │  │Manipulate│  │ Speech  │       │   │
│   │  └─────────┘  └──────────┘  └─────────┘       │   │
│   └────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Core Agent Class

```python
# embodied_agent.py
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time

class AgentState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    EXECUTING = "executing"
    INTERACTING = "interacting"
    ERROR = "error"

@dataclass
class AgentConfig:
    """Configuration for embodied agent."""
    name: str = "HumanoidAgent"
    perception_rate: float = 30.0  # Hz
    planning_rate: float = 10.0  # Hz
    control_rate: float = 100.0  # Hz
    enable_speech: bool = True
    enable_memory: bool = True
    max_plan_steps: int = 20

@dataclass
class AgentContext:
    """Current context for agent decisions."""
    current_task: Optional[str] = None
    environment_state: Dict = field(default_factory=dict)
    human_present: bool = False
    last_human_input: Optional[str] = None
    confidence: float = 1.0

class EmbodiedAgent:
    """Main embodied AI agent class."""

    def __init__(
        self,
        config: AgentConfig,
        robot_interface,
        perception_system,
        planner,
        memory_system=None
    ):
        self.config = config
        self.robot = robot_interface
        self.perception = perception_system
        self.planner = planner
        self.memory = memory_system

        self.state = AgentState.IDLE
        self.context = AgentContext()
        self.current_plan = None

        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        # Running flag
        self.running = False

    async def start(self):
        """Start the agent."""
        self.running = True
        self._set_state(AgentState.IDLE)

        # Start concurrent loops
        await asyncio.gather(
            self._perception_loop(),
            self._planning_loop(),
            self._execution_loop()
        )

    async def stop(self):
        """Stop the agent."""
        self.running = False
        self.robot.stop()

    def give_task(self, task: str):
        """Assign a task to the agent."""
        self.context.current_task = task
        self.context.confidence = 1.0
        self._set_state(AgentState.PLANNING)

    def handle_human_input(self, input_text: str):
        """Handle input from human."""
        self.context.last_human_input = input_text
        self.context.human_present = True

        # Process input
        if self._is_new_task(input_text):
            self.give_task(input_text)
        elif self._is_clarification(input_text):
            self._incorporate_clarification(input_text)
        elif self._is_stop_command(input_text):
            self._handle_stop()

    async def _perception_loop(self):
        """Continuous perception processing."""
        period = 1.0 / self.config.perception_rate

        while self.running:
            start = time.time()

            # Get sensor data
            sensor_data = self.robot.get_sensor_data()

            # Process perception
            perception_output = self.perception.process(sensor_data)

            # Update context
            self.context.environment_state = {
                "objects": perception_output.detected_objects,
                "scene": perception_output.scene_description,
                "contacts": perception_output.contact_info
            }

            # Check for human speech
            if perception_output.speech_text:
                self.handle_human_input(perception_output.speech_text)

            # Sleep for remaining time
            elapsed = time.time() - start
            if elapsed < period:
                await asyncio.sleep(period - elapsed)

    async def _planning_loop(self):
        """Continuous planning and replanning."""
        period = 1.0 / self.config.planning_rate

        while self.running:
            start = time.time()

            if self.state == AgentState.PLANNING and self.context.current_task:
                # Generate plan
                plan = await self._generate_plan()

                if plan:
                    self.current_plan = plan
                    self._set_state(AgentState.EXECUTING)
                else:
                    self._handle_planning_failure()

            elif self.state == AgentState.EXECUTING:
                # Check if replanning needed
                if self._should_replan():
                    self._set_state(AgentState.PLANNING)

            elapsed = time.time() - start
            if elapsed < period:
                await asyncio.sleep(period - elapsed)

    async def _execution_loop(self):
        """Execute current plan."""
        period = 1.0 / self.config.control_rate

        while self.running:
            start = time.time()

            if self.state == AgentState.EXECUTING and self.current_plan:
                # Execute next action
                success = await self._execute_step()

                if not success:
                    self._handle_execution_failure()

                # Check completion
                if self._plan_complete():
                    self._handle_task_complete()

            elapsed = time.time() - start
            if elapsed < period:
                await asyncio.sleep(period - elapsed)

    async def _generate_plan(self):
        """Generate plan for current task."""
        plan = self.planner.create_plan(
            self.context.current_task,
            self.context.environment_state,
            max_steps=self.config.max_plan_steps
        )

        # Store in memory
        if self.memory:
            self.memory.store_plan(
                task=self.context.current_task,
                plan=plan,
                context=self.context.environment_state
            )

        return plan

    async def _execute_step(self) -> bool:
        """Execute one step of the plan."""
        if not self.current_plan or not self.current_plan.steps:
            return False

        step = self.current_plan.steps[0]

        # Get skill
        skill = self.planner.skills.get(step.action)
        if not skill:
            return False

        # Execute
        result = skill.execute(self.robot, **step.parameters)

        if result.status == SkillStatus.SUCCESS:
            self.current_plan.steps.pop(0)

            # Store experience
            if self.memory:
                self.memory.store_experience(
                    action=step.action,
                    parameters=step.parameters,
                    result=result,
                    state=self.context.environment_state
                )

            return True

        return False

    def _should_replan(self) -> bool:
        """Check if replanning is needed."""
        # Check for significant environment changes
        # Check for human intervention
        # Check for execution failures
        return False

    def _plan_complete(self) -> bool:
        """Check if plan is complete."""
        return self.current_plan is None or len(self.current_plan.steps) == 0

    def _handle_task_complete(self):
        """Handle task completion."""
        task = self.context.current_task
        self.context.current_task = None
        self.current_plan = None
        self._set_state(AgentState.IDLE)

        if self.on_task_complete:
            self.on_task_complete(task)

        # Speak confirmation
        if self.config.enable_speech:
            self.robot.speak(f"Task completed: {task}")

    def _handle_planning_failure(self):
        """Handle failure to generate plan."""
        self._set_state(AgentState.ERROR)

        if self.config.enable_speech:
            self.robot.speak("I couldn't figure out how to do that. Can you help?")

    def _handle_execution_failure(self):
        """Handle execution failure."""
        # Try replanning first
        self._set_state(AgentState.PLANNING)

    def _set_state(self, new_state: AgentState):
        """Update agent state."""
        old_state = self.state
        self.state = new_state

        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def _is_new_task(self, text: str) -> bool:
        """Check if input is a new task."""
        task_keywords = ["please", "can you", "could you", "go", "pick", "bring", "find"]
        return any(kw in text.lower() for kw in task_keywords)

    def _is_clarification(self, text: str) -> bool:
        """Check if input is a clarification."""
        return self.state == AgentState.INTERACTING

    def _is_stop_command(self, text: str) -> bool:
        """Check if input is a stop command."""
        stop_keywords = ["stop", "halt", "cancel", "abort"]
        return any(kw in text.lower() for kw in stop_keywords)

    def _handle_stop(self):
        """Handle stop command."""
        self.robot.stop()
        self.current_plan = None
        self.context.current_task = None
        self._set_state(AgentState.IDLE)

    def _incorporate_clarification(self, text: str):
        """Incorporate clarification into current context."""
        # Update context with clarification
        pass
```

---

## 16.2 Memory Systems

### Episodic and Semantic Memory

```python
# memory_system.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import faiss  # For similarity search

@dataclass
class Episode:
    """Single episode in memory."""
    id: str
    timestamp: datetime
    task: str
    actions: List[Dict]
    outcome: str
    state_before: Dict
    state_after: Dict
    embedding: Optional[np.ndarray] = None

@dataclass
class SemanticFact:
    """Semantic knowledge fact."""
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str  # Where this fact came from
    timestamp: datetime

class EpisodicMemory:
    """Store and retrieve episodic experiences."""

    def __init__(self, embedding_model, max_episodes: int = 10000):
        self.embedding_model = embedding_model
        self.max_episodes = max_episodes

        self.episodes: List[Episode] = []
        self.index = None  # FAISS index

    def store(self, episode: Episode):
        """Store a new episode."""
        # Generate embedding
        text_repr = self._episode_to_text(episode)
        episode.embedding = self.embedding_model.encode(text_repr)

        self.episodes.append(episode)

        # Rebuild index
        self._rebuild_index()

        # Remove old episodes if needed
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
            self._rebuild_index()

    def retrieve_similar(
        self,
        query: str,
        k: int = 5
    ) -> List[Episode]:
        """Retrieve similar episodes."""
        if self.index is None or len(self.episodes) == 0:
            return []

        query_embedding = self.embedding_model.encode(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.episodes):
                results.append(self.episodes[idx])

        return results

    def retrieve_by_task(self, task: str) -> List[Episode]:
        """Retrieve all episodes for a task type."""
        return [e for e in self.episodes if task.lower() in e.task.lower()]

    def _episode_to_text(self, episode: Episode) -> str:
        """Convert episode to text for embedding."""
        actions_str = ", ".join([a["action"] for a in episode.actions])
        return f"Task: {episode.task}. Actions: {actions_str}. Outcome: {episode.outcome}"

    def _rebuild_index(self):
        """Rebuild FAISS index."""
        if len(self.episodes) == 0:
            self.index = None
            return

        embeddings = np.array([e.embedding for e in self.episodes]).astype('float32')
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)


class SemanticMemory:
    """Store and query semantic knowledge."""

    def __init__(self):
        self.facts: List[SemanticFact] = []
        self.knowledge_graph: Dict[str, Dict[str, List[str]]] = {}

    def add_fact(self, fact: SemanticFact):
        """Add a semantic fact."""
        self.facts.append(fact)

        # Update knowledge graph
        if fact.subject not in self.knowledge_graph:
            self.knowledge_graph[fact.subject] = {}

        if fact.predicate not in self.knowledge_graph[fact.subject]:
            self.knowledge_graph[fact.subject][fact.predicate] = []

        self.knowledge_graph[fact.subject][fact.predicate].append(fact.object)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None
    ) -> List[SemanticFact]:
        """Query facts matching criteria."""
        results = []

        for fact in self.facts:
            match = True
            if subject and fact.subject != subject:
                match = False
            if predicate and fact.predicate != predicate:
                match = False
            if object and fact.object != object:
                match = False

            if match:
                results.append(fact)

        return results

    def get_properties(self, subject: str) -> Dict[str, List[str]]:
        """Get all properties of a subject."""
        return self.knowledge_graph.get(subject, {})


class WorkingMemory:
    """Short-term working memory for current context."""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[Dict] = []
        self.attention_weights: List[float] = []

    def add(self, item: Dict, importance: float = 1.0):
        """Add item to working memory."""
        self.items.append(item)
        self.attention_weights.append(importance)

        # Remove least important if over capacity
        while len(self.items) > self.capacity:
            min_idx = np.argmin(self.attention_weights)
            self.items.pop(min_idx)
            self.attention_weights.pop(min_idx)

    def get_context(self) -> List[Dict]:
        """Get current context items sorted by importance."""
        sorted_indices = np.argsort(self.attention_weights)[::-1]
        return [self.items[i] for i in sorted_indices]

    def update_attention(self, item_idx: int, new_weight: float):
        """Update attention weight for an item."""
        if 0 <= item_idx < len(self.items):
            self.attention_weights[item_idx] = new_weight

    def clear(self):
        """Clear working memory."""
        self.items = []
        self.attention_weights = []


class MemorySystem:
    """Unified memory system combining episodic, semantic, and working memory."""

    def __init__(self, embedding_model):
        self.episodic = EpisodicMemory(embedding_model)
        self.semantic = SemanticMemory()
        self.working = WorkingMemory()

    def store_experience(
        self,
        action: str,
        parameters: Dict,
        result: Any,
        state: Dict
    ):
        """Store experience from action execution."""
        # Add to working memory
        self.working.add({
            "type": "action",
            "action": action,
            "result": str(result)
        })

        # Extract semantic facts
        self._extract_facts(action, parameters, result, state)

    def store_plan(
        self,
        task: str,
        plan: Any,
        context: Dict
    ):
        """Store a generated plan."""
        episode = Episode(
            id=f"plan_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            task=task,
            actions=[{"action": s.action, "params": s.parameters} for s in plan.steps],
            outcome="planned",
            state_before=context,
            state_after={}
        )
        self.episodic.store(episode)

    def get_relevant_experiences(self, task: str, k: int = 3) -> List[Episode]:
        """Get experiences relevant to a task."""
        return self.episodic.retrieve_similar(task, k)

    def get_object_knowledge(self, object_name: str) -> Dict:
        """Get knowledge about an object."""
        return self.semantic.get_properties(object_name)

    def get_current_context(self) -> List[Dict]:
        """Get current working memory context."""
        return self.working.get_context()

    def _extract_facts(
        self,
        action: str,
        parameters: Dict,
        result: Any,
        state: Dict
    ):
        """Extract semantic facts from experience."""
        # Example: learn object locations
        if action == "pick_up" and result.status == "success":
            obj = parameters.get("object")
            location = state.get("robot_location")
            if obj and location:
                self.semantic.add_fact(SemanticFact(
                    subject=obj,
                    predicate="found_at",
                    object=location,
                    confidence=0.9,
                    source="experience",
                    timestamp=datetime.now()
                ))
```

---

## 16.3 Human Interaction

### Natural Language Interface

```python
# human_interaction.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
from enum import Enum

class InteractionMode(Enum):
    COMMAND = "command"
    CONVERSATION = "conversation"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"

@dataclass
class DialogueTurn:
    """Single turn in dialogue."""
    speaker: str  # "human" or "robot"
    text: str
    intent: Optional[str] = None
    entities: Optional[Dict] = None

class DialogueManager:
    """Manage human-robot dialogue."""

    def __init__(self, llm_client, robot_capabilities: List[str]):
        self.llm = llm_client
        self.capabilities = robot_capabilities

        self.history: List[DialogueTurn] = []
        self.mode = InteractionMode.COMMAND
        self.pending_clarification = None

    def process_input(self, text: str) -> str:
        """Process human input and generate response."""
        # Add to history
        self.history.append(DialogueTurn(
            speaker="human",
            text=text
        ))

        # Classify intent
        intent, entities = self._classify_intent(text)

        # Generate response based on mode
        if self.mode == InteractionMode.CLARIFICATION:
            response = self._handle_clarification(text, entities)
        else:
            response = self._handle_command(text, intent, entities)

        # Add response to history
        self.history.append(DialogueTurn(
            speaker="robot",
            text=response
        ))

        return response

    def _classify_intent(self, text: str) -> tuple:
        """Classify user intent."""
        prompt = f"""Classify this user input for a humanoid robot:

Input: "{text}"

Available robot capabilities:
{', '.join(self.capabilities)}

Respond with JSON:
{{
    "intent": "command|question|feedback|greeting|other",
    "action": "capability name if command",
    "entities": {{"object": "...", "location": "..."}}
}}
"""
        response = self.llm.generate(prompt)

        try:
            import json
            result = json.loads(response)
            return result.get("intent"), result.get("entities", {})
        except:
            return "other", {}

    def _handle_command(self, text: str, intent: str, entities: Dict) -> str:
        """Handle command input."""
        if intent == "command":
            # Check if enough information
            if self._needs_clarification(entities):
                self.mode = InteractionMode.CLARIFICATION
                self.pending_clarification = entities
                return self._generate_clarification_question(entities)
            else:
                return f"I'll {text.lower()}."

        elif intent == "question":
            return self._answer_question(text)

        elif intent == "greeting":
            return "Hello! How can I help you today?"

        elif intent == "feedback":
            return "Thank you for the feedback."

        else:
            return "I'm not sure I understand. Could you rephrase that?"

    def _handle_clarification(self, text: str, entities: Dict) -> str:
        """Handle clarification response."""
        # Merge with pending
        if self.pending_clarification:
            self.pending_clarification.update(entities)

        if self._needs_clarification(self.pending_clarification):
            return self._generate_clarification_question(self.pending_clarification)

        # Complete - return to command mode
        self.mode = InteractionMode.COMMAND
        response = f"Got it. I'll do that now."
        self.pending_clarification = None
        return response

    def _needs_clarification(self, entities: Dict) -> bool:
        """Check if clarification is needed."""
        # Example: check for ambiguous references
        if entities.get("object") == "it":
            return True
        if entities.get("location") in ["there", "here"]:
            return True
        return False

    def _generate_clarification_question(self, entities: Dict) -> str:
        """Generate clarification question."""
        if entities.get("object") == "it":
            return "Which object do you mean?"
        if entities.get("location") in ["there", "here"]:
            return "Could you specify the location?"
        return "Could you provide more details?"

    def _answer_question(self, question: str) -> str:
        """Answer a question about capabilities or state."""
        prompt = f"""Answer this question about a humanoid robot's capabilities:

Question: {question}

The robot can:
{', '.join(self.capabilities)}

Provide a helpful, concise answer:"""

        return self.llm.generate(prompt)

    def get_dialogue_context(self, num_turns: int = 5) -> str:
        """Get recent dialogue context."""
        recent = self.history[-num_turns:]
        return "\n".join([f"{t.speaker}: {t.text}" for t in recent])


class GestureRecognition:
    """Recognize human gestures for interaction."""

    def __init__(self):
        self.gesture_map = {
            "wave": self._detect_wave,
            "point": self._detect_point,
            "stop": self._detect_stop,
            "come": self._detect_come
        }

    def recognize(self, pose_landmarks: Dict) -> Optional[str]:
        """Recognize gesture from pose landmarks."""
        for gesture_name, detector in self.gesture_map.items():
            if detector(pose_landmarks):
                return gesture_name
        return None

    def _detect_wave(self, landmarks: Dict) -> bool:
        """Detect waving gesture."""
        # Hand above head, moving side to side
        return False  # Implement based on pose format

    def _detect_point(self, landmarks: Dict) -> bool:
        """Detect pointing gesture."""
        return False

    def _detect_stop(self, landmarks: Dict) -> bool:
        """Detect stop gesture (palm facing robot)."""
        return False

    def _detect_come(self, landmarks: Dict) -> bool:
        """Detect come here gesture."""
        return False


class HumanInteractionModule:
    """Complete human interaction module."""

    def __init__(
        self,
        llm_client,
        speech_recognizer,
        speech_synthesizer,
        robot_capabilities: List[str]
    ):
        self.dialogue = DialogueManager(llm_client, robot_capabilities)
        self.speech_recognizer = speech_recognizer
        self.speech_synthesizer = speech_synthesizer
        self.gesture_recognition = GestureRecognition()

        self.on_command: Optional[Callable] = None
        self.on_stop: Optional[Callable] = None

    def process_audio(self, audio: np.ndarray) -> Optional[str]:
        """Process audio input."""
        text = self.speech_recognizer.transcribe(audio)
        if text:
            response = self.dialogue.process_input(text)
            self.speech_synthesizer.speak(response)
            return text
        return None

    def process_gesture(self, pose_landmarks: Dict) -> Optional[str]:
        """Process gesture input."""
        gesture = self.gesture_recognition.recognize(pose_landmarks)

        if gesture == "stop" and self.on_stop:
            self.on_stop()
            self.speech_synthesizer.speak("Stopping.")

        elif gesture == "come" and self.on_command:
            self.on_command("come to me")
            self.speech_synthesizer.speak("Coming.")

        return gesture
```

---

## 16.4 Learning from Experience

### Online Learning

```python
# online_learning.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
from collections import deque

class ExperienceBuffer:
    """Buffer for online learning experiences."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to buffer."""
        self.buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })

    def sample(self, batch_size: int) -> Dict:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]

        return {
            "states": np.array([b["state"] for b in batch]),
            "actions": np.array([b["action"] for b in batch]),
            "rewards": np.array([b["reward"] for b in batch]),
            "next_states": np.array([b["next_state"] for b in batch]),
            "dones": np.array([b["done"] for b in batch])
        }

    def __len__(self):
        return len(self.buffer)


class OnlinePolicyAdapter:
    """Adapt policy online from experiences."""

    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 1e-4,
        adaptation_steps: int = 10
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate
        )
        self.adaptation_steps = adaptation_steps
        self.buffer = ExperienceBuffer()

    def add_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience and potentially trigger adaptation."""
        self.buffer.add(state, action, reward, next_state, done)

        # Adapt periodically
        if len(self.buffer) >= 64 and len(self.buffer) % 10 == 0:
            self.adapt()

    def adapt(self, batch_size: int = 32):
        """Perform adaptation step."""
        if len(self.buffer) < batch_size:
            return

        for _ in range(self.adaptation_steps):
            batch = self.buffer.sample(batch_size)

            states = torch.tensor(batch["states"], dtype=torch.float32)
            actions = torch.tensor(batch["actions"], dtype=torch.float32)
            rewards = torch.tensor(batch["rewards"], dtype=torch.float32)

            # Simple behavior cloning weighted by reward
            predicted_actions = self.policy(states)
            loss = ((predicted_actions - actions) ** 2).mean(dim=-1)

            # Weight by normalized rewards
            weights = torch.softmax(rewards, dim=0)
            weighted_loss = (loss * weights).sum()

            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()


class ImitationLearner:
    """Learn from human demonstrations."""

    def __init__(self, policy: nn.Module, feature_extractor):
        self.policy = policy
        self.feature_extractor = feature_extractor

        self.demonstrations = []
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    def add_demonstration(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray]
    ):
        """Add a demonstration."""
        self.demonstrations.append({
            "observations": observations,
            "actions": actions
        })

    def train(self, epochs: int = 100, batch_size: int = 32):
        """Train policy from demonstrations."""
        # Flatten demonstrations
        all_obs = []
        all_actions = []

        for demo in self.demonstrations:
            all_obs.extend(demo["observations"])
            all_actions.extend(demo["actions"])

        all_obs = np.array(all_obs)
        all_actions = np.array(all_actions)

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(len(all_obs))

            total_loss = 0
            num_batches = 0

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]

                obs = torch.tensor(all_obs[batch_indices], dtype=torch.float32)
                actions = torch.tensor(all_actions[batch_indices], dtype=torch.float32)

                # Forward pass
                predicted = self.policy(obs)
                loss = nn.MSELoss()(predicted, actions)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / num_batches:.4f}")


class SkillLearner:
    """Learn new skills from demonstrations and practice."""

    def __init__(self, base_policy, skill_library):
        self.base_policy = base_policy
        self.skill_library = skill_library
        self.imitation_learner = ImitationLearner(base_policy, None)

    def learn_skill_from_demo(
        self,
        skill_name: str,
        demonstrations: List[Dict]
    ):
        """Learn a new skill from demonstrations."""
        # Extract observations and actions
        obs_list = []
        action_list = []

        for demo in demonstrations:
            obs_list.append(demo["observations"])
            action_list.append(demo["actions"])

        # Flatten
        all_obs = [o for obs in obs_list for o in obs]
        all_actions = [a for act in action_list for a in act]

        # Train
        self.imitation_learner.add_demonstration(all_obs, all_actions)
        self.imitation_learner.train()

        # Register skill
        # self.skill_library.register(...)

        print(f"Learned skill: {skill_name}")

    def refine_skill(
        self,
        skill_name: str,
        experiences: List[Dict]
    ):
        """Refine skill from online experiences."""
        # Extract successful experiences
        successful = [e for e in experiences if e["success"]]

        if successful:
            obs = [e["observation"] for e in successful]
            actions = [e["action"] for e in successful]
            self.imitation_learner.add_demonstration(obs, actions)
            self.imitation_learner.train(epochs=20)
```

---

## 16.5 Deployment

### Complete Agent System

```python
# deployment.py
import asyncio
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanoidAgentSystem:
    """Complete deployable humanoid agent system."""

    def __init__(self, config: Dict):
        self.config = config

        # Initialize components
        logger.info("Initializing robot interface...")
        self.robot = self._init_robot(config["robot"])

        logger.info("Initializing perception...")
        self.perception = self._init_perception(config["perception"])

        logger.info("Initializing planner...")
        self.planner = self._init_planner(config["planner"])

        logger.info("Initializing memory...")
        self.memory = self._init_memory(config["memory"])

        logger.info("Initializing human interaction...")
        self.interaction = self._init_interaction(config["interaction"])

        # Create agent
        self.agent = EmbodiedAgent(
            config=AgentConfig(**config.get("agent", {})),
            robot_interface=self.robot,
            perception_system=self.perception,
            planner=self.planner,
            memory_system=self.memory
        )

        # Connect callbacks
        self.agent.on_state_change = self._on_state_change
        self.agent.on_task_complete = self._on_task_complete
        self.agent.on_error = self._on_error

        self.interaction.on_command = self.agent.give_task
        self.interaction.on_stop = self.agent.stop

    def _init_robot(self, config: Dict):
        """Initialize robot interface."""
        # Return appropriate robot interface
        pass

    def _init_perception(self, config: Dict):
        """Initialize perception system."""
        return MultimodalPerceptionPipeline(config)

    def _init_planner(self, config: Dict):
        """Initialize planner."""
        # Return LLM planner with skill library
        pass

    def _init_memory(self, config: Dict):
        """Initialize memory system."""
        # Return memory system
        pass

    def _init_interaction(self, config: Dict):
        """Initialize human interaction."""
        # Return interaction module
        pass

    def _on_state_change(self, old_state, new_state):
        """Handle state change."""
        logger.info(f"Agent state: {old_state} -> {new_state}")

    def _on_task_complete(self, task: str):
        """Handle task completion."""
        logger.info(f"Task completed: {task}")

    def _on_error(self, error: str):
        """Handle error."""
        logger.error(f"Agent error: {error}")

    async def run(self):
        """Run the agent system."""
        logger.info("Starting humanoid agent system...")

        try:
            await self.agent.start()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await self.agent.stop()

    def run_sync(self):
        """Synchronous run method."""
        asyncio.run(self.run())


# Example configuration
EXAMPLE_CONFIG = {
    "robot": {
        "type": "ros2",
        "namespace": "/humanoid"
    },
    "perception": {
        "vision": {"model": "owlvit"},
        "audio": {"model": "whisper"},
        "tactile": {"enabled": True}
    },
    "planner": {
        "llm": "gpt-4",
        "skills": ["navigate", "pick_up", "place", "speak"]
    },
    "memory": {
        "episodic_capacity": 10000,
        "embedding_model": "all-MiniLM-L6-v2"
    },
    "interaction": {
        "speech_enabled": True,
        "gesture_enabled": True
    },
    "agent": {
        "name": "Atlas",
        "perception_rate": 30.0,
        "planning_rate": 10.0,
        "control_rate": 100.0
    }
}

if __name__ == "__main__":
    system = HumanoidAgentSystem(EXAMPLE_CONFIG)
    system.run_sync()
```

---

## Exercises

### Exercise 16.1: Basic Agent Loop

**Objective**: Implement perception-action loop.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Create basic agent class
2. Implement perception loop
3. Add simple reactive control
4. Test with simulated robot

---

### Exercise 16.2: Memory System

**Objective**: Add episodic memory to agent.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Implement experience storage
2. Add similarity-based retrieval
3. Use memory for planning
4. Test memory recall

---

### Exercise 16.3: Dialogue System

**Objective**: Add natural language interaction.

**Difficulty**: Intermediate | **Estimated Time**: 60 minutes

#### Instructions

1. Implement intent classification
2. Add dialogue management
3. Handle clarifications
4. Test conversation flow

---

### Exercise 16.4: Deploy Complete Agent

**Objective**: Deploy full agent system.

**Difficulty**: Advanced | **Estimated Time**: 90 minutes

#### Instructions

1. Integrate all components
2. Configure for your robot
3. Test end-to-end
4. Handle edge cases

---

## Summary

In this chapter, you learned:

- **Embodied agents** combine perception, planning, and action
- **Memory systems** enable learning from experience
- **Human interaction** makes robots accessible
- **Online learning** improves performance over time
- **System deployment** requires careful integration

---

## References

[1] D. Driess et al., "PaLM-E: An Embodied Multimodal Language Model," in *ICML*, 2023.

[2] A. Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," in *CoRL*, 2023.

[3] Y. Jiang et al., "VIMA: General Robot Manipulation with Multimodal Prompts," in *ICML*, 2023.

[4] Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models," 2024.
