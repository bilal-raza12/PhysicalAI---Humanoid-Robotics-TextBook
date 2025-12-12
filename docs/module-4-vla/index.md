---
id: index
title: "Module 4: Vision-Language-Action (VLA)"
sidebar_position: 1
---

# Module 4: Vision-Language-Action (VLA)

**Estimated Time**: 24 hours | **Chapters**: 4 | **Exercises**: 13+

## Overview

Vision-Language-Action (VLA) represents the frontier of human-robot interaction. In this module, you'll build systems that enable robots to understand spoken commands, reason about tasks using large language models, and execute complex multi-step behaviors. You'll create robots that don't just follow scripts—they understand intent and adapt to changing situations.

## Learning Outcomes

By completing this module, you will be able to:

- **Integrate speech recognition** using OpenAI Whisper with ROS 2
- **Build cognitive planners** that use LLMs to decompose high-level tasks
- **Design perception-action loops** for closed-loop robot control
- **Create natural language interfaces** that ground language in robot actions
- **Implement end-to-end VLA pipelines** from voice to robot motion

## Prerequisites

Before starting this module, ensure you have completed:

- [ ] **Module 1**: ROS 2 fundamentals
- [ ] **Module 3**: Isaac Sim and perception basics
- [ ] OpenAI API key (or local LLM setup with Ollama)
- [ ] Microphone for voice input testing

## Chapters

### Chapter 13: Voice-to-Action with Whisper + ROS 2

Build voice command interfaces that translate spoken instructions into robot actions using OpenAI's Whisper model.

**Topics**: Whisper integration, audio processing, intent recognition, ROS 2 audio pipeline

### Chapter 14: LLM Cognitive Planning for Robots

Connect large language models to robot action spaces for intelligent task decomposition and planning.

**Topics**: LLM APIs, action primitives, task decomposition, failure recovery, local LLMs with Ollama

### Chapter 15: Perception-Action Loops

Design closed-loop control systems that continuously sense, plan, and act in dynamic environments.

**Topics**: Control architectures, reactive behaviors, sensor fusion, latency optimization

### Chapter 16: Natural-Language Task Execution

Build complete systems that ground natural language in robot actions and execute complex commands.

**Topics**: Language grounding, semantic parsing, action sequencing, error handling

## Key Concepts

| Concept | Description |
|---------|-------------|
| **VLA** | Vision-Language-Action model architecture |
| **Whisper** | OpenAI's speech recognition model |
| **LLM** | Large Language Model for reasoning |
| **Grounding** | Mapping language to physical actions |
| **Closed-Loop Control** | Continuous sensing and action feedback |

## Software Requirements

| Software | Version | Installation |
|----------|---------|--------------|
| Whisper | Latest | Chapter 13 |
| OpenAI API | GPT-4 | Chapter 14 |
| Ollama | Latest | Chapter 14 (alternative) |
| LangChain | Latest | Chapter 14 |

## API Requirements

You'll need one of:
- **OpenAI API Key**: For cloud-based LLM and Whisper
- **Ollama + Local LLM**: For offline/free alternative (documented in Chapter 14)

:::tip Cost Management
This module documents both cloud APIs and free local alternatives. You can complete all exercises using only free, local models if preferred.
:::

## What's Next

With VLA capabilities complete, you're ready for the **Capstone Project**: building a fully autonomous humanoid that integrates everything you've learned—navigation, perception, manipulation, and natural language control.

---

_Ready to begin? Start with [Chapter 13: Voice-to-Action with Whisper + ROS 2](./ch13-voice-action)._
