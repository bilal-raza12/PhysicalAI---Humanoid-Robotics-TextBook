# Research: Physical AI & Humanoid Robotics Textbook

**Feature**: `1-humanoid-robotics-textbook`
**Date**: 2025-12-12
**Status**: Complete

## Decision Log

### D1: ROS Version Selection

**Decision**: ROS 2 Humble Hawksbill (LTS)

**Rationale**:
- Long-term support until May 2027 provides stability for educational content
- Mature ecosystem with comprehensive documentation
- Wide industry adoption ensures reader skills are transferable
- Compatible with both Gazebo Fortress and NVIDIA Isaac

**Alternatives Considered**:
| Alternative | Pros | Cons | Why Rejected |
|------------|------|------|--------------|
| ROS 2 Iron | Latest features, improved performance | EOL May 2024 (too soon) | Short support window inappropriate for textbook longevity |
| ROS 2 Jazzy | Newest LTS (2024) | Less mature, fewer tutorials | Insufficient community resources for educational content |
| ROS 1 Noetic | Large existing codebase | EOL May 2025, no future | Deprecated platform inappropriate for new learners |

---

### D2: Simulation Platform Architecture

**Decision**: Dual-platform approach - Gazebo Fortress (primary) + Unity (supplementary)

**Rationale**:
- Gazebo Fortress: Native ROS 2 integration, open-source, physics-accurate, industry standard
- Unity: High-fidelity rendering for human-robot interaction scenarios, ML-Agents integration
- Both platforms complement each other rather than compete
- Readers gain exposure to multiple industry tools

**Alternatives Considered**:
| Alternative | Pros | Cons | Why Rejected |
|------------|------|------|--------------|
| Gazebo only | Simpler, single toolchain | Limited visual fidelity | Misses HRI use cases |
| Unity only | Best graphics, wide adoption | Poor ROS 2 integration, closed source | Not native robotics simulator |
| MuJoCo | Excellent physics, DeepMind backing | Steep learning curve, limited sensors | Too specialized for general textbook |
| PyBullet | Python-native, simple | Limited features, poor scaling | Insufficient for advanced content |

---

### D3: AI Planning Engine for VLA Module

**Decision**: Hybrid approach - OpenAI API (cloud) with local LLM fallback (Ollama + Llama)

**Rationale**:
- OpenAI provides state-of-the-art performance for primary examples
- Local LLM option ensures readers without API access can still learn
- Demonstrates both cloud and edge deployment patterns
- Cost-conscious approach aligns with educational context

**Alternatives Considered**:
| Alternative | Pros | Cons | Why Rejected |
|------------|------|------|--------------|
| OpenAI only | Best quality, simplest setup | Requires paid API, internet dependency | Excludes budget-constrained readers |
| Local only | Free, offline, privacy | Lower quality, high hardware requirements | Misses cloud integration skills |
| Anthropic Claude | Strong reasoning | Smaller ecosystem, API-only | Less robotics-specific tooling |

---

### D4: Deployment Method

**Decision**: Docusaurus via Context7 MCP + GitHub Pages

**Rationale**:
- Docusaurus: Purpose-built for documentation, excellent Markdown support, versioning
- Context7 MCP: Provides AI-assisted content management capabilities
- GitHub Pages: Free hosting, CI/CD integration, aligns with constitution requirements
- Combined approach enables automated builds per constitution

**Alternatives Considered**:
| Alternative | Pros | Cons | Why Rejected |
|------------|------|------|--------------|
| GitBook | Beautiful UI, easy setup | Limited customization, paid for teams | Free tier insufficient |
| MkDocs | Python ecosystem, simple | Less feature-rich, weaker versioning | Missing advanced features |
| Sphinx | Python standard, powerful | Complex configuration, dated styling | Poor modern UX |
| Custom Next.js | Full control | Maintenance burden, time investment | Over-engineering for content site |

---

### D5: Citation Style

**Decision**: IEEE Citation Style

**Rationale**:
- Industry standard for robotics and computer science publications
- Numeric citations minimize text disruption
- Well-supported by reference managers (Zotero, Mendeley)
- Consistent with target audience expectations (academic/technical)

**Alternatives Considered**:
| Alternative | Pros | Cons | Why Rejected |
|------------|------|------|--------------|
| APA | Author-date visible | Longer inline citations | Less common in robotics |
| Chicago | Flexible, comprehensive | Complex rules | Overkill for technical content |
| Harvard | Simple author-date | Less precise numbering | Not standard in engineering |

---

### D6: Humanoid Robot Model

**Decision**: Use open-source humanoid models (Digit, Atlas-inspired, custom URDF)

**Rationale**:
- Open-source models allow readers to modify and experiment
- Multiple models demonstrate transferable skills
- Custom URDF creation teaches fundamental concepts
- Avoids licensing issues with proprietary robots

**Selected Models**:
1. **Primary**: Custom humanoid URDF (built progressively in Module 1)
2. **Reference**: Digit (Agility Robotics) - open simulation model
3. **Advanced**: NVIDIA Isaac humanoid assets for Module 3

---

### D7: Code Example Languages

**Decision**: Python primary, C++ secondary for performance-critical sections

**Rationale**:
- Python: rclpy is the standard for ROS 2 Python development
- Lower barrier to entry for target audience
- C++ shown only where necessary (real-time control, performance)
- Aligns with industry trends toward Python-first robotics

---

## Technology Stack Summary

| Component | Selected Technology | Version |
|-----------|-------------------|---------|
| Robot Framework | ROS 2 | Humble Hawksbill |
| Primary Simulator | Gazebo | Fortress (LTS) |
| Secondary Simulator | Unity | 2022.3 LTS |
| AI Simulator | NVIDIA Isaac Sim | 2023.1+ |
| Speech Recognition | OpenAI Whisper | Latest |
| LLM Planning | OpenAI GPT-4 / Ollama + Llama | Latest |
| Navigation Stack | Nav2 | Humble-compatible |
| Book Framework | Docusaurus | 3.x |
| Deployment | GitHub Pages | - |
| CI/CD | GitHub Actions | - |
| Code Language | Python 3.10+ | Primary |
| Citation Style | IEEE | - |

---

## Content Research

### Module 1 - ROS 2 References

1. **Official ROS 2 Documentation**: https://docs.ros.org/en/humble/
2. **ROS 2 Design**: https://design.ros2.org/
3. **URDF Tutorial**: https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/
4. **rclpy API**: https://docs.ros2.org/humble/api/rclpy/

### Module 2 - Simulation References

1. **Gazebo Fortress**: https://gazebosim.org/docs/fortress/
2. **ROS 2 Gazebo Integration**: https://gazebosim.org/docs/fortress/ros2_integration
3. **Unity Robotics Hub**: https://github.com/Unity-Technologies/Unity-Robotics-Hub
4. **Sensor Plugins**: https://gazebosim.org/api/sensors/

### Module 3 - NVIDIA Isaac References

1. **Isaac Sim Documentation**: https://docs.omniverse.nvidia.com/isaacsim/
2. **Isaac ROS**: https://nvidia-isaac-ros.github.io/
3. **Nav2 Documentation**: https://navigation.ros.org/
4. **Synthetic Data Generation**: https://docs.omniverse.nvidia.com/replicator/

### Module 4 - VLA References

1. **Whisper API**: https://platform.openai.com/docs/guides/speech-to-text
2. **LangChain for Robotics**: https://python.langchain.com/
3. **ROS 2 Audio Common**: https://github.com/ros-drivers/audio_common
4. **SayCan Paper**: "Do As I Can, Not As I Say" (Google Research, 2022)

### Capstone References

1. **MoveIt 2**: https://moveit.ros.org/
2. **Object Detection with ROS 2**: YOLO/Detectron2 integration guides
3. **Task Planning**: PDDL + ROSPlan resources

---

## Hardware Requirements

### Minimum Requirements
- **CPU**: 8-core modern processor (Intel i7/AMD Ryzen 7)
- **RAM**: 16GB (32GB recommended)
- **GPU**: NVIDIA RTX 3060 or equivalent (for Isaac Sim)
- **Storage**: 100GB free SSD space
- **OS**: Ubuntu 22.04 LTS

### Cloud Alternatives
- **AWS RoboMaker**: For Gazebo simulation
- **NVIDIA NGC**: For Isaac Sim cloud instances
- **Google Colab Pro**: For ML/LLM experiments

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Software version changes | High | Medium | Pin versions, document compatibility |
| NVIDIA Isaac access barriers | Medium | High | Provide Gazebo-only alternatives |
| OpenAI API costs | Medium | Medium | Document local LLM fallbacks |
| Reader hardware limitations | High | High | Cloud alternatives for all modules |
| Gazebo/ROS breaking changes | Low | High | Test against LTS versions only |

---

## Open Questions (Resolved)

1. ~~Which humanoid model to use?~~ → Custom URDF + open-source references
2. ~~Gazebo Classic vs Ignition/Fortress?~~ → Gazebo Fortress (modern, supported)
3. ~~Include manipulation in capstone?~~ → Yes, per FR-029
4. ~~Cloud vs local LLM?~~ → Hybrid approach with both options
