# Tasks: Physical AI & Humanoid Robotics Textbook

**Input**: Design documents from `/specs/1-humanoid-robotics-textbook/`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md, contracts/book-structure.yaml

**Tests**: Tests are NOT explicitly requested in the specification. Build validation tasks are included as part of infrastructure setup.

**Organization**: Tasks are organized by user story to enable independent implementation and testing of each module.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US5)
- Include exact file paths in descriptions

## Path Conventions

- **Documentation**: `docs/` at repository root
- **Static assets**: `static/img/` and `static/code/`
- **Configuration**: Root level (`docusaurus.config.js`, `sidebars.js`)

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Initialize Docusaurus project and CI/CD infrastructure

- [x] T001 Initialize Docusaurus 3.x project with `npx create-docusaurus@latest` in repository root
- [x] T002 Configure site metadata in docusaurus.config.js (title, tagline, URL, baseUrl)
- [x] T003 [P] Create custom CSS file at src/css/custom.css with book typography styles
- [x] T004 [P] Configure sidebars.js with module-based navigation structure
- [x] T005 [P] Create GitHub Actions workflow at .github/workflows/deploy.yml for CI/CD
- [x] T006 [P] Configure markdownlint with .markdownlint.json for content validation
- [x] T007 Create directory structure: docs/, static/img/diagrams/, static/img/screenshots/, static/code/
- [x] T008 [P] Add package.json scripts for lint, build, and deploy commands
- [ ] T009 Verify local development server runs with `npm run start`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Front matter and book structure that MUST be complete before module content

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T010 Create landing page at docs/intro.md with book overview and navigation guide
- [x] T011 [P] Create prerequisites checklist at docs/prerequisites.md with self-assessment criteria
- [x] T012 [P] Create conventions document at docs/conventions.md explaining notation and formatting
- [x] T013 [P] Create glossary structure at docs/appendices/glossary.md with initial term list
- [x] T014 [P] Create appendices category config at docs/appendices/_category_.json
- [x] T015 Create Module 1 category config at docs/module-1-ros2/_category_.json
- [x] T016 [P] Create Module 2 category config at docs/module-2-digital-twin/_category_.json
- [x] T017 [P] Create Module 3 category config at docs/module-3-nvidia-isaac/_category_.json
- [x] T018 [P] Create Module 4 category config at docs/module-4-vla/_category_.json
- [x] T019 [P] Create Capstone category config at docs/capstone/_category_.json
- [x] T020 Create Module 1 overview at docs/module-1-ros2/index.md with learning outcomes
- [x] T021 [P] Create Module 2 overview at docs/module-2-digital-twin/index.md with learning outcomes
- [x] T022 [P] Create Module 3 overview at docs/module-3-nvidia-isaac/index.md with learning outcomes
- [x] T023 [P] Create Module 4 overview at docs/module-4-vla/index.md with learning outcomes
- [x] T024 [P] Create Capstone overview at docs/capstone/index.md with project requirements
- [ ] T025 Verify build succeeds with `npm run build` and all navigation links work

**Checkpoint**: Foundation ready - user story implementation can begin

---

## Phase 3: User Story 1 - ROS 2 Foundations (Priority: P1)

**Goal**: Complete foundational ROS 2 content (4 chapters, ~20 hours reader time)

**Independent Test**: Reader can create a humanoid URDF model with ROS 2 nodes communicating via topics after completing Module 1

### Chapter 1: Introduction to ROS 2 for Humanoids

- [ ] T026 [P] [US1] Create chapter file at docs/module-1-ros2/ch01-intro-ros2.md with front matter
- [ ] T027 [US1] Write learning objectives section (3-5 objectives) in ch01-intro-ros2.md
- [ ] T028 [US1] Write introduction section explaining ROS 2 for humanoids in ch01-intro-ros2.md
- [ ] T029 [US1] Write ROS 2 architecture section (nodes, topics, services overview) in ch01-intro-ros2.md
- [ ] T030 [US1] Write development environment setup section in ch01-intro-ros2.md
- [ ] T031 [P] [US1] Create architecture diagram at static/img/diagrams/ch01-ros2-architecture.svg
- [ ] T032 [US1] Write Exercise 1.1: Install ROS 2 Humble with troubleshooting in ch01-intro-ros2.md
- [ ] T033 [US1] Write Exercise 1.2: Create first ROS 2 package with verification in ch01-intro-ros2.md
- [ ] T034 [US1] Write Exercise 1.3: Run talker/listener demo with expected output in ch01-intro-ros2.md
- [ ] T035 [US1] Write summary and references section (IEEE format) in ch01-intro-ros2.md
- [ ] T036 [P] [US1] Create code examples at static/code/module-1/ch01/ (package setup, basic node)

### Chapter 2: Nodes, Topics, Services & Launch Systems

- [ ] T037 [P] [US1] Create chapter file at docs/module-1-ros2/ch02-nodes-topics.md with front matter
- [ ] T038 [US1] Write learning objectives section in ch02-nodes-topics.md
- [ ] T039 [US1] Write publisher/subscriber patterns section in ch02-nodes-topics.md
- [ ] T040 [US1] Write services and actions section in ch02-nodes-topics.md
- [ ] T041 [US1] Write launch files section in ch02-nodes-topics.md
- [ ] T042 [P] [US1] Create node communication diagram at static/img/diagrams/ch02-node-communication.svg
- [ ] T043 [US1] Write Exercise 2.1: Create custom publisher node with troubleshooting in ch02-nodes-topics.md
- [ ] T044 [US1] Write Exercise 2.2: Implement service server/client with verification in ch02-nodes-topics.md
- [ ] T045 [US1] Write Exercise 2.3: Build multi-node launch file with expected output in ch02-nodes-topics.md
- [ ] T046 [US1] Write Exercise 2.4: Debug node communication with tools in ch02-nodes-topics.md
- [ ] T047 [US1] Write summary and references section in ch02-nodes-topics.md
- [ ] T048 [P] [US1] Create code examples at static/code/module-1/ch02/ (publisher, subscriber, service, launch)

### Chapter 3: URDF & Modeling Humanoid Kinematics

- [ ] T049 [P] [US1] Create chapter file at docs/module-1-ros2/ch03-urdf-kinematics.md with front matter
- [ ] T050 [US1] Write learning objectives section in ch03-urdf-kinematics.md
- [ ] T051 [US1] Write URDF fundamentals section (links, joints, visual/collision) in ch03-urdf-kinematics.md
- [ ] T052 [US1] Write humanoid kinematic chain section in ch03-urdf-kinematics.md
- [ ] T053 [US1] Write RViz2 visualization section in ch03-urdf-kinematics.md
- [ ] T054 [P] [US1] Create humanoid URDF diagram at static/img/diagrams/ch03-humanoid-urdf.svg
- [ ] T055 [P] [US1] Create kinematic chain diagram at static/img/diagrams/ch03-kinematic-chain.svg
- [ ] T056 [US1] Write Exercise 3.1: Create simple robot arm URDF with troubleshooting in ch03-urdf-kinematics.md
- [ ] T057 [US1] Write Exercise 3.2: Build humanoid torso with joints in ch03-urdf-kinematics.md
- [ ] T058 [US1] Write Exercise 3.3: Visualize model in RViz2 with verification in ch03-urdf-kinematics.md
- [ ] T059 [US1] Write summary and references section in ch03-urdf-kinematics.md
- [ ] T060 [P] [US1] Create URDF examples at static/code/module-1/ch03/ (simple arm, humanoid torso)

### Chapter 4: Integrating Python AI Agents with rclpy

- [ ] T061 [P] [US1] Create chapter file at docs/module-1-ros2/ch04-python-agents.md with front matter
- [ ] T062 [US1] Write learning objectives section in ch04-python-agents.md
- [ ] T063 [US1] Write rclpy fundamentals section in ch04-python-agents.md
- [ ] T064 [US1] Write behavior trees section in ch04-python-agents.md
- [ ] T065 [US1] Write external AI service integration section in ch04-python-agents.md
- [ ] T066 [P] [US1] Create AI agent architecture diagram at static/img/diagrams/ch04-ai-agent-arch.svg
- [ ] T067 [US1] Write Exercise 4.1: Create rclpy-based decision node with troubleshooting in ch04-python-agents.md
- [ ] T068 [US1] Write Exercise 4.2: Implement simple behavior tree in ch04-python-agents.md
- [ ] T069 [US1] Write Exercise 4.3: Connect to external API from ROS 2 node in ch04-python-agents.md
- [ ] T070 [US1] Write summary and references section in ch04-python-agents.md
- [ ] T071 [P] [US1] Create Python examples at static/code/module-1/ch04/ (rclpy node, behavior tree)

**Checkpoint**: Module 1 complete - readers can create ROS 2 nodes and URDF models independently

---

## Phase 4: User Story 2 - Digital Twin Simulation (Priority: P2)

**Goal**: Complete simulation content (4 chapters, ~24 hours reader time)

**Independent Test**: Reader can run a simulated humanoid robot in Gazebo with sensor data publishing to ROS 2 topics

### Chapter 5: Introduction to Physics Simulation

- [ ] T072 [P] [US2] Create chapter file at docs/module-2-digital-twin/ch05-physics-sim.md with front matter
- [ ] T073 [US2] Write learning objectives section in ch05-physics-sim.md
- [ ] T074 [US2] Write physics engine fundamentals section in ch05-physics-sim.md
- [ ] T075 [US2] Write simulation parameters section in ch05-physics-sim.md
- [ ] T076 [US2] Write collision detection section in ch05-physics-sim.md
- [ ] T077 [P] [US2] Create physics simulation diagram at static/img/diagrams/ch05-physics-sim.svg
- [ ] T078 [US2] Write Exercise 5.1: Configure physics parameters with troubleshooting in ch05-physics-sim.md
- [ ] T079 [US2] Write Exercise 5.2: Test collision detection in ch05-physics-sim.md
- [ ] T080 [US2] Write Exercise 5.3: Optimize simulation performance in ch05-physics-sim.md
- [ ] T081 [US2] Write summary and references section in ch05-physics-sim.md

### Chapter 6: Building a Humanoid Digital Twin in Gazebo

- [ ] T082 [P] [US2] Create chapter file at docs/module-2-digital-twin/ch06-gazebo-twin.md with front matter
- [ ] T083 [US2] Write learning objectives section in ch06-gazebo-twin.md
- [ ] T084 [US2] Write URDF to SDF conversion section in ch06-gazebo-twin.md
- [ ] T085 [US2] Write Gazebo world configuration section in ch06-gazebo-twin.md
- [ ] T086 [US2] Write ROS 2-Gazebo bridge section in ch06-gazebo-twin.md
- [ ] T087 [P] [US2] Create Gazebo architecture diagram at static/img/diagrams/ch06-gazebo-arch.svg
- [ ] T088 [US2] Write Exercise 6.1: Convert URDF to SDF with troubleshooting in ch06-gazebo-twin.md
- [ ] T089 [US2] Write Exercise 6.2: Create custom Gazebo world in ch06-gazebo-twin.md
- [ ] T090 [US2] Write Exercise 6.3: Configure ros_gz_bridge in ch06-gazebo-twin.md
- [ ] T091 [US2] Write Exercise 6.4: Spawn humanoid in Gazebo world in ch06-gazebo-twin.md
- [ ] T092 [US2] Write summary and references section in ch06-gazebo-twin.md
- [ ] T093 [P] [US2] Create code examples at static/code/module-2/ch06/ (SDF, world file, bridge config)

### Chapter 7: Sensor Simulation: LiDAR, Depth, IMU

- [ ] T094 [P] [US2] Create chapter file at docs/module-2-digital-twin/ch07-sensor-sim.md with front matter
- [ ] T095 [US2] Write learning objectives section in ch07-sensor-sim.md
- [ ] T096 [US2] Write LiDAR simulation section with noise models in ch07-sensor-sim.md
- [ ] T097 [US2] Write depth camera simulation section in ch07-sensor-sim.md
- [ ] T098 [US2] Write IMU simulation section in ch07-sensor-sim.md
- [ ] T099 [P] [US2] Create sensor data flow diagram at static/img/diagrams/ch07-sensor-flow.svg
- [ ] T100 [US2] Write Exercise 7.1: Add LiDAR sensor to humanoid with troubleshooting in ch07-sensor-sim.md
- [ ] T101 [US2] Write Exercise 7.2: Configure depth camera with visualization in ch07-sensor-sim.md
- [ ] T102 [US2] Write Exercise 7.3: Implement IMU data pipeline in ch07-sensor-sim.md
- [ ] T103 [US2] Write summary and references section in ch07-sensor-sim.md
- [ ] T104 [P] [US2] Create sensor config examples at static/code/module-2/ch07/

### Chapter 8: Unity for High-Fidelity Interaction Scenes

- [ ] T105 [P] [US2] Create chapter file at docs/module-2-digital-twin/ch08-unity-hri.md with front matter
- [ ] T106 [US2] Write learning objectives section in ch08-unity-hri.md
- [ ] T107 [US2] Write Unity Robotics Hub setup section in ch08-unity-hri.md
- [ ] T108 [US2] Write photorealistic environment creation section in ch08-unity-hri.md
- [ ] T109 [US2] Write ROS-TCP connector section in ch08-unity-hri.md
- [ ] T110 [P] [US2] Create Unity-ROS architecture diagram at static/img/diagrams/ch08-unity-ros.svg
- [ ] T111 [US2] Write Exercise 8.1: Install Unity Robotics Hub with troubleshooting in ch08-unity-hri.md
- [ ] T112 [US2] Write Exercise 8.2: Create indoor environment in ch08-unity-hri.md
- [ ] T113 [US2] Write Exercise 8.3: Connect Unity to ROS 2 via TCP in ch08-unity-hri.md
- [ ] T114 [US2] Write summary and references section in ch08-unity-hri.md
- [ ] T115 [P] [US2] Create Unity project examples at static/code/module-2/ch08/

**Checkpoint**: Module 2 complete - readers can run humanoid simulation with sensors

---

## Phase 5: User Story 3 - NVIDIA Isaac Integration (Priority: P3)

**Goal**: Complete AI/Isaac integration content (4 chapters, ~28 hours reader time)

**Independent Test**: Reader can run a humanoid robot navigating autonomously in Isaac Sim using VSLAM and Nav2

### Chapter 9: Isaac Sim Photorealistic Simulation

- [ ] T116 [P] [US3] Create chapter file at docs/module-3-nvidia-isaac/ch09-isaac-sim.md with front matter
- [ ] T117 [US3] Write learning objectives section in ch09-isaac-sim.md
- [ ] T118 [US3] Write Isaac Sim installation section in ch09-isaac-sim.md
- [ ] T119 [US3] Write Omniverse model import section in ch09-isaac-sim.md
- [ ] T120 [US3] Write domain randomization section in ch09-isaac-sim.md
- [ ] T121 [P] [US3] Create Isaac Sim architecture diagram at static/img/diagrams/ch09-isaac-arch.svg
- [ ] T122 [US3] Write Exercise 9.1: Install Isaac Sim with troubleshooting in ch09-isaac-sim.md
- [ ] T123 [US3] Write Exercise 9.2: Import humanoid model in ch09-isaac-sim.md
- [ ] T124 [US3] Write Exercise 9.3: Create domain-randomized scene in ch09-isaac-sim.md
- [ ] T125 [US3] Write summary and references section in ch09-isaac-sim.md
- [ ] T126 [P] [US3] Create Isaac config examples at static/code/module-3/ch09/

### Chapter 10: Isaac ROS: VSLAM & Navigation

- [ ] T127 [P] [US3] Create chapter file at docs/module-3-nvidia-isaac/ch10-isaac-ros.md with front matter
- [ ] T128 [US3] Write learning objectives section in ch10-isaac-ros.md
- [ ] T129 [US3] Write Isaac ROS packages section in ch10-isaac-ros.md
- [ ] T130 [US3] Write visual SLAM section in ch10-isaac-ros.md
- [ ] T131 [US3] Write localization pipeline section in ch10-isaac-ros.md
- [ ] T132 [P] [US3] Create VSLAM pipeline diagram at static/img/diagrams/ch10-vslam-pipeline.svg
- [ ] T133 [US3] Write Exercise 10.1: Deploy Isaac ROS packages with troubleshooting in ch10-isaac-ros.md
- [ ] T134 [US3] Write Exercise 10.2: Implement VSLAM for humanoid in ch10-isaac-ros.md
- [ ] T135 [US3] Write Exercise 10.3: Configure localization pipeline in ch10-isaac-ros.md
- [ ] T136 [US3] Write Exercise 10.4: Test mapping in Isaac Sim in ch10-isaac-ros.md
- [ ] T137 [US3] Write summary and references section in ch10-isaac-ros.md
- [ ] T138 [P] [US3] Create VSLAM examples at static/code/module-3/ch10/

### Chapter 11: Nav2 for Bipedal Locomotion

- [ ] T139 [P] [US3] Create chapter file at docs/module-3-nvidia-isaac/ch11-nav2-bipedal.md with front matter
- [ ] T140 [US3] Write learning objectives section in ch11-nav2-bipedal.md
- [ ] T141 [US3] Write Nav2 adaptation for bipedal section in ch11-nav2-bipedal.md
- [ ] T142 [US3] Write footstep planning section in ch11-nav2-bipedal.md
- [ ] T143 [US3] Write dynamic obstacle avoidance section in ch11-nav2-bipedal.md
- [ ] T144 [P] [US3] Create Nav2 bipedal diagram at static/img/diagrams/ch11-nav2-bipedal.svg
- [ ] T145 [US3] Write Exercise 11.1: Configure Nav2 for bipedal robot with troubleshooting in ch11-nav2-bipedal.md
- [ ] T146 [US3] Write Exercise 11.2: Implement footstep planner in ch11-nav2-bipedal.md
- [ ] T147 [US3] Write Exercise 11.3: Test obstacle avoidance in ch11-nav2-bipedal.md
- [ ] T148 [US3] Write summary and references section in ch11-nav2-bipedal.md
- [ ] T149 [P] [US3] Create Nav2 config examples at static/code/module-3/ch11/

### Chapter 12: Synthetic Data & Perception Pipelines

- [ ] T150 [P] [US3] Create chapter file at docs/module-3-nvidia-isaac/ch12-synthetic-data.md with front matter
- [ ] T151 [US3] Write learning objectives section in ch12-synthetic-data.md
- [ ] T152 [US3] Write Replicator synthetic data section in ch12-synthetic-data.md
- [ ] T153 [US3] Write perception pipeline section in ch12-synthetic-data.md
- [ ] T154 [US3] Write model training and deployment section in ch12-synthetic-data.md
- [ ] T155 [P] [US3] Create perception pipeline diagram at static/img/diagrams/ch12-perception.svg
- [ ] T156 [US3] Write Exercise 12.1: Generate synthetic dataset with troubleshooting in ch12-synthetic-data.md
- [ ] T157 [US3] Write Exercise 12.2: Build object detection pipeline in ch12-synthetic-data.md
- [ ] T158 [US3] Write Exercise 12.3: Deploy model to Isaac Sim in ch12-synthetic-data.md
- [ ] T159 [US3] Write summary and references section in ch12-synthetic-data.md
- [ ] T160 [P] [US3] Create perception examples at static/code/module-3/ch12/

**Checkpoint**: Module 3 complete - readers can implement VSLAM navigation in Isaac Sim

---

## Phase 6: User Story 4 - Vision-Language-Action (Priority: P4)

**Goal**: Complete VLA content (4 chapters, ~24 hours reader time)

**Independent Test**: Reader can demonstrate a robot executing a natural language command through the full VLA pipeline

### Chapter 13: Voice-to-Action with Whisper + ROS 2

- [ ] T161 [P] [US4] Create chapter file at docs/module-4-vla/ch13-voice-action.md with front matter
- [ ] T162 [US4] Write learning objectives section in ch13-voice-action.md
- [ ] T163 [US4] Write Whisper integration section in ch13-voice-action.md
- [ ] T164 [US4] Write voice command interface section in ch13-voice-action.md
- [ ] T165 [US4] Write multi-modal input section in ch13-voice-action.md
- [ ] T166 [P] [US4] Create voice pipeline diagram at static/img/diagrams/ch13-voice-pipeline.svg
- [ ] T167 [US4] Write Exercise 13.1: Integrate Whisper with ROS 2 with troubleshooting in ch13-voice-action.md
- [ ] T168 [US4] Write Exercise 13.2: Build voice command parser in ch13-voice-action.md
- [ ] T169 [US4] Write Exercise 13.3: Handle multi-modal inputs in ch13-voice-action.md
- [ ] T170 [US4] Write summary and references section in ch13-voice-action.md
- [ ] T171 [P] [US4] Create Whisper examples at static/code/module-4/ch13/

### Chapter 14: LLM Cognitive Planning for Robots

- [ ] T172 [P] [US4] Create chapter file at docs/module-4-vla/ch14-llm-planning.md with front matter
- [ ] T173 [US4] Write learning objectives section in ch14-llm-planning.md
- [ ] T174 [US4] Write LLM-action space connection section in ch14-llm-planning.md
- [ ] T175 [US4] Write task decomposition section in ch14-llm-planning.md
- [ ] T176 [US4] Write planning failure handling section in ch14-llm-planning.md
- [ ] T177 [P] [US4] Create LLM planning diagram at static/img/diagrams/ch14-llm-planning.svg
- [ ] T178 [US4] Write Exercise 14.1: Connect LLM to robot actions with troubleshooting in ch14-llm-planning.md
- [ ] T179 [US4] Write Exercise 14.2: Implement task decomposer in ch14-llm-planning.md
- [ ] T180 [US4] Write Exercise 14.3: Handle planning failures gracefully in ch14-llm-planning.md
- [ ] T181 [US4] Write Exercise 14.4: Test with local LLM (Ollama) in ch14-llm-planning.md
- [ ] T182 [US4] Write summary and references section in ch14-llm-planning.md
- [ ] T183 [P] [US4] Create LLM integration examples at static/code/module-4/ch14/

### Chapter 15: Perception-Action Loops

- [ ] T184 [P] [US4] Create chapter file at docs/module-4-vla/ch15-perception-action.md with front matter
- [ ] T185 [US4] Write learning objectives section in ch15-perception-action.md
- [ ] T186 [US4] Write closed-loop control section in ch15-perception-action.md
- [ ] T187 [US4] Write reactive behaviors section in ch15-perception-action.md
- [ ] T188 [US4] Write sensor-action latency section in ch15-perception-action.md
- [ ] T189 [P] [US4] Create perception-action loop diagram at static/img/diagrams/ch15-perception-action.svg
- [ ] T190 [US4] Write Exercise 15.1: Build closed-loop controller with troubleshooting in ch15-perception-action.md
- [ ] T191 [US4] Write Exercise 15.2: Implement reactive behavior in ch15-perception-action.md
- [ ] T192 [US4] Write Exercise 15.3: Optimize latency in pipeline in ch15-perception-action.md
- [ ] T193 [US4] Write summary and references section in ch15-perception-action.md
- [ ] T194 [P] [US4] Create control examples at static/code/module-4/ch15/

### Chapter 16: Natural-Language Task Execution

- [ ] T195 [P] [US4] Create chapter file at docs/module-4-vla/ch16-nl-execution.md with front matter
- [ ] T196 [US4] Write learning objectives section in ch16-nl-execution.md
- [ ] T197 [US4] Write natural language parsing section in ch16-nl-execution.md
- [ ] T198 [US4] Write language grounding section in ch16-nl-execution.md
- [ ] T199 [US4] Write end-to-end VLA pipeline section in ch16-nl-execution.md
- [ ] T200 [P] [US4] Create VLA pipeline diagram at static/img/diagrams/ch16-vla-pipeline.svg
- [ ] T201 [US4] Write Exercise 16.1: Parse natural language commands with troubleshooting in ch16-nl-execution.md
- [ ] T202 [US4] Write Exercise 16.2: Ground language to robot actions in ch16-nl-execution.md
- [ ] T203 [US4] Write Exercise 16.3: Build complete VLA system in ch16-nl-execution.md
- [ ] T204 [US4] Write summary and references section in ch16-nl-execution.md
- [ ] T205 [P] [US4] Create VLA examples at static/code/module-4/ch16/

**Checkpoint**: Module 4 complete - readers can execute natural language commands on robot

---

## Phase 7: User Story 5 - Capstone Project (Priority: P5)

**Goal**: Complete integration project (1 chapter, ~40 hours reader time)

**Independent Test**: Reader demonstrates an autonomous humanoid that navigates, detects objects, manipulates them, and follows task plans

### Chapter 17: Full Autonomous Humanoid Project

- [ ] T206 [P] [US5] Create chapter file at docs/capstone/ch17-autonomous-humanoid.md with front matter
- [ ] T207 [US5] Write learning objectives section (4 objectives) in ch17-autonomous-humanoid.md
- [ ] T208 [US5] Write project overview and requirements section in ch17-autonomous-humanoid.md
- [ ] T209 [US5] Write system architecture section in ch17-autonomous-humanoid.md
- [ ] T210 [P] [US5] Create capstone architecture diagram at static/img/diagrams/ch17-capstone-arch.svg
- [ ] T211 [US5] Write navigation subsystem integration section in ch17-autonomous-humanoid.md
- [ ] T212 [US5] Write object detection integration section in ch17-autonomous-humanoid.md
- [ ] T213 [US5] Write manipulation subsystem section in ch17-autonomous-humanoid.md
- [ ] T214 [US5] Write task planning integration section in ch17-autonomous-humanoid.md
- [ ] T215 [P] [US5] Create subsystem integration diagram at static/img/diagrams/ch17-integration.svg
- [ ] T216 [US5] Write Exercise 17.1: Set up project workspace with troubleshooting in ch17-autonomous-humanoid.md
- [ ] T217 [US5] Write Exercise 17.2: Integrate navigation subsystem in ch17-autonomous-humanoid.md
- [ ] T218 [US5] Write Exercise 17.3: Add object detection pipeline in ch17-autonomous-humanoid.md
- [ ] T219 [US5] Write Exercise 17.4: Implement manipulation control in ch17-autonomous-humanoid.md
- [ ] T220 [US5] Write Exercise 17.5: Connect task planner to subsystems in ch17-autonomous-humanoid.md
- [ ] T221 [US5] Write testing and validation procedures section in ch17-autonomous-humanoid.md
- [ ] T222 [US5] Write summary and references section in ch17-autonomous-humanoid.md
- [ ] T223 [P] [US5] Create complete capstone code at static/code/capstone/ with README

**Checkpoint**: Capstone complete - readers have full autonomous humanoid system

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Back matter, final validation, and quality assurance

### Appendices

- [ ] T224 [P] Create Appendix A: Software Installation Guide at docs/appendices/installation.md
- [ ] T225 [P] Create Appendix B: Hardware Requirements at docs/appendices/hardware.md
- [ ] T226 [P] Create Appendix C: Troubleshooting Common Issues at docs/appendices/troubleshooting.md
- [ ] T227 Complete glossary with all technical terms at docs/appendices/glossary.md
- [ ] T228 [P] Create bibliography with all IEEE references at docs/appendices/bibliography.md

### Final Validation

- [ ] T229 Run full build with `npm run build -- --strict` to check all links
- [ ] T230 [P] Run markdownlint on all docs/*.md files
- [ ] T231 Verify all 17 chapters have required sections (objectives, exercises, summary, references)
- [ ] T232 [P] Verify all code examples are present in static/code/
- [ ] T233 [P] Verify all diagrams are present in static/img/diagrams/
- [ ] T234 Verify page count estimate meets 300-400 page target
- [ ] T235 Final review of navigation and sidebar structure
- [ ] T236 Test deployment to GitHub Pages with `npm run deploy`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 - BLOCKS all user stories
- **Phases 3-7 (User Stories)**: All depend on Phase 2 completion
  - US1 (Module 1): No story dependencies
  - US2 (Module 2): Content references Module 1 but can be written in parallel
  - US3 (Module 3): Content references Modules 1-2 but can be written in parallel
  - US4 (Module 4): Content references Module 3 but can be written in parallel
  - US5 (Capstone): Content references all modules but can be written in parallel
- **Phase 8 (Polish)**: Depends on all user story phases

### User Story Dependencies (Content References)

```text
US1 (Module 1) ─────────────────────────────────────────┐
     │                                                   │
     ├─────► US2 (Module 2) ────────────────────────────┤
     │              │                                    │
     └──────────────┼─────► US3 (Module 3) ─────────────┤
                    │              │                     │
                    └──────────────┼─────► US4 (Module 4)│
                                   │              │      │
                                   └──────────────┴──────┼─► US5 (Capstone)
                                                         │
                                                         ▼
                                                   Phase 8 (Polish)
```

### Within Each User Story

1. Create chapter file with front matter
2. Write learning objectives
3. Write content sections (can be parallelized with [P] diagrams)
4. Create diagrams (parallelizable)
5. Write exercises with troubleshooting
6. Write summary and references
7. Create code examples (parallelizable)

---

## Parallel Execution Examples

### Phase 2: Foundation (Parallel Category Configs)

```bash
# Launch module category configs in parallel:
Task: T016 [P] Create Module 2 category config
Task: T017 [P] Create Module 3 category config
Task: T018 [P] Create Module 4 category config
Task: T019 [P] Create Capstone category config
```

### Phase 3: User Story 1 (Parallel Diagrams + Code)

```bash
# While writing chapter content, parallelize diagrams:
Task: T031 [P] [US1] Create architecture diagram
Task: T036 [P] [US1] Create code examples ch01
Task: T042 [P] [US1] Create node communication diagram
Task: T048 [P] [US1] Create code examples ch02
```

### Cross-Story Parallelization

Since content writing is independent once Phase 2 completes:

```bash
# With multiple authors, parallelize entire modules:
Author A: Tasks T026-T071 (User Story 1 - Module 1)
Author B: Tasks T072-T115 (User Story 2 - Module 2)
Author C: Tasks T116-T160 (User Story 3 - Module 3)
Author D: Tasks T161-T205 (User Story 4 - Module 4)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Module 1 - ROS 2)
4. **STOP and VALIDATE**: Deploy, test navigation, verify exercises
5. This delivers a functional 4-chapter book on ROS 2 for humanoids

### Incremental Delivery

1. Setup + Foundational → Foundation ready (deployable site structure)
2. Add User Story 1 (Module 1) → Deploy → **MVP Ready!**
3. Add User Story 2 (Module 2) → Deploy → Enhanced with simulation
4. Add User Story 3 (Module 3) → Deploy → Isaac integration added
5. Add User Story 4 (Module 4) → Deploy → VLA content added
6. Add User Story 5 (Capstone) → Deploy → Complete book
7. Add Phase 8 (Polish) → Deploy → Production ready

### Parallel Team Strategy

With 4 content authors after Phase 2 completion:

- **Author A**: User Story 1 (Module 1 - ROS 2) + Appendix A
- **Author B**: User Story 2 (Module 2 - Digital Twin) + Appendix B
- **Author C**: User Story 3 (Module 3 - Isaac) + Appendix C
- **Author D**: User Story 4 (Module 4 - VLA) + Glossary
- **All authors**: User Story 5 (Capstone) collaboratively
- **DevOps**: Phase 1 setup, Phase 8 validation

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| **Total Tasks** | 236 |
| **Phase 1 (Setup)** | 9 tasks |
| **Phase 2 (Foundational)** | 16 tasks |
| **Phase 3 (US1 - ROS 2)** | 46 tasks |
| **Phase 4 (US2 - Digital Twin)** | 44 tasks |
| **Phase 5 (US3 - Isaac)** | 45 tasks |
| **Phase 6 (US4 - VLA)** | 45 tasks |
| **Phase 7 (US5 - Capstone)** | 18 tasks |
| **Phase 8 (Polish)** | 13 tasks |
| **Parallelizable Tasks** | 89 tasks (38%) |
| **Chapters** | 17 |
| **Exercises (minimum)** | 51 |
| **Diagrams** | 20+ |
| **Code Example Sets** | 17 |

---

## Notes

- [P] tasks = different files, no dependencies
- [US#] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Commit after each chapter or logical group
- Stop at any checkpoint to validate story independently
- MVP = Phase 1 + Phase 2 + Phase 3 (deployable ROS 2 module)
