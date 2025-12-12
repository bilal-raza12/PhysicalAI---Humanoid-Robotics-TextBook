---
id: prerequisites
title: Prerequisites Checklist
sidebar_position: 2
---

# Prerequisites Checklist

Before starting this textbook, assess your readiness using the checklist below. Each section includes resources if you need to strengthen your background.

## Programming Skills

### Python Proficiency (Required)

You should be comfortable with:

- [ ] **Variables and data types**: strings, integers, floats, booleans, lists, dictionaries
- [ ] **Control flow**: if/else statements, for loops, while loops
- [ ] **Functions**: defining functions, parameters, return values, scope
- [ ] **Classes and objects**: creating classes, methods, inheritance basics
- [ ] **Modules and packages**: importing, pip/conda package management
- [ ] **File I/O**: reading and writing files
- [ ] **Error handling**: try/except blocks

**Self-Assessment**: Can you write a Python class that reads a configuration file and processes data?

**Resources if needed**:
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)

### Command Line Basics (Required)

You should be comfortable with:

- [ ] **Navigation**: `cd`, `ls`, `pwd`, `mkdir`
- [ ] **File operations**: `cp`, `mv`, `rm`, `cat`, `nano`/`vim`
- [ ] **Permissions**: `chmod`, `sudo`
- [ ] **Environment variables**: `export`, `.bashrc`
- [ ] **Package management**: `apt` on Ubuntu

**Self-Assessment**: Can you navigate to a directory, create a Python file, make it executable, and run it?

---

## Operating System

### Ubuntu 22.04 LTS (Required)

This book targets Ubuntu 22.04 (Jammy Jellyfish). You need:

- [ ] **Native Ubuntu 22.04 installation** OR
- [ ] **Windows Subsystem for Linux 2 (WSL2)** with Ubuntu 22.04 OR
- [ ] **Virtual machine** with Ubuntu 22.04 (VMware, VirtualBox)

:::caution WSL2 Users
GPU passthrough for Isaac Sim requires additional configuration. See [Appendix B: Hardware Requirements](/docs/appendices/hardware) for WSL2-specific setup.
:::

**Self-Assessment**: Can you open a terminal, update packages with `sudo apt update`, and install software?

---

## Hardware Requirements

### Minimum Configuration

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 8 cores (Intel i7/AMD Ryzen 7) | 12+ cores |
| **RAM** | 16 GB | 32 GB |
| **GPU** | NVIDIA GTX 1070 | NVIDIA RTX 3060+ |
| **Storage** | 100 GB SSD free | 200 GB SSD free |
| **Display** | 1920x1080 | 2560x1440 |

### GPU Requirements

- [ ] **NVIDIA GPU** with CUDA support (required for Isaac Sim)
- [ ] **CUDA 11.8+** installed
- [ ] **GPU driver 525+** installed

:::tip Cloud Alternatives
If you lack GPU hardware, cloud options are available:
- **AWS RoboMaker** for Gazebo simulation
- **NVIDIA NGC** for Isaac Sim cloud instances
- See [Appendix B](/docs/appendices/hardware) for detailed cloud setup
:::

---

## Mathematics Background

### Linear Algebra (Helpful)

Understanding these concepts will help with robot kinematics:

- [ ] **Vectors**: addition, scaling, dot product
- [ ] **Matrices**: multiplication, transpose, inverse
- [ ] **Transformations**: rotation matrices, homogeneous coordinates

**Self-Assessment**: Can you multiply two 3x3 matrices by hand?

### Calculus (Helpful)

Basic calculus aids understanding of:

- [ ] **Derivatives**: rates of change, gradients
- [ ] **Integration**: area under curves (for trajectory planning)

---

## Software Installation Checklist

Before starting Module 1, install the following:

### Core Requirements

- [ ] **Ubuntu 22.04 LTS** (see above)
- [ ] **Python 3.10+** (usually pre-installed)
- [ ] **Git** (`sudo apt install git`)

### Module 1 Requirements

- [ ] **ROS 2 Humble** - Installation covered in Chapter 1
- [ ] **colcon** build tool
- [ ] **VS Code** (recommended) with ROS extension

### Module 2 Requirements

- [ ] **Gazebo Fortress** - Installation covered in Chapter 5
- [ ] **Unity 2022.3 LTS** - Installation covered in Chapter 8

### Module 3 Requirements

- [ ] **NVIDIA Isaac Sim 2023.1+** - Installation covered in Chapter 9
- [ ] **NVIDIA Omniverse Launcher**
- [ ] **Docker** (optional, for containerized deployment)

### Module 4 Requirements

- [ ] **OpenAI API key** (or local LLM setup)
- [ ] **Whisper** speech recognition

---

## Readiness Assessment

### Ready to Start

If you checked **all required items** above, you're ready for Module 1!

### Need Preparation

If you're missing prerequisites:

1. **Programming gaps**: Complete a Python tutorial (2-4 weeks)
2. **Linux unfamiliar**: Take a Linux basics course (1-2 weeks)
3. **Hardware limitations**: Set up cloud alternatives before Module 3

### Estimated Preparation Time

| Starting Level | Preparation Time |
|----------------|------------------|
| Experienced programmer, Linux user | 0 weeks |
| Programmer, new to Linux | 1-2 weeks |
| Some programming, new to Linux | 3-4 weeks |
| Beginner | 6-8 weeks |

---

## Next Steps

Once you've verified your prerequisites:

1. Review the [Conventions Used](/docs/conventions) in this book
2. Begin [Module 1: The Robotic Nervous System](/docs/module-1-ros2)

:::info Installation Help
Detailed installation instructions for all software are provided in [Appendix A: Software Installation Guide](/docs/appendices/installation).
:::
