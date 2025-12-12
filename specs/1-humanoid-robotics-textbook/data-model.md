# Data Model: Physical AI & Humanoid Robotics Textbook

**Feature**: `1-humanoid-robotics-textbook`
**Date**: 2025-12-12
**Status**: Complete

## Overview

This document defines the content structure and relationships for the textbook. The data model supports Docusaurus's content organization and enables future RAG chatbot integration.

---

## Entity Definitions

### E1: Book

The root entity representing the entire textbook.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| title | string | Yes | "Physical AI & Humanoid Robotics" |
| subtitle | string | Yes | "A Simulation-First Approach" |
| version | semver | Yes | Book version (e.g., "1.0.0") |
| authors | Author[] | Yes | List of contributing authors |
| modules | Module[] | Yes | Ordered list of modules |
| frontMatter | FrontMatter | Yes | Preface, prerequisites, etc. |
| backMatter | BackMatter | Yes | Appendices, glossary, index |
| metadata | BookMetadata | Yes | ISBN, publication info |

---

### E2: Module

A thematic grouping of related chapters (4 modules + capstone).

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Unique identifier (e.g., "module-1-ros2") |
| number | integer | Yes | Display order (1-5) |
| title | string | Yes | Module title |
| subtitle | string | No | Optional subtitle |
| description | text | Yes | 2-3 sentence overview |
| learningOutcomes | string[] | Yes | What readers will achieve |
| prerequisites | string[] | Yes | Required prior knowledge |
| chapters | Chapter[] | Yes | Ordered list of chapters |
| estimatedHours | integer | Yes | Completion time estimate |

**Instances**:
1. Module 1: The Robotic Nervous System (ROS 2)
2. Module 2: The Digital Twin (Gazebo & Unity)
3. Module 3: The AI-Robot Brain (NVIDIA Isaac)
4. Module 4: Vision-Language-Action (VLA)
5. Capstone: The Autonomous Humanoid

---

### E3: Chapter

A focused learning unit within a module.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Unique identifier (e.g., "ch01-intro-ros2") |
| moduleId | string | Yes | Parent module reference |
| number | integer | Yes | Chapter number within module |
| title | string | Yes | Chapter title |
| slug | string | Yes | URL-safe identifier |
| learningObjectives | string[] | Yes | 3-5 specific objectives |
| sections | Section[] | Yes | Content sections |
| exercises | Exercise[] | Yes | Hands-on activities |
| summary | text | Yes | Key takeaways |
| references | Reference[] | Yes | IEEE-formatted citations |
| estimatedMinutes | integer | Yes | Reading/completion time |

**Chapter Count by Module**:
- Module 1: 4 chapters
- Module 2: 4 chapters
- Module 3: 4 chapters
- Module 4: 4 chapters
- Capstone: 1 chapter
- **Total**: 17 chapters

---

### E4: Section

A content division within a chapter.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Unique identifier |
| chapterId | string | Yes | Parent chapter reference |
| heading | string | Yes | Section heading (H2/H3) |
| level | integer | Yes | Heading level (2 or 3) |
| content | markdown | Yes | Section body content |
| codeExamples | CodeExample[] | No | Associated code blocks |
| diagrams | Diagram[] | No | Visual aids |
| notes | Note[] | No | Tips, warnings, info boxes |

---

### E5: Exercise

A hands-on activity within a chapter.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Unique identifier (e.g., "ex-1-1") |
| chapterId | string | Yes | Parent chapter reference |
| number | integer | Yes | Exercise number in chapter |
| title | string | Yes | Exercise title |
| objective | text | Yes | What the exercise achieves |
| difficulty | enum | Yes | beginner, intermediate, advanced |
| estimatedMinutes | integer | Yes | Completion time |
| prerequisites | string[] | No | Prior exercises required |
| instructions | Step[] | Yes | Ordered implementation steps |
| expectedOutcome | text | Yes | What success looks like |
| verification | text | Yes | How to verify completion |
| troubleshooting | Tip[] | Yes | Common issues and solutions |
| solutionHint | text | No | Optional hint without full answer |

---

### E6: CodeExample

Runnable code demonstrating concepts.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Unique identifier |
| language | enum | Yes | python, cpp, yaml, xml, bash |
| title | string | Yes | Descriptive title |
| description | text | Yes | What the code demonstrates |
| code | text | Yes | The actual code |
| lineHighlights | integer[] | No | Lines to emphasize |
| expectedOutput | text | No | Console output if applicable |
| sourceCitation | Reference | No | If adapted from external source |
| runnable | boolean | Yes | Can be executed as-is |
| dependencies | string[] | No | Required packages |

---

### E7: Diagram

Visual representation of concepts.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Unique identifier |
| title | string | Yes | Figure title |
| description | text | Yes | Detailed description |
| format | enum | Yes | svg, png, mermaid |
| filePath | string | Yes | Path to asset file |
| altText | string | Yes | Accessibility description |
| caption | string | No | Figure caption |
| sourceFile | string | No | Original editable file |

---

### E8: Reference

IEEE-formatted citation.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Citation key (e.g., "[1]") |
| type | enum | Yes | article, book, website, paper |
| authors | string[] | Yes | Author names |
| title | string | Yes | Work title |
| publication | string | No | Journal/conference name |
| year | integer | Yes | Publication year |
| url | string | No | Web link if applicable |
| accessDate | date | No | For web resources |
| doi | string | No | Digital Object Identifier |

---

### E9: FrontMatter

Introductory content before modules.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| preface | markdown | Yes | Book introduction |
| audience | markdown | Yes | Who this book is for |
| prerequisites | Checklist | Yes | Required knowledge checklist |
| howToUse | markdown | Yes | Reading guide |
| conventions | markdown | Yes | Notation and formatting guide |
| acknowledgments | markdown | No | Credits and thanks |

---

### E10: BackMatter

Supplementary content after modules.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| appendices | Appendix[] | Yes | Supplementary technical content |
| glossary | GlossaryEntry[] | Yes | Term definitions |
| bibliography | Reference[] | Yes | Complete reference list |
| index | IndexEntry[] | No | Searchable index |

---

## Entity Relationships

```text
Book
├── FrontMatter
├── Module[] (1:N)
│   └── Chapter[] (1:N)
│       ├── Section[] (1:N)
│       │   ├── CodeExample[] (0:N)
│       │   ├── Diagram[] (0:N)
│       │   └── Note[] (0:N)
│       ├── Exercise[] (1:N)
│       └── Reference[] (1:N)
└── BackMatter
    ├── Appendix[]
    ├── GlossaryEntry[]
    └── Reference[] (aggregated)
```

---

## Content Volume Estimates

| Entity | Count | Avg Size | Total Pages |
|--------|-------|----------|-------------|
| Modules | 5 | - | - |
| Chapters | 17 | 18 pages | 306 pages |
| Sections | ~85 | 3 pages | (included above) |
| Exercises | ~51 | 2 pages | 102 pages |
| Code Examples | ~68 | 0.5 pages | 34 pages |
| Diagrams | ~40 | 0.5 pages | 20 pages |
| Front/Back Matter | - | - | 30 pages |
| **Total Estimate** | - | - | **350-400 pages** |

---

## Docusaurus Mapping

### Directory Structure

```text
docs/
├── intro.md                    # FrontMatter.howToUse
├── prerequisites.md            # FrontMatter.prerequisites
├── module-1-ros2/
│   ├── _category_.json         # Module metadata
│   ├── index.md                # Module overview
│   ├── ch01-introduction.md    # Chapter
│   ├── ch02-nodes-topics.md
│   ├── ch03-urdf-kinematics.md
│   └── ch04-python-agents.md
├── module-2-digital-twin/
│   └── ...
├── module-3-nvidia-isaac/
│   └── ...
├── module-4-vla/
│   └── ...
├── capstone/
│   └── index.md
└── appendices/
    ├── glossary.md
    ├── troubleshooting.md
    └── resources.md

static/
├── img/
│   ├── diagrams/
│   └── screenshots/
└── code/
    ├── module-1/
    └── ...
```

### Front Matter Schema (per chapter)

```yaml
---
id: ch01-introduction
title: Introduction to ROS 2 for Humanoids
sidebar_position: 1
tags: [ros2, basics, humanoid]
---
```

---

## Validation Rules

1. **Module**: MUST have 3-4 chapters (except capstone: 1)
2. **Chapter**: MUST have at least 3 learning objectives
3. **Chapter**: MUST have at least 1 exercise
4. **Exercise**: MUST have troubleshooting section
5. **CodeExample**: MUST specify if runnable
6. **Diagram**: MUST have altText for accessibility
7. **Reference**: MUST follow IEEE format
8. **Total**: Book MUST have 300-400 pages
