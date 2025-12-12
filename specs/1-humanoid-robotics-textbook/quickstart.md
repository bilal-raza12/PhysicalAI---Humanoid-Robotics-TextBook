# Quickstart: Physical AI & Humanoid Robotics Textbook

**Feature**: `1-humanoid-robotics-textbook`
**Date**: 2025-12-12

This guide helps you get started with developing the textbook content.

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Node.js | 18.x LTS | Docusaurus runtime |
| npm/yarn | Latest | Package management |
| Git | 2.x+ | Version control |
| VS Code | Latest | Recommended editor |

### Recommended Extensions (VS Code)

- Markdown All in One
- markdownlint
- Draw.io Integration (diagrams)
- YAML
- Prettier

---

## Quick Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd Physical_ai_&_Humanoid_Robotics

# Switch to feature branch
git checkout 1-humanoid-robotics-textbook

# Install dependencies
npm install
```

### 2. Start Development Server

```bash
npm run start
```

The site will be available at `http://localhost:3000`.

### 3. Verify Setup

1. Navigate to `http://localhost:3000`
2. Check that the sidebar loads
3. Edit `docs/intro.md` and verify hot reload works

---

## Project Structure

```text
Physical_ai_&_Humanoid_Robotics/
├── docs/                      # Book content (Markdown)
│   ├── intro.md               # Landing page
│   ├── prerequisites.md       # Prerequisites checklist
│   ├── module-1-ros2/         # Module 1 chapters
│   │   ├── _category_.json    # Sidebar config
│   │   ├── index.md           # Module overview
│   │   └── ch01-*.md          # Chapters
│   ├── module-2-digital-twin/
│   ├── module-3-nvidia-isaac/
│   ├── module-4-vla/
│   ├── capstone/
│   └── appendices/
├── static/                    # Static assets
│   ├── img/
│   │   └── diagrams/          # Chapter diagrams
│   └── code/                  # Downloadable code
├── src/                       # Docusaurus customization
│   └── css/                   # Custom styles
├── specs/                     # Feature specifications
│   └── 1-humanoid-robotics-textbook/
├── docusaurus.config.js       # Site configuration
├── sidebars.js                # Navigation structure
└── package.json
```

---

## Writing Workflow

### Create a New Chapter

1. **Create the file**:
   ```bash
   touch docs/module-1-ros2/ch02-nodes-topics.md
   ```

2. **Add front matter**:
   ```yaml
   ---
   id: ch02-nodes-topics
   title: Nodes, Topics, Services & Launch Systems
   sidebar_position: 2
   tags: [ros2, nodes, topics, services]
   ---
   ```

3. **Follow the chapter template** from `specs/1-humanoid-robotics-textbook/contracts/chapter-template.yaml`

4. **Add to sidebar** (if not auto-detected):
   ```javascript
   // sidebars.js
   module.exports = {
     docs: [
       // ...
     ],
   };
   ```

### Add a Diagram

1. **Create diagram** using draw.io or preferred tool
2. **Export** to SVG (preferred) or PNG
3. **Save** to `static/img/diagrams/[chapter-id]-[name].svg`
4. **Reference** in Markdown:
   ```markdown
   ![ROS 2 Node Architecture](../../../static/img/diagrams/ch01-node-architecture.svg)
   *Figure 1.1: ROS 2 Node Architecture*
   ```

### Add a Code Example

```markdown
```python title="publisher_node.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        # ... rest of code
```

**Expected Output:**
```text
[INFO] Publishing: "Hello World: 0"
[INFO] Publishing: "Hello World: 1"
```
```

### Add an Exercise

Follow the structure from `contracts/chapter-template.yaml`:

```markdown
### Exercise 1.1: Create Your First ROS 2 Node

**Objective**: Create and run a minimal ROS 2 publisher node.

**Difficulty**: Beginner

**Estimated Time**: 20 minutes

#### Instructions

1. Create a new package:
   ```bash
   ros2 pkg create --build-type ament_python my_first_node
   ```

2. [Additional steps...]

#### Expected Outcome

You should see messages being published to the `/topic` topic.

#### Verification

```bash
ros2 topic echo /topic
```

#### Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| "Package not found" | Workspace not sourced | Run `source install/setup.bash` |
```

---

## Content Guidelines

### Style Guide

1. **Voice**: Second person ("you will learn...")
2. **Tense**: Present tense for instructions
3. **Technical terms**: Define on first use, add to glossary
4. **Code comments**: Explain non-obvious logic

### Citation Format (IEEE)

```markdown
According to recent research [1], humanoid robots require...

## References

[1] A. Koenig et al., "Humanoid Robot Control," *IEEE Robotics*, vol. 15, pp. 100-110, 2024.
```

### Accessibility

- All images must have alt text
- Use semantic headings (H2 → H3 → H4)
- Provide text descriptions for diagrams

---

## Building and Testing

### Local Build

```bash
npm run build
```

### Check for Issues

```bash
# Check for broken links
npm run build -- --strict

# Lint Markdown
npx markdownlint docs/**/*.md
```

### Preview Production Build

```bash
npm run serve
```

---

## Common Tasks

### Update Module Overview

Edit `docs/module-X/index.md` to update the module landing page.

### Reorder Chapters

Update `sidebar_position` in each chapter's front matter.

### Add Glossary Term

Edit `docs/appendices/glossary.md`:

```markdown
**ROS 2**: Robot Operating System 2, a set of software libraries and tools
for building robot applications.
```

---

## Deployment

Deployment is automated via GitHub Actions on merge to `main`:

1. Build triggers on push to `main`
2. Docusaurus builds the static site
3. Site deploys to GitHub Pages
4. Available at: `https://<org>.github.io/<repo>`

### Manual Deployment (if needed)

```bash
USE_SSH=true npm run deploy
```

---

## Getting Help

- **Spec**: `specs/1-humanoid-robotics-textbook/spec.md`
- **Data Model**: `specs/1-humanoid-robotics-textbook/data-model.md`
- **Contracts**: `specs/1-humanoid-robotics-textbook/contracts/`
- **Constitution**: `.specify/memory/constitution.md`
