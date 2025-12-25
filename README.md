# Physical AI & Humanoid Robotics Textbook

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Node](https://img.shields.io/badge/node-%3E%3D18.0-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)
![Docusaurus](https://img.shields.io/badge/docusaurus-3.6-green.svg)

**A Simulation-First Approach to Building Intelligent Humanoid Systems**

[Live Demo](https://bilal-raza12.github.io/PhysicalAI---Humanoid-Robotics-TextBook/) | [Documentation](#course-modules) | [Getting Started](#quick-start)

</div>

---

## Overview

A comprehensive, open-source textbook covering Physical AI, Embodied Intelligence, and Humanoid Robotics. This project combines a beautiful Docusaurus-powered documentation site with an AI-powered RAG (Retrieval Augmented Generation) chatbot that can answer questions about the textbook content.

### Key Features

- **4 Comprehensive Modules** covering ROS 2, Digital Twins, NVIDIA Isaac, and Vision-Language-Action systems
- **18+ Chapters** with hands-on exercises and code examples
- **AI Chat Assistant** powered by RAG for intelligent Q&A about textbook content
- **Modern UI** with animated hero section, module cards, and responsive design
- **Capstone Project** for building a complete autonomous humanoid robot

---

## Course Modules

| Module | Title | Description | Chapters |
|--------|-------|-------------|----------|
| **1** | ROS 2 Fundamentals | Master the Robot Operating System 2 - nodes, topics, services, actions, URDF modeling | 5 |
| **2** | Digital Twin & Simulation | Build virtual replicas in Gazebo, Unity, and create accurate physics simulations | 4 |
| **3** | NVIDIA Isaac Platform | Leverage GPU-accelerated simulation, synthetic data generation, and Sim2Real transfer | 4 |
| **4** | Vision-Language-Action | Integrate VLA models for robots that understand and execute natural language commands | 4 |

---

## Technology Stack

### Frontend (Docusaurus)

| Technology | Purpose |
|------------|---------|
| [Docusaurus 3.6](https://docusaurus.io/) | Static site generator for documentation |
| [React 18](https://react.dev/) | UI component library |
| [TypeScript](https://www.typescriptlang.org/) | Type-safe JavaScript |
| [MDX](https://mdxjs.com/) | Markdown with JSX components |

### Backend (FastAPI + RAG)

| Technology | Purpose |
|------------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | High-performance Python API framework |
| [OpenAI Agents](https://platform.openai.com/) | AI-powered question answering |
| [Cohere](https://cohere.com/) | Text embeddings for semantic search |
| [Qdrant](https://qdrant.tech/) | Vector database for document retrieval |
| [BeautifulSoup4](https://beautiful-soup-4.readthedocs.io/) | HTML parsing for content extraction |

---

## Quick Start

### Prerequisites

- **Node.js** >= 18.0
- **Python** >= 3.13
- **uv** (Python package manager) - [Install uv](https://docs.astral.sh/uv/)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/bilal-raza12/PhysicalAI---Humanoid-Robotics-TextBook.git
cd PhysicalAI---Humanoid-Robotics-TextBook
```

### 2. Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The frontend will be available at `http://localhost:3000`

### 3. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create environment file
cp .env.example .env

# Edit .env with your API keys
# COHERE_API_KEY=your-cohere-api-key
# QDRANT_URL=https://your-cluster.qdrant.io
# QDRANT_API_KEY=your-qdrant-api-key

# Install dependencies and run
uv sync
uv run uvicorn main:app --reload
```

The backend API will be available at `http://localhost:8000`

---

## Project Structure

```
Physical_ai_&_Humanoid_Robotics/
├── docs/                          # Textbook content (MDX/Markdown)
│   ├── module-1-ros2/             # ROS 2 Fundamentals
│   ├── module-2-digital-twin/     # Digital Twin & Simulation
│   ├── module-3-nvidia-isaac/     # NVIDIA Isaac Platform
│   ├── module-4-vla/              # Vision-Language-Action
│   ├── capstone/                  # Capstone Project
│   └── appendices/                # Installation, Troubleshooting, Glossary
├── src/
│   ├── components/
│   │   └── ChatWidget/            # AI Chat Assistant component
│   ├── css/
│   │   └── custom.css             # Global styles
│   └── pages/
│       ├── index.tsx              # Homepage
│       └── index.module.css       # Homepage styles
├── backend/
│   ├── main.py                    # FastAPI application & RAG pipeline
│   ├── routes.py                  # API endpoints
│   ├── models.py                  # Pydantic models
│   ├── chunks.json                # Processed text chunks
│   ├── embeddings.json            # Vector embeddings
│   └── pyproject.toml             # Python dependencies
├── static/                        # Static assets (images, icons)
├── .github/
│   └── workflows/
│       └── deploy.yml             # GitHub Pages deployment
├── docusaurus.config.ts           # Docusaurus configuration
├── sidebars.ts                    # Documentation sidebar structure
└── package.json                   # Node.js dependencies
```

---

## API Endpoints

### Backend API (`/api`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ask` | POST | Ask a question about the textbook |
| `/api/health` | GET | Health check endpoint |
| `/docs` | GET | Interactive API documentation (Swagger) |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ROS 2?"}'
```

### Example Response

```json
{
  "answer": "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software...",
  "citations": [
    {
      "source_url": "/docs/module-1-ros2/ch01-intro-ros2",
      "chapter": "Introduction to ROS 2",
      "section": "What is ROS 2?",
      "score": 0.95
    }
  ],
  "grounded": true,
  "refused": false,
  "metadata": {
    "retrieval_time_ms": 45,
    "generation_time_ms": 1200,
    "total_time_ms": 1245
  }
}
```

---

## Development

### Available Scripts

#### Frontend

```bash
npm start          # Start development server
npm run build      # Build for production
npm run serve      # Serve production build locally
npm run lint       # Lint markdown files
npm run lint:fix   # Fix markdown lint issues
```

#### Backend

```bash
uv run uvicorn main:app --reload   # Start dev server with hot reload
uv run python main.py              # Run embedding pipeline
uv run black .                     # Format Python code
```

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Cohere API (for embeddings)
COHERE_API_KEY=your-cohere-api-key

# Qdrant Cloud (for vector storage)
QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Optional
LOG_LEVEL=INFO
```

---

## Deployment

### GitHub Pages (Frontend)

The frontend automatically deploys to GitHub Pages on push to `main` branch via GitHub Actions.

**Live URL**: https://bilal-raza12.github.io/PhysicalAI---Humanoid-Robotics-TextBook/

### Backend Deployment

The backend can be deployed to any platform that supports Python:

- **Railway**: `railway up`
- **Render**: Connect GitHub repo
- **Fly.io**: `fly deploy`
- **Docker**: Use provided Dockerfile

---

## Chat Widget Features

The AI-powered chat assistant provides:

- **Contextual Q&A** - Ask questions about any topic in the textbook
- **Citation Links** - Responses include links to source chapters
- **Suggestion Chips** - Quick-start questions for common topics
- **Modern UI** - Avatars, typing indicators, and smooth animations
- **Grounded Responses** - Answers are based only on textbook content

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Commit Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **ROS 2 Community** for the robotics framework
- **NVIDIA** for Isaac Sim and Omniverse
- **OpenAI** for AI capabilities
- **Docusaurus** for the documentation platform
- **Cohere & Qdrant** for RAG infrastructure

---

## Contact

- **Author**: Bilal Raza
- **GitHub**: [@bilal-raza12](https://github.com/bilal-raza12)
- **Project Link**: [PhysicalAI---Humanoid-Robotics-TextBook](https://github.com/bilal-raza12/PhysicalAI---Humanoid-Robotics-TextBook)

---

<div align="center">

**Built with passion for robotics education**

</div>
