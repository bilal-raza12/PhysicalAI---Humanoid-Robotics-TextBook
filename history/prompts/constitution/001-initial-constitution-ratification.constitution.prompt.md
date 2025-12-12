---
id: 001
title: Initial Constitution Ratification
stage: constitution
date: 2025-12-12
surface: agent
model: claude-opus-4-5-20251101
feature: none
branch: master
user: system
command: /sp.constitution
labels: ["constitution", "initialization", "governance", "rag-chatbot", "docusaurus"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - N/A (governance document)
---

## Prompt

/sp.constitution ## Project
**AI/Spec-Driven Book + Embedded RAG Chatbot**

## Core Principles
- **Accuracy:** All content and chatbot answers must trace back to specific book sections.
- **Clarity:** Written for advanced learners with clean structure and readability.
- **Reproducibility:** Automated build, deploy, embedding, and indexing workflows.
- **Security:** User-selected text protected; no secrets committed to the repo.

## Standards
- Book written in **Docusaurus** using **Spec-Kit Plus** structure.
- Drafts may use **Claude Code**, but all content must undergo manual review.
- Citations: **Markdown footnotes** or **APA style**.
- Chatbot stack: **FastAPI**, **Qdrant**, **Neon Postgres**, **OpenAI Agents/ChatKit**.
- Code quality: **ESLint** (JS/TS), **Black** (Python).
- API documented using **OpenAPI YAML Specification**.

## Constraints
- Minimum **40+ page** book for first public release.
- **CI pipeline** must auto-build and auto-deploy to GitHub Pages.
- Embeddings must auto-update only for modified pages.
- Infrastructure must operate within **free-tier limits**.

## Success Criteria
- Book builds successfully and deploys to GitHub Pages with CI.
- Chatbot must:
  1. Answer general book questions using indexed content.
  2. Answer based solely on **user-selected text** when provided.
- Every chatbot answer includes citations back to the relevant **page/section**.
- Automated tests cover API endpoints and retrieval flow.

## RAG Requirements
- Deterministic chunking of book pages.
- Store metadata: `doc_id`, `path`, `section`, `chunk_id`, `text`.
- Embeddings stored in **Qdrant**; metadata stored in **Neon DB**.
- Hybrid retrieval (vector + optional keyword search).
- Extractive answers required when using selected text.

## Workflow
- **Local Development:** Docusaurus dev server + local Qdrant/Postgres or Neon dev.
- **CI (main):** lint → test → build → embed → index update → deploy.
- Only build artifacts committed to `gh-pages` by CI.

## Policies
- User text is **never** stored permanently.
- Branch protection required; secrets stored in GitHub Actions.
- Minimal logs: timestamp + anonymized metadata only.

## Roles
- **Author/Editor:** Oversees content quality and structure.
- **Backend Engineer:** Implements FastAPI, embeddings, Qdrant, Neon.
- **DevOps Engineer:** Handles CI/CD pipelines and deployment.

## Response snapshot

Created the project constitution v1.0.0 for "AI/Spec-Driven Book + Embedded RAG Chatbot" with:
- 4 Core Principles: Accuracy, Clarity, Reproducibility, Security
- Technology stack: Docusaurus, FastAPI, Qdrant, Neon Postgres, OpenAI Agents
- Code quality standards: ESLint (JS/TS), Black (Python), OpenAPI specs
- Constraints: 40+ pages, free-tier infrastructure, incremental embeddings
- RAG requirements: deterministic chunking, hybrid retrieval, citation support
- CI workflow: lint → test → build → embed → index → deploy
- Privacy policies: no permanent user text storage, minimal logging
- 3 defined roles: Author/Editor, Backend Engineer, DevOps Engineer
- Governance: amendment process, semantic versioning, compliance review

## Outcome

- Impact: Established foundational governance document for the project
- Tests: N/A (governance document)
- Files: .specify/memory/constitution.md (created/updated)
- Next prompts: /sp.specify to create first feature spec, project scaffolding
- Reflection: Constitution captures all user requirements; ready for feature development

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Manual validation PASS - no unresolved placeholders
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
