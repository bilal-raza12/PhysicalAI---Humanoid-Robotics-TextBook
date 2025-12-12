<!--
  ============================================================================
  SYNC IMPACT REPORT
  ============================================================================
  Version Change: 0.0.0 → 1.0.0 (MAJOR - initial constitution ratification)

  Modified Principles:
    - N/A (initial creation)

  Added Sections:
    - Core Principles (4): Accuracy, Clarity, Reproducibility, Security
    - Standards (tech stack, tooling, documentation)
    - Constraints (scope, CI, infrastructure limits)
    - Success Criteria (build, chatbot, testing)
    - RAG Requirements (chunking, storage, retrieval)
    - Workflow (local dev, CI pipeline)
    - Policies (privacy, security, logging)
    - Roles (Author/Editor, Backend Engineer, DevOps Engineer)
    - Governance

  Removed Sections:
    - N/A (initial creation)

  Templates Requiring Updates:
    - .specify/templates/plan-template.md: ✅ No updates required (Constitution Check section is dynamic)
    - .specify/templates/spec-template.md: ✅ No updates required (generic structure)
    - .specify/templates/tasks-template.md: ✅ No updates required (generic structure)

  Follow-up TODOs:
    - None
  ============================================================================
-->

# AI/Spec-Driven Book + Embedded RAG Chatbot Constitution

## Core Principles

### I. Accuracy

All content and chatbot answers MUST trace back to specific book sections. No hallucinated or unverifiable information is permitted.

**Rationale**: The book serves as an authoritative reference. Users must trust that every chatbot response and every written statement has a verifiable source within the book content. This establishes credibility and enables fact-checking.

**Non-negotiables**:
- Every chatbot answer MUST include citations to relevant page/section
- All technical claims MUST reference source material
- AI-generated content MUST undergo manual review before publication

### II. Clarity

All content MUST be written for advanced learners with clean structure and readability.

**Rationale**: The target audience possesses foundational knowledge and expects professional-grade material. Clarity reduces friction, accelerates learning, and minimizes support burden.

**Non-negotiables**:
- Use consistent heading hierarchy throughout the book
- Maintain clean Markdown formatting standards
- Technical terminology MUST be precise and consistent
- Complex concepts MUST include examples where appropriate

### III. Reproducibility

All build, deploy, embedding, and indexing workflows MUST be automated and reproducible.

**Rationale**: Reproducibility ensures any team member can recreate the production environment, debug issues, and verify changes. It eliminates "works on my machine" problems and enables reliable CI/CD.

**Non-negotiables**:
- CI pipeline MUST auto-build and auto-deploy to GitHub Pages
- Embeddings MUST auto-update only for modified pages
- All infrastructure configuration MUST be version-controlled
- Environment setup MUST be documented and scriptable

### IV. Security

User-selected text MUST be protected; no secrets committed to the repo.

**Rationale**: User trust depends on data protection. Secrets in repositories create attack vectors and compliance risks.

**Non-negotiables**:
- User text is NEVER stored permanently
- Secrets MUST be stored in GitHub Actions (or equivalent secure store)
- No credentials, tokens, or API keys in source code
- Branch protection MUST be enabled
- Minimal logging: timestamp + anonymized metadata only

## Standards

### Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Book Framework | Docusaurus | Spec-Kit Plus structure |
| Backend API | FastAPI | RESTful endpoints |
| Vector Database | Qdrant | Embeddings storage |
| Relational Database | Neon Postgres | Metadata storage |
| AI/Chatbot | OpenAI Agents/ChatKit | RAG implementation |

### Code Quality

- **JavaScript/TypeScript**: ESLint for linting
- **Python**: Black for formatting
- **API Documentation**: OpenAPI YAML Specification

### Documentation Standards

- Drafts MAY use Claude Code for initial generation
- All content MUST undergo manual review before merge
- Citations: Markdown footnotes or APA style

## Constraints

### Scope

- Minimum **40+ pages** for first public release
- Book content MUST be structured for deterministic chunking

### Infrastructure

- All infrastructure MUST operate within **free-tier limits**
- CI pipeline MUST be GitHub Actions-based
- Deployment target: GitHub Pages

### Embedding Pipeline

- Embeddings MUST auto-update only for modified pages (incremental)
- Full re-indexing SHOULD be available as manual trigger

## Success Criteria

### Build & Deploy

- [ ] Book builds successfully via CI
- [ ] Book deploys automatically to GitHub Pages on merge to main
- [ ] Build failures block deployment

### Chatbot Functionality

- [ ] Chatbot answers general book questions using indexed content
- [ ] Chatbot answers based solely on user-selected text when provided
- [ ] Every chatbot answer includes citations to relevant page/section

### Quality Assurance

- [ ] Automated tests cover API endpoints
- [ ] Automated tests cover retrieval flow
- [ ] ESLint passes on all JS/TS files
- [ ] Black passes on all Python files

## RAG Requirements

### Chunking

- Deterministic chunking of book pages
- Chunk boundaries MUST be consistent across rebuilds
- Chunk size SHOULD be optimized for retrieval quality

### Storage Schema

Metadata fields (stored in Neon DB):
- `doc_id`: Unique document identifier
- `path`: File path to source
- `section`: Section/heading name
- `chunk_id`: Unique chunk identifier
- `text`: Chunk content (for reference/debugging)

Embeddings stored in Qdrant with matching `chunk_id` for join operations.

### Retrieval

- Hybrid retrieval: vector similarity + optional keyword search
- Extractive answers REQUIRED when using user-selected text
- Context window MUST include relevant chunk metadata for citation

## Workflow

### Local Development

```text
Docusaurus dev server + local Qdrant/Postgres (or Neon dev branch)
```

- Hot reload for book content
- Local embedding pipeline for testing
- Environment variables via `.env` (not committed)

### CI Pipeline (main branch)

```text
lint → test → build → embed → index update → deploy
```

**Stage Details**:
1. **lint**: ESLint + Black checks
2. **test**: Unit and integration tests
3. **build**: Docusaurus production build
4. **embed**: Generate embeddings for modified pages
5. **index update**: Update Qdrant + Neon with new/changed chunks
6. **deploy**: Push to GitHub Pages

Only build artifacts committed to `gh-pages` branch by CI.

## Policies

### Data Privacy

- User-selected text is processed in-memory only
- User text is NEVER persisted to database or logs
- Session data expires after request completion

### Logging

- Timestamp + anonymized metadata only
- No PII in logs
- No user-submitted text in logs

### Repository Security

- Branch protection REQUIRED on main
- Secrets stored in GitHub Actions only
- No hardcoded credentials in source

## Roles

### Author/Editor

**Responsibilities**:
- Oversees content quality and structure
- Reviews AI-generated drafts
- Ensures accuracy and clarity standards
- Final approval on published content

### Backend Engineer

**Responsibilities**:
- Implements FastAPI endpoints
- Manages embeddings pipeline
- Integrates Qdrant and Neon
- Implements RAG retrieval logic

### DevOps Engineer

**Responsibilities**:
- Maintains CI/CD pipelines
- Manages GitHub Actions workflows
- Handles deployment automation
- Monitors infrastructure within free-tier limits

## Governance

### Amendment Process

1. Propose change via pull request to constitution.md
2. Document rationale and impact assessment
3. Require approval from at least one representative of each role
4. Update version number according to semantic versioning
5. Update LAST_AMENDED_DATE to current date

### Versioning Policy

- **MAJOR**: Backward-incompatible principle changes or removals
- **MINOR**: New principles/sections added or materially expanded
- **PATCH**: Clarifications, wording fixes, non-semantic refinements

### Compliance Review

- All PRs MUST verify compliance with constitution principles
- CI checks SHOULD enforce measurable standards (linting, tests)
- Architectural decisions SHOULD be documented in ADRs

### Conflict Resolution

Constitution principles supersede all other project documentation. When conflicts arise:
1. Constitution takes precedence
2. Escalate to role representatives for interpretation
3. Amend constitution if clarification needed

**Version**: 1.0.0 | **Ratified**: 2025-12-12 | **Last Amended**: 2025-12-12
