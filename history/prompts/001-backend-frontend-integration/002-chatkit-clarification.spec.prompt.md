---
id: 002
title: ChatKit UI Library Clarification
stage: spec
date: 2025-12-25
surface: agent
model: claude-opus-4-5-20251101
feature: 001-backend-frontend-integration
branch: 001-backend-frontend-integration
user: developer
command: /sp.clarify
labels: [clarification, chatkit, openai, frontend, ui-library]
links:
  spec: specs/001-backend-frontend-integration/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-backend-frontend-integration/spec.md
  - specs/001-backend-frontend-integration/checklists/requirements.md
tests:
  - null
---

## Prompt

/sp.clarify plz use chatkit for the frontend of the chatbot
/sp.clarify plz use the docs from mcp context 7 which is connected now for chatkit ui use that and modify the spec
/sp.clarify https://github.com/openai/chatkit-js this is the documentation for chatkit ui now modify the spec.md file according to the requirement

## Response snapshot

Updated specification with OpenAI ChatKit (`@openai/chatkit-react`) as the frontend chat UI library:

**Clarification Applied**:
- Q: Which UI library for frontend chat? → A: OpenAI ChatKit from https://github.com/openai/chatkit-js

**Changes Made**:
1. Added `## Clarifications` section with session log
2. Updated all user stories to reference ChatKit widget and features
3. Added new backend requirement FR-001: `/api/chatkit/session` endpoint for token generation
4. Updated frontend requirements (FR-010 to FR-017) for ChatKit-specific integration:
   - ChatKit component and useChatKit hook
   - getClientSecret API callback
   - Response streaming display
5. Added ChatKitSession to Key Entities
6. Updated Constraints to include ChatKit as required UI library
7. Added SC-007: ChatKit widget renders on all textbook pages
8. Updated Dependencies to include `@openai/chatkit-react`

**ChatKit Features Leveraged**:
- Built-in loading/typing indicators
- Response streaming support
- Native message scrolling
- Session token authentication

## Outcome

- ✅ Impact: Spec clarified with explicit ChatKit requirement, frontend requirements updated
- 🧪 Tests: No tests - specification clarification phase
- 📁 Files: 2 files updated (spec.md, checklists/requirements.md)
- 🔁 Next prompts: `/sp.plan` to begin implementation planning
- 🧠 Reflection: ChatKit provides batteries-included chat UI, reducing custom frontend work

## Evaluation notes (flywheel)

- Failure modes observed: Initial confusion about which "ChatKit" library - resolved by user providing GitHub URL
- Graders run and results (PASS/FAIL): Quality checklist PASS (16/16)
- Prompt variant (if applicable): User-initiated clarification with external documentation reference
- Next experiment (smallest change to try): Proceed to planning phase with ChatKit-specific architecture
