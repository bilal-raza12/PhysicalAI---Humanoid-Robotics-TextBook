# Quickstart: Backend-Frontend Integration

**Feature**: 001-backend-frontend-integration
**Date**: 2025-12-25

## Prerequisites

- Python 3.11+
- Node.js 18+
- uv package manager (Python)
- npm (Node.js)
- Environment variables configured in `.env`

## Environment Setup

### Backend (.env)

```bash
# Copy from backend/.env.example
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=...

# FastAPI settings
BACKEND_PORT=8000
LOG_LEVEL=INFO
```

### Frontend (docusaurus.config.ts)

```typescript
// Add to customFields
customFields: {
  backendUrl: process.env.BACKEND_URL || 'http://localhost:8000',
},
```

## Quick Start

### 1. Start Backend

```bash
cd backend

# Install dependencies
uv sync

# Start FastAPI server
uv run uvicorn main:app --reload --port 8000
```

Backend will be available at `http://localhost:8000`

### 2. Start Frontend

```bash
# From repository root

# Install dependencies (including ChatKit)
npm install

# Start Docusaurus dev server
npm start
```

Frontend will be available at `http://localhost:3000`

### 3. Verify Integration

1. Open `http://localhost:3000` in browser
2. Look for ChatKit widget on any textbook page
3. Ask a question: "What is ROS 2?"
4. Verify:
   - Loading indicator appears
   - Answer displays with citations
   - Source links are clickable

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chatkit/session` | POST | Get ChatKit session token |
| `/api/ask` | POST | Ask a question |
| `/api/search` | POST | Search knowledge base |
| `/api/respond` | POST | Alias for /ask |
| `/api/store` | POST | Store embeddings |
| `/api/embed` | POST | Generate embeddings |

## Testing Endpoints

### Test Session Token

```bash
curl -X POST http://localhost:8000/api/chatkit/session
```

Expected response:
```json
{"client_secret": "ck_sess_..."}
```

### Test Query

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ROS 2?"}'
```

Expected response:
```json
{
  "answer": "ROS 2 is...",
  "citations": [...],
  "grounded": true,
  "refused": false,
  "metadata": {...}
}
```

### Test Search

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "robot operating system", "k": 5}'
```

## Concurrent Development

Run both servers simultaneously:

```bash
# Terminal 1: Backend
cd backend && uv run uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
npm start
```

Or use a process manager like `concurrently`:

```bash
npm install -g concurrently

concurrently \
  "cd backend && uv run uvicorn main:app --reload" \
  "npm start"
```

## Troubleshooting

### CORS Errors

If you see CORS errors in browser console:
- Verify backend is running on port 8000
- Check CORS middleware allows `localhost:3000`
- Clear browser cache and retry

### ChatKit Not Loading

- Check browser console for JavaScript errors
- Verify `@openai/chatkit-react` is installed
- Check backend URL in docusaurus.config.ts

### Agent Timeout

- Default timeout is 30 seconds
- Large queries may take longer
- Check backend logs for errors

### Empty Responses

- Verify Qdrant collection has data
- Check API keys in .env
- Test `/api/search` endpoint directly

## Development Workflow

1. **Backend changes**: Auto-reload via `--reload` flag
2. **Frontend changes**: Hot reload via Docusaurus
3. **API changes**: Update `contracts/api.yaml`
4. **Schema changes**: Update `data-model.md`

## Next Steps

After quickstart:
1. Run `/sp.tasks` to generate implementation tasks
2. Implement `backend/routes.py`
3. Create `src/components/ChatWidget/`
4. Test end-to-end flow
