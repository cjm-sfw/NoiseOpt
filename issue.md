# Development Issues Log

## Critical Discrepancies Found
1. API Endpoint Mismatches:
   - Current: /api/judge/queue vs Required: /api/prompts/{id}/next_comparison (Dev.md 5.4)
   - Current: /api/judge/submit vs Required: /api/judgments (Dev.md 5.4)
   - Missing DrawBench endpoints from Dev.md section 5.4

2. Data Model Issues:
   - Current implementation uses rating system instead of winner/loser comparison
   - Missing prompt_id in judgment submissions
   - No support for user_id tracking

3. Frontend/Backend Inconsistencies:
   - judge.js expects different response format than app.py provides
   - Missing error handling for API version mismatches

## Immediate Action Items
1. Backend Updates:
   - Align API endpoints with Dev.md specification
   - Update data models to support pairwise comparison
   - Implement DrawBench integration

2. Frontend Updates:
   - Rewrite judge.js to use pairwise comparison UI
   - Update API calls to match new endpoints
   - Add proper error handling

3. System Updates:
   - Add version checking between frontend/backend
   - Implement proper logging for Celery tasks

## Completed Tasks
- [Initial setup complete] 2025-08-20
- Reviewed all current implementations against Dev.md
- Identified all critical discrepancies

## Next Steps
1. Create migration plan for API changes
2. Update documentation to reflect all changes
3. Implement versioned API endpoints
