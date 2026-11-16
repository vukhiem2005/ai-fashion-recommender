
# ELAN V5 - AI Agent Instructions

## Project Context

ELAN is a full-stack fashion recommendation application.

Technology stack:
- Backend: Python, FastAPI, SQLite, Scikit-learn
- Frontend: Next.js, TypeScript
- Current recommender: V1 TF-IDF + Cosine Similarity
- Active catalog: 328 products, 343 variants
- Deployment: Vercel frontend, Render backend

V1 is stable and currently powers the live website.

## Development Scope

Work only on branch `feat/recommendation-v5`.

The current goal is to research and develop:
1. Synthetic user-item interaction datasets
2. Collaborative Filtering
3. Association Rule Mining
4. Multimodal Embeddings
5. Hybrid Recommendation and offline evaluation

Do not implement all approaches at once.

## Safety Rules

- Preserve existing functionality and repository history.
- Do not modify the current V1 recommendation engine without approval.
- Do not modify the production database or overwrite catalog snapshots.
- Do not delete or replace existing product images.
- Do not modify production environment variables or secrets.
- Do not deploy to Vercel or Render.
- Do not run git push, merge, reset, clean, or force-push.
- Do not commit changes without explicit user approval.
- Do not install dependencies or download large models without approval.

## AI Development Workflow

Before making changes:

1. Inspect relevant source code and architecture.
2. Explain the proposed implementation in English.
3. List the exact files to create or modify.
4. Identify possible risks to existing functionality.
5. Wait for explicit approval before editing files.

After approved changes:

1. Run focused checks relevant to the changes.
2. Review Git diff and check for accidental modifications.
3. Report test results, limitations, and outstanding issues.
4. Request approval before committing.

## Recommendation Research Rules

- Keep new recommendation experiments isolated from V1.
- Use synthetic data only for simulation and technical validation.
- Label synthetic datasets clearly.
- Use reproducible random seeds and documented generation rules.
- Avoid training/test leakage.
- Compare new models against the existing baseline.
- Evaluate with relevant metrics such as Precision@5 and nDCG@5.
- Do not claim real-world recommendation improvements from synthetic data.
- Prefer efficient, low-cost solutions suitable for local development.

## Communication

-  Respond in English.
- Write all technical explanations, plans, reports, code comments, and documentation in English.
- Explain technical decisions clearly.
- Work incrementally, one development stage at a time.
- Never modify unrelated files.
- Ask for approval before making substantial changes.
