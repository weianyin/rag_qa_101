# Project: RAG Q&A System with Evaluation Harness

## What this project is
A retrieval-augmented generation (RAG) question-answering system over a chosen
document set, built to demonstrate LLM engineering skills for job interviews:
retrieval quality, evaluation rigor, iteration on failure cases, and basic
production concerns (latency, cost, logging).

This is a portfolio/learning project, built over ~3 weeks of steady evening work.

## Tech stack
- Language: Python 3.11+
- API layer: FastAPI
- Vector store: Chroma (local, no external dependency) — swap to a hosted
  store (Pinecone/Weaviate) only if there's a specific reason to
- Embeddings: local `sentence-transformers` (`all-MiniLM-L6-v2`) — keeps the
  pipeline to a single external API dependency (Claude), no second API key
  or per-embedding cost. Revisit in Week 2 if retrieval quality warrants a
  hosted model.
- LLM: Claude API (Sonnet) for generation and for eval judging
- Testing: pytest
- Dependency management: uv

## Coding conventions
- Type hints on all function signatures
- Docstrings on public functions (Google style)
- Config (model names, chunk size, top-k) lives in a single `config.py` or
  `.env`, not hardcoded inline
- No notebook-only logic that isn't also in a script — notebooks are for
  exploration, not the final pipeline
- Every pipeline stage (ingest, chunk, embed, retrieve, generate) should be
  independently testable

## How to run things
- Install deps: `uv sync`
- Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`
- Build the index (after dropping documents into `data/raw/`): `uv run python scripts/build_index.py`
- Run the API: `uv run uvicorn rag_qa.api:app --reload`
- Run tests: `uv run pytest`
- Run the eval harness: `<fill in once built — Week 2>`

## Project phases (do not skip ahead)
1. **Week 1 — Core pipeline**: document ingestion, chunking, embeddings,
   vector store, retrieval + generation loop, minimal FastAPI endpoint.
2. **Week 2 — Evaluation**: build a 20–50 question eval set with reference
   answers. Measure retrieval precision/recall and answer quality (LLM-as-judge
   using Claude). Try at least 2–3 different chunking/retrieval strategies and
   compare with actual numbers, not vibes.
3. **Week 3 — Production concerns**: add per-query logging (latency, tokens,
   cost), handle "no relevant docs found" and ambiguous-question cases, basic
   guardrails, write up architecture + tradeoffs in README.

## What "done" looks like for each phase
- Phase 1: I can send a question to the API and get a grounded answer with
  cited source chunks.
- Phase 2: I have a table comparing retrieval strategies with real metrics,
  and I can explain *why* the winning strategy won.
- Phase 3: I can explain what breaks at 10x scale and what I'd change.

## Things to avoid
- Don't add features not tied to a phase above without updating this file first
- Don't skip writing the eval set to "save time" — it's the point of the project
- Don't hide retrieval failures behind a good-sounding LLM answer — surface them
