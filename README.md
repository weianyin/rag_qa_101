# RAG Q&A System — Project Spec

## Why this project
Built to demonstrate LLM/GenAI engineering skills for interviews: not just
"can call an LLM API," but retrieval quality, evaluation rigor, handling
failure cases, and production awareness (latency, cost, monitoring).

## Timeline: ~3 weeks, steady evenings

### Week 1 — Core pipeline
- [ ] Choose a document set you know well (company public docs, a domain
      you're interested in, or a set of technical papers/books)
- [ ] Build ingestion + chunking (start simple: fixed-size chunks with overlap)
- [ ] Embed chunks and store in a vector DB (Chroma to start)
- [ ] Build retrieval (top-k similarity search)
- [ ] Build generation: retrieved chunks + question → Claude → answer with
      citations back to source chunks
- [ ] Wrap in a minimal FastAPI endpoint (`POST /ask`)
- **Done when:** you can ask a real question and get a grounded, cited answer.

### Week 2 — Evaluation
- [ ] Write 20–50 question/reference-answer pairs covering: easy factual
      questions, questions requiring multiple chunks, and questions with
      no good answer in the docs (to test refusal behavior)
- [ ] Build a retrieval metric: precision/recall @k against a labeled
      "which chunks are relevant" set
- [ ] Build an answer-quality metric: use Claude as an LLM judge, scoring
      against the reference answers on a rubric (correctness, groundedness,
      completeness)
- [ ] Run the eval against at least 2–3 variations: different chunk sizes,
      different top-k, with/without reranking
- [ ] Produce a comparison table with real numbers and a short writeup of
      why the winning configuration won
- **Done when:** you have data-backed evidence for your retrieval choices,
  not just intuition.

### Week 3 — Production concerns
- [ ] Add structured logging per query: latency breakdown (retrieval vs.
      generation), token counts, estimated cost
- [ ] Handle edge cases explicitly: no relevant chunks found → say so, don't
      hallucinate; ambiguous question → ask for clarification or hedge
- [ ] Add basic guardrails (e.g. refuse to answer clearly out-of-scope questions)
- [ ] Write the architecture section below and a "tradeoffs & what I'd do at
      scale" section
- **Done when:** you can walk an interviewer through what breaks at 10x
  traffic and what you'd change.

## Architecture
*(fill in once built — a simple diagram: ingestion → chunking → embedding →
vector store; query → retrieval → generation → response, with logging
attached at each stage)*

## Tradeoffs & what I'd do differently at scale
*(fill in during Week 3 — this section is often exactly what gets asked
about in interviews)*

## Key numbers to be ready to cite in interviews
- Retrieval precision/recall for your chosen configuration
- Answer quality score and how it's measured
- p50/p95 latency
- Rough cost per query
