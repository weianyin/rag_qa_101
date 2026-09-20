"""Stage 6: generate a grounded, cited answer from retrieved chunks.

The prompt instructs Claude to answer only from the provided chunks and to
say so plainly when they don't contain an answer — retrieval failures
should surface as an honest "not found," not get papered over by a
plausible-sounding hallucination.
"""

import re

import anthropic

from rag_qa.models import Answer, Citation, RetrievedChunk

_SYSTEM_PROMPT = """You are a Q&A assistant that answers strictly from the provided \
source excerpts. Rules:
- Only use information in the excerpts below. Do not use outside knowledge.
- Every factual claim must be traceable to an excerpt. Cite excerpts inline \
using their bracketed number, e.g. [1].
- If the excerpts do not contain enough information to answer, say so \
explicitly instead of guessing.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i + 1}] (source: {c.source_path})\n{c.text}" for i, c in enumerate(chunks))


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int = 1024,
) -> Answer:
    """Call Claude with the retrieved chunks as grounding context.

    Args:
        question: The user's question.
        retrieved_chunks: Chunks from the retrieval stage, most relevant first.
        client: An initialized Anthropic client.
        model: Claude model name to use for generation.
        max_tokens: Max tokens in the generated answer.

    Returns:
        An `Answer` with the generated text and the citations that were
        actually referenced. If no chunks were retrieved, returns a
        not-grounded answer without calling the API.
    """
    if not retrieved_chunks:
        return Answer(
            question=question,
            text="I couldn't find any relevant information in the document set to answer this.",
            citations=[],
            grounded=False,
        )

    context = _format_context(retrieved_chunks)
    user_message = f"Source excerpts:\n\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer_text = "".join(block.text for block in response.content if block.type == "text")

    cited_indices = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    citations = [
        Citation(source_path=c.source_path, chunk_id=c.chunk_id, text=c.text)
        for i, c in enumerate(retrieved_chunks)
        if (i + 1) in cited_indices
    ]

    return Answer(
        question=question,
        text=answer_text,
        citations=citations,
        grounded=bool(citations),
    )
