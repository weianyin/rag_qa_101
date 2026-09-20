"""Stage 3: turn text into embedding vectors.

Uses a local sentence-transformers model rather than a hosted embeddings API,
so the pipeline needs only ANTHROPIC_API_KEY (for generation) to run
end-to-end — no second API key, no per-embedding cost. Swap the model in
config.py if retrieval quality warrants it during Week 2 comparisons.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer


class Embedder:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _load_model(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    # Cached so repeated Embedder(...) construction (e.g. per-request in the
    # API layer) doesn't reload weights from disk every time.
    return SentenceTransformer(model_name)
