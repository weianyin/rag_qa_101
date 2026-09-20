from unittest.mock import MagicMock

from rag_qa.generate import generate_answer
from rag_qa.models import RetrievedChunk


def _make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id="doc.txt", source_path="/doc.txt", text=text, distance=0.1
    )


def test_no_chunks_skips_api_call_and_reports_not_grounded():
    client = MagicMock()

    answer = generate_answer("what is x?", [], client, model="claude-sonnet-5")

    assert answer.grounded is False
    assert answer.citations == []
    client.messages.create.assert_not_called()


def test_cited_chunks_are_extracted_from_response():
    client = MagicMock()
    text_block = MagicMock(type="text", text="X is a widget [1].")
    client.messages.create.return_value = MagicMock(content=[text_block])

    chunks = [_make_chunk("c1", "a widget is a small device")]
    answer = generate_answer("what is x?", chunks, client, model="claude-sonnet-5")

    assert answer.grounded is True
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "c1"


def test_uncited_response_is_not_grounded():
    client = MagicMock()
    text_block = MagicMock(type="text", text="I don't have enough information to answer that.")
    client.messages.create.return_value = MagicMock(content=[text_block])

    chunks = [_make_chunk("c1", "unrelated text")]
    answer = generate_answer("what is x?", chunks, client, model="claude-sonnet-5")

    assert answer.grounded is False
    assert answer.citations == []
