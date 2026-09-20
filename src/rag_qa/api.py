"""Minimal FastAPI wrapper around the query pipeline (`POST /ask`)."""

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from rag_qa.pipeline import answer_question

load_dotenv()

app = FastAPI(title="RAG Q&A 101")


class AskRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    source_path: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[CitationResponse]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = answer_question(request.question)
    return AskResponse(
        answer=result.text,
        grounded=result.grounded,
        citations=[
            CitationResponse(source_path=c.source_path, chunk_id=c.chunk_id)
            for c in result.citations
        ],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
