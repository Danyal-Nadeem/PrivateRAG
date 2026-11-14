"""FastAPI application for PrivateRAG.

Endpoints
---------
POST /login-role
    Body: {"role": "employee"}
    Returns: {"token": "...", "role": "employee"}

    NOTE: /login-role accepts a role without a password to simulate SSO
    for prototype purposes. This is NOT real authentication — in production
    replace this with your identity provider.

POST /ask
    Header: Authorization: Bearer <token>
    Body: {"query": "..."}     ← "role" field in body is ignored
    Returns: {"query", "role", "answer", "retrieved_documents"}

GET /health
    Returns: {"status", "qdrant_persistent", "total_chunks", "llm_provider"}
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from pydantic import BaseModel

load_dotenv()

from src.api.auth import VALID_ROLES, create_token, get_current_role
from src.llm import get_llm_provider
from src.retrieval.qdrant_service import get_collection_info, search

TOP_K = 3


# ---------------------------------------------------------------------------
# Lifespan — warm up embedder on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load the embedding model so the first /ask call is not slow.
    # Skipped when TESTING=1 to keep unit tests fast.
    if os.getenv("TESTING") != "1":
        from src.retrieval import embedder
        embedder.embed(["warm-up"])
    yield


app = FastAPI(
    title="PrivateRAG",
    description=(
        "Role-based access controlled RAG system. "
        "POST /login-role simulates SSO for prototype purposes — not real auth."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    role: str


class LoginResponse(BaseModel):
    token: str
    role: str


class AskRequest(BaseModel):
    query: str
    # NOTE: any "role" field sent in the body is intentionally NOT declared
    # here — Pydantic will ignore unknown fields (extra="ignore" is the
    # default for FastAPI models), so it can never influence RBAC.

    model_config = {"extra": "ignore"}


class RetrievedDoc(BaseModel):
    doc_id: str
    access_level: str
    chunk_index: int
    title: str
    score: float


class AskResponse(BaseModel):
    query: str
    role: str
    answer: str
    retrieved_documents: list[RetrievedDoc]


class HealthResponse(BaseModel):
    status: str
    qdrant_persistent: bool
    total_chunks: int
    llm_provider: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/login-role", response_model=LoginResponse)
def login_role(body: LoginRequest) -> LoginResponse:
    """Issue a JWT for *role*.

    This endpoint simulates SSO for the prototype and is NOT real
    authentication — no password is required.
    """
    if body.role not in VALID_ROLES:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Unknown role {body.role!r}. Valid roles: {sorted(VALID_ROLES)}",
        )
    token = create_token(body.role)
    return LoginResponse(token=token, role=body.role)


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, role: str = Depends(get_current_role)) -> AskResponse:
    """Answer *query* using RBAC-filtered retrieval.

    The role is extracted exclusively from the JWT in the Authorization header.
    Any "role" field in the request body is ignored.
    """
    hits = search(body.query, role=role, top_k=TOP_K)

    llm = get_llm_provider()
    context_chunks = [h["text"] for h in hits]
    answer = llm.generate(prompt=body.query, context_chunks=context_chunks)

    retrieved = [
        RetrievedDoc(
            doc_id=h["doc_id"],
            access_level=h["access_level"],
            chunk_index=h["chunk_index"],
            title=h["title"],
            score=round(h["score"], 4),
        )
        for h in hits
    ]

    return AskResponse(
        query=body.query,
        role=role,
        answer=answer,
        retrieved_documents=retrieved,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return system health information."""
    try:
        info = get_collection_info()
        total_chunks = info.get("points_count", 0) or 0
        qdrant_ok = True
    except Exception:
        total_chunks = -1
        qdrant_ok = False

    return HealthResponse(
        status="ok",
        qdrant_persistent=qdrant_ok,
        total_chunks=total_chunks,
        llm_provider=os.getenv("LLM_PROVIDER", "extractive"),
    )
