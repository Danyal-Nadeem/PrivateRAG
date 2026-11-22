"""FastAPI application for PrivateRAG.

Endpoints
---------
POST /login
    Body: {"username": "employee", "password": "..."}
    Returns: {"token": "...", "role": "employee", "username": "employee"}

    Credentials are verified against the server-side user store
    (src/api/users.py). The role is assigned by the server from the
    user record — clients can never self-assign a privilege level.

POST /ask
    Header: Authorization: Bearer <token>
    Body: {"query": "..."}     ← "role" field in body is ignored
    Returns: {"query", "role", "answer", "retrieved_documents"}

GET /health
    Returns: {"status", "qdrant_persistent", "total_chunks", "llm_provider",
              "ner_model_loaded"}
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from src.api.auth import create_token, get_current_role
from src.api.users import (
    AuthenticationError,
    LoginRateLimitedError,
    UserStoreError,
    authenticate,
)
from src.llm import get_llm_provider
from src.ner.infer import NerModelUnavailableError
from src.ner.masking import sanitize_text
from src.retrieval.qdrant_service import get_collection_info, search

TOP_K = 3


# ---------------------------------------------------------------------------
# Lifespan — warm up embedder on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on a missing/weak JWT signing key — a known or short key
    # would allow anyone to forge admin tokens.
    from src.api.auth import _get_secret
    _get_secret()

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
        "POST /login authenticates credentials; roles are server-assigned."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: explicit frontend origins only. A wildcard ("*") combined with
# credentials is invalid per the CORS spec and unsafe for a security product.
# Override with CORS_ORIGINS="https://app.example.com,http://localhost:5173".
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
] or _DEFAULT_CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str
    # Any "role" field sent by the client is ignored — roles are
    # server-assigned from the user store after authentication.
    model_config = {"extra": "ignore"}


class LoginResponse(BaseModel):
    token: str
    role: str
    username: str


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
    ner_model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    """Authenticate credentials and issue a JWT with the server-assigned role.

    The role comes exclusively from the user store record — never from
    client input. Failed logins return a generic 401 (no user enumeration);
    repeated failures lock the username out temporarily (429); a missing
    user store fails closed (503).
    """
    try:
        user = authenticate(body.username, body.password)
    except LoginRateLimitedError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except UserStoreError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication service is not configured (no user store). "
                "Create data/users.json or set ALLOW_DEMO_ACCOUNTS=1."
            ),
        )
    except AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )
    token = create_token(user["role"], subject=user["username"])
    return LoginResponse(token=token, role=user["role"], username=user["username"])


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, role: str = Depends(get_current_role)) -> AskResponse:
    """Answer *query* using RBAC-filtered retrieval.

    The role is extracted exclusively from the JWT in the Authorization header.
    Any "role" field in the request body is ignored. The generated answer is
    sanitized via NER masking before being returned to the client.

    In fail-closed mode (NER_STRICT=1) an unavailable DLP model yields a
    deliberate 503 — never an unsanitized answer, never a generic 500.
    """
    hits = search(body.query, role=role, top_k=TOP_K)

    llm = get_llm_provider()
    context_chunks = [h["text"] for h in hits]
    raw_answer = llm.generate(prompt=body.query, context_chunks=context_chunks)
    try:
        masked_answer = sanitize_text(raw_answer)
    except NerModelUnavailableError:
        import logging
        logging.getLogger("privaterag.ner").error(
            "/ask: refusing to answer — DLP model unavailable in NER_STRICT mode."
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Sanitization service unavailable (fail-closed mode). "
                "The answer cannot be released without DLP screening. "
                "Restore the NER model checkpoint or disable NER_STRICT."
            ),
        )

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
        answer=masked_answer,
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

    from src.ner.infer import model_available
    ner_loaded = model_available()
    if not ner_loaded:
        import logging
        logging.getLogger("privaterag.ner").warning(
            "/health: DLP model NOT loaded — answers are regex-sanitized only "
            "(PERSON/ADDRESS unmasked)."
        )

    return HealthResponse(
        status="ok",
        qdrant_persistent=qdrant_ok,
        total_chunks=total_chunks,
        llm_provider=os.getenv("LLM_PROVIDER", "extractive"),
        ner_model_loaded=ner_loaded,
    )
