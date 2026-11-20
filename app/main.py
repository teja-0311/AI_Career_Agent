"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.auth import router as auth_router
from app.api.resume import router as resume_router
from app.api.jobs import router as jobs_router
from app.api.agent import router as agent_router

app = FastAPI(
    title="AI Career Agent API",
    description=(
        "Production-grade backend for AI Job Matching, Semantic Retrieval, "
        "Candidate Profiling, and Explainable Hybrid Scoring built with FastAPI, LangGraph, and Gemini."
    ),
    version="2.0.0",
)

# CORS configuration from settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origin_list(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"], summary="Service Health Check")
async def health_check():
    return {
        "status": "healthy",
        "service": "AI Career Agent",
        "version": "2.0.0",
        "gemini_active": settings.is_gemini_active(),
        "vector_backend": settings.vector_store_backend,
        "embedding_backend": settings.embedding_backend,
    }


# Register routers
app.include_router(auth_router)
app.include_router(resume_router)
app.include_router(jobs_router)
app.include_router(agent_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
