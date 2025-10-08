"""Main FastAPI application and server startup."""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="RAG Papers API", description="Data-Aware RAG for PDF Research Papers")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "RAG Papers API is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def run():
    """Run the development server."""
    uvicorn.run("rag_papers.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()