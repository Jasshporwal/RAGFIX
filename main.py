from fastapi import FastAPI
from app.api.routes import router
from app.core.pinecone_manager import pinecone_manager
from app.utils.logger import setup_logging
import uvicorn

# Setup logging
setup_logging()

# Initialize FastAPI app
app = FastAPI(title="RAGFix", description="Hallucination Mitigation System for LLMs")

# Initialize Pinecone
index = pinecone_manager.create_index_if_not_exists()

# Include routers
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
