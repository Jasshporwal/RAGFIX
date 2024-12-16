
from pinecone import Pinecone, ServerlessSpec
import openai
import logging
import time
import re
from config.settings import settings
from fastapi import HTTPException
from datetime import datetime

logger = logging.getLogger(__name__)


class PineconeManager:
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = self._validate_index_name(settings.PINECONE_INDEX_NAME)
        self.dimension = settings.EMBEDDING_DIMENSION

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        self.create_index_if_not_exists()

    def _validate_index_name(self, index_name: str) -> str:
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME environment variable is not set")

        logger.info(f"Validating index name: '{index_name}'")

        index_name = index_name.lower()
        index_name = re.sub(r"[^a-z0-9-]", "-", index_name)
        index_name = re.sub(r"-+", "-", index_name)
        index_name = index_name.strip("-")

        if not index_name:
            raise ValueError(
                "Index name must contain at least one alphanumeric character"
            )

        if len(index_name) > 45:
            index_name = index_name[:45].rstrip("-")

        return index_name
    
    async def generate_embedding(self, text: str):
        try:
            response = openai.embeddings.create(
                input=text, model=settings.EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise HTTPException(status_code=500, detail="Embedding generation failed")

    def create_index_if_not_exists(self):
        try:
            existing_indexes = self.pc.list_indexes().names()

            if self.index_name not in existing_indexes:
                logger.info(f"Creating new index: '{self.index_name}'")

                params = {
                    "name": self.index_name,
                    "dimension": self.dimension,
                    "metric": "cosine",
                    "spec": ServerlessSpec(cloud="aws", region="us-east-1"),
                }

                self.pc.create_index(**params)

                while not self.pc.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)

                logger.info(f"Index '{self.index_name}' created successfully")
            else:
                logger.info(f"Index '{self.index_name}' already exists")

            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to create index: {str(e)}"
            )

    def upsert_embedding(self, vector_id: str, embedding: list, metadata: dict):
        """
        Upsert a single embedding with metadata to Pinecone.
        """
        try:
            self.index.upsert(
                vectors=[
                    (
                        vector_id,
                        embedding,
                        metadata,
                    )
                ]
            )
            logger.info(f"Successfully upserted vector ID: {vector_id}")
        except Exception as e:
            logger.error(f"Error upserting embedding: {str(e)}")
            raise HTTPException(status_code=500, detail="Upsert failed")

    def vector_exists(self, vector_id: str) -> bool:
        """
        Check if a vector with the given ID already exists in Pinecone.
        """
        try:
            result = self.index.fetch(ids=[vector_id])
            return vector_id in result.get("vectors", {})
        except Exception as e:
            logger.error(f"Error checking vector existence: {str(e)}")
            return False

    def query_embeddings(self, query_embedding: list, top_k: int = 10):
        """
        Query the Pinecone index using an embedding and return the top matches.
        """
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
            )
            if not results.matches:
                logger.warning("No matches found in Pinecone.")
                return []

            logger.info(f"Query returned {len(results.matches)} matches.")
            return [
                {
                    "content": match.metadata.get("content", ""),
                    "source": match.metadata.get("source", "Unknown"),
                    "chunk_index": match.metadata.get("chunk_index", -1),
                    "confidence": match.score,
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error(f"Error querying embeddings: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to query Pinecone index"
            )


pinecone_manager = PineconeManager()