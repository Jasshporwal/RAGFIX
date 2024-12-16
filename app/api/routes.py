from fastapi import APIRouter, HTTPException
from .models import Query, Response, Feedback, FactUpload
from ..core.ragfix import ragfix
from ..core.pinecone_manager import pinecone_manager
import hashlib
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload_fact")
async def upload_fact(fact: FactUpload):
    """
    Endpoint to upload a single fact and its metadata.
    """
    try:
        embedding = await pinecone_manager.generate_embedding(fact.content)
        fact_id = hashlib.md5(fact.content.encode()).hexdigest()

        pinecone_manager.upsert_embedding(
            vector_id=fact_id,
            embedding=embedding,
            metadata={
                "content": fact.content,
                "source": fact.source,
                "category": fact.category,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        return {"status": "success", "message": "Fact uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading fact: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload fact")


@router.post("/process_folder")
async def process_folder(folder_path: str):
    """
    Endpoint to process all `.txt` files in a folder.
    """
    try:
        ragfix.process_folder(folder_path)
        return {"status": "success", "message": "Folder processed successfully"}
    except Exception as e:
        logger.error(f"Error processing folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Folder processing failed: {str(e)}")


@router.post("/query", response_model=Response)
async def process_query(query: Query):
    try:
        embedding = ragfix.generate_embedding(query.text)
        retrieved_facts = ragfix.retrieve_facts(embedding)
        
        # Verify facts before generating the response
        verified_facts = ragfix.verify_facts(retrieved_facts)
        response_text =  ragfix.generate_response(query.text, verified_facts)

        response = Response(
            query=query.text,
            response=response_text,
            sources=verified_facts,
            timestamp=datetime.utcnow().isoformat(),
        )
        print(response_text)
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")



@router.post("/feedback")
async def submit_feedback(feedback: Feedback):
    """
    Endpoint to collect user feedback on a response.
    """
    try:
        logger.info(f"Received feedback: {feedback.dict()}")
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process feedback")