## RAGFix - Hallucination Mitigation System for LLMs

RAGFix is a FastAPI-based system designed to reduce hallucinations in Large Language Models (LLMs) using Retrieval-Augmented Generation (RAG). It provides a robust fact-checking mechanism by maintaining a verified knowledge base in Pinecone and using it to ground LLM responses.

## Features

- 🔍 Fact-based response generation using RAG
- 💾 Persistent storage of verified facts using Pinecone
- 🚀 High-performance vector similarity search
- 📊 Confidence scoring for responses
- 🔄 Query result caching
- 📝 User feedback collection
- 🔒 Fact verification system
- 📈 Response quality metrics

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── .env.example
├── main.py
├── config/
│   └── settings.py
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── models.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ragfix.py
│   │   └── pinecone_manager.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py

```
 
## Setup

1. Clone the repository:
   ```
   git clone 
   cd 
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   uvicorn main:app --reload
   ```

The server will start running on `http://localhost:8000`.

## Configuration

The application uses environment variables to configure the Pinecone API key and index name. Set the following environment variables:
```
PINECONE_API_KEY = YOUR_PINECONE_API_KEY
PINECONE_INDEX_NAME = YOUR_PINECONE_INDEX_NAME
OPENAI_API_KEY = YOUR_OPENAI_API_KEY. 
```

## API ENDPOINTS

### Upload a New Fact
```http
POST /upload_fact
Upload a new verified fact to the knowledge base.
```
```json
{
    "content": "The Earth orbits the Sun at an average distance of 93 million miles.",
    "source": "NASA",
    "category": "astronomy"
}
```

### Query the System
```http
POST /query
Query the system with a question to get a fact-based response.
```
```json
{
    "text": "How far is Earth from the Sun?",
    "context": {}
}
```

### Submit Feedback
```http
POST /feedback
Submit feedback for a response received from the system.
```
```json
{
    "response_id": "12345",
    "rating": 5,
    "comments": "Very accurate response"
}
```
## Response Format

The system returns responses in the following format:
```json
{
    "query": "How far is Earth from the Sun?",
    "response": "According to NASA, the Earth orbits the Sun at an average distance of 93 million miles.",
    "sources": [
        {
            "content": "The Earth orbits the Sun at an average distance of 93 million miles.",
            "source": "NASA",
            "chunk_index":"1",
            "confidence": 0.95
        }
    ],
    "timestamp": "2024-11-19T10:30:00Z"
}
```

## Contributing

Contributions are welcome! Please submit a pull request with your changes.



