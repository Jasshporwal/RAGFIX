from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str
    OPENAI_API_KEY: str
    EMBEDDING_DIMENSION: int = 1536
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.3
    MODEL_NAME: str = "gpt-4"
    GROQ_API_KEY: str 
    EMBEDDING_MODEL: str = "text-embedding-ada-002"

    class Config:
        env_file = ".env"


settings = Settings()