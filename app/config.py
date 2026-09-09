import os
from dotenv import load_dotenv

load_dotenv()

class Settings: 
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = "enterprise-rag"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL")
    GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL")


settings = Settings()
