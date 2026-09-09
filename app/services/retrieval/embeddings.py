import time
import logfire
from langchain_google_genai import GoogleGenAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768

_active_model = None
_model_type:str | None = None

def _probe_gemini():
    # health check for Gemini API key and endpoint
    try:
        model = GoogleGenerativeAIEmbeddings(
            model= "models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY
        )
        model.embed_query("probe")
        logfire.info("gemini embeddings are ready  (gemini-embedding-2-preview).")
        return model
    except Exception as e:
        logfire.
        
    pass

def _load_fallback():
    pass

def _init():
    return

def get_embeddin_dim():
    # returen vector dimension of the current active model call after _init().
    pass

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using the active model.
    """
    pass

def _embed_query(query: str) -> list[float]:
    return

