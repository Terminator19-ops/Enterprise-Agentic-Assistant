import os
import sys
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embeddings import embed_texts, get_embeddin_dim
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.office import parse_office
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")\

PROCESSED_DATA_DIR = "processed_data"

qdrant_client = QdrantClient(
    url=settings.QDRANT_CLUSTER_ENDPOINT,
    api_key=settings.QDRANT_API_KEY,
)

def save_processed_locally(data: dict, source_type: str, filename: str):
    """Save parsed chunk data locally as json file in /processed_data/"""
    folder = os.path.join(PROCESSED_DATA_DIR,source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder,f"{filename}.json")
    with open(dest, "w", encoding = "utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest

def process_file(file_path: str, file_name: str, source_type: str):
    """parse > chunk > save > embed > index in qdrant"""
    with logfire.span("Processing File", file=file_name, source=source_type):
        try:
            ext = file_name.lower().rsplit(".",1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext == "html":
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx", "doc","ppt"):
                full_text = parse_office(file_path)
            else:
                logfire.warning(F"skipping unsupported file type : {file_name}")
                return
            if not full_text or not full_text.strip():
                logfire.warning(f"no text extracted from {file_name} - skipping")
                return

            chunks = chunk_text(full_text)
            if not chunks:
                return

            processed_data = {
                "filename": file_name,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, file_name)
            logfire.info(f"saved process data at {local_path}")

            with logfire.span("vectorizing & indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id = str(uuid.uuid4()),
                        vector = vector,
                        payload = {
                            "text": chunk,
                            "source": file_name,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

                qdrant_client.upsert(
                    collection_name = settings.QDRANT_COLLECTION,
                    points = points
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {file_name}")

        except Exception as e:
            logfire.error(f"Failed to process {file_name}: {e}")

    return full_text

def process_directory(dir_path: str, source_type: str):
    """processes every file in directory"""
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,f))]
    logfire.info(f"Found {len(files)} files in {dir_path}")
    for filename in files:
        process_file(os.path.join(dir_path, filename), filename, source_type)

    
def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base dir, map subfolders to source type and ingest all the documents.
    Pass --wipe to draw and recreate the qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory = base_dir):
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embeddin_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}"
                f"({dim}-dim, Cosine)"
            )

        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found in processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)

if __name__ == "__main__":
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]
    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completedhttps://logfire-us.pydantic.dev/auth/device/3A0iJXCA6EYEN4X88um9tykKGdI9U1nlyK2K8XdkQd0")