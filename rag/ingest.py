from pathlib import Path
import sys

from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "playground" / "code" / "shop_handbook.pdf"

CHAT_PROVIDER = "ollama"
CHAT_MODEL = "gemma3:1b"

EMBED_PROVIDER = "ollama"
EMBED_MODEL = "nomic-embed-text:latest"
EMBED_DIM = 768

TOP_K = 3

COLLECTION_NAME = "shop_handbook"
QDRANT_PATH = BASE_DIR / "qdrant_data"

SYSTEM_PROMPT = (
    "You are a helpful Clause Shop assistant. "
    "Answer only from the provided context. "
    "If the answer is not in the context, say you don't know."
)


# STEP 1: LOAD DOCUMENTS

def main():
    pdf_path = (
        Path(sys.argv[1]).expanduser().resolve()
        if len(sys.argv) > 1
        else DATA_PATH
    )

    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata={
                "source": str(pdf_path),
                "page": index,
            },
        )
        for index, page in enumerate(reader.pages)
    ]

    print(f"Loaded {len(pages)} pages from {pdf_path.name}.")
    preview = next((page.page_content.strip() for page in pages if page.page_content.strip()), "")
    if preview:
        print(f"Text preview:\n{preview[:500]}")
    else:
        print("No text was extracted. This PDF may require OCR.")


if __name__ == "__main__":
    main()
