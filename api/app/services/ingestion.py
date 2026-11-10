import re
from pathlib import Path

# from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, Filter, FilterSelector, VectorParams

from app.core.config import API_DIR, get_embeddings, get_qdrant_client, get_settings


def load(path: Path) -> list[Document]:
    return PyPDFLoader(path).load()


def document_title(pages: list[Document], fallback: str) -> str:
    lines = [l.strip() for l in pages[0].page_content.splitlines() if l.strip()] if pages else []
    if not lines:
        return fallback
    title = lines[0]
    # PDF titles often wrap, e.g. "Handbook (2026" / "edition)".
    if title.count("(") > title.count(")") and len(lines) > 1:
        title = f"{title} {lines[1]}"
    return title


def overview(pages: list[Document], title: str, source: str) -> Document:
    """A chunk describing the whole document, so questions like
    "how many MCQs are there?" have something to retrieve."""
    text = "\n".join(p.page_content for p in pages)
    summary = f"Document overview: '{title}' has {len(pages)} pages."
    questions = set(re.findall(r"(?m)^Q(\d+)\.", text))
    if questions:
        summary += (
            f" It contains a total of {len(questions)} multiple-choice questions "
            f"(MCQs), numbered Q1 to Q{max(map(int, questions))}."
        )
    return Document(page_content=summary, metadata={"source": source, "page": 0})


def ingest(path: str | None = None) -> int:
    settings = get_settings()

    # 1. Load (RAG_DATA_PATH may list several PDFs separated by commas)
    files = [Path(p.strip()) for p in (path or settings.data_path).split(",") if p.strip()]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: list[Document] = []
    for file in files:
        if not file.is_absolute():
            file = API_DIR / file
        pages = load(file)
        title = document_title(pages, file.stem)

        # 2. Chunk, labelling each chunk with its document title so that
        # retrieval can tell the documents apart.
        for chunk in text_splitter.split_documents(pages):
            chunk.page_content = f"From: {title}\n{chunk.page_content}"
            chunks.append(chunk)
        chunks.append(overview(pages, title, str(file)))
    if not chunks:
        raise ValueError(f"No text found in {files}")

    # 3. Embed: check the model works before the old collection is deleted
    embed = get_embeddings(settings)

    actual_dim = len(embed.embed_query("dimension check"))
    if actual_dim != settings.embed_dim:
        raise ValueError(
            f"RAG_EMBED_DIM is {settings.embed_dim}, "
            f"but {settings.embed_model} returns {actual_dim} dimensions"
        )

    # 4. Store
    client = get_qdrant_client(settings)

    try:
        if client.collection_exists(settings.collection_name):
            client.delete_collection(settings.collection_name)

        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embed_dim, distance=Distance.COSINE
            ),
        )

        # On Windows, local-mode Qdrant can't remove the locked storage.sqlite
        # file, so the "deleted" collection comes back with its old points.
        # Wipe any leftovers so the index only holds the current document.
        if client.count(settings.collection_name).count:
            client.delete(
                settings.collection_name,
                points_selector=FilterSelector(filter=Filter()),
            )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embed,
        )
        vector_store.add_documents(chunks)
    finally:
        client.close()

    return len(chunks)
