import argparse
from pathlib import Path
import sys

# Support both `uv run app/main.py` and `uv run -m app.main`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chat import RAGService
from app.services.ingestion import ingest


def main():
    parser = argparse.ArgumentParser(description="Ask Clause a question or index a document.")
    parser.add_argument("--ingest", action="store_true", help="Rebuild the search collection.")
    parser.add_argument("--path", help="PDF to index (used with --ingest).")
    args = parser.parse_args()
    if args.path and not args.ingest:
        parser.error("--path requires --ingest")
    if args.ingest:
        print(f"Indexed {ingest(args.path)} chunks.")
        return

    rag_service = RAGService()
    try:
        query = input("Ask a question: ").strip()
        result = rag_service.ask(query)
        print(f"AI: {result.answer}")
    finally:
        rag_service.close()


if __name__ == "__main__":
    main()
