# Clause

Run commands from the `api` directory. Configure `.env` using `.env.example`.
Start Ollama and install the models configured in `.env` (defaults below):

```powershell
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

Index the PDF before the first chat. Ingestion replaces the configured collection:

```powershell
uv run app/main.py --ingest
uv run app/main.py
```

Use `--ingest --path path/to/document.pdf` to index another PDF. Relative document
and local Qdrant paths resolve from `api`. The module entry point also works:
`uv run -m app.main`.
