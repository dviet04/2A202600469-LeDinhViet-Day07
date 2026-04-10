from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    # "data/python_intro.txt",
    # "data/vector_store_notes.md",
    # "data/rag_system_design.md",
    # "data/customer_support_playbook.txt",
    # "data/chunking_experiment_report.md",
    # "data/vi_retrieval_notes.md",
    "data/file1.txt",
    "data/file2.txt",
    "data/file3.txt",
    "data/file4.txt",
    "data/file5.txt",
    "data/file6.txt",
    "data/file1.txt"
]

def openai_llm(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a helpful QA assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


# def demo_llm(prompt: str) -> str:
#     """A simple mock LLM for manual RAG testing."""
#     preview = prompt[:400].replace("\n", " ")
#     return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(
    question: str | None = None,
    sample_files: list[str] | None = None
) -> int:

    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")

    docs = load_documents_from_files(files)
    if not docs:
        print("No valid documents found.")
        return 1

    print(f"\nLoaded {len(docs)} documents")

    # Load env
    load_dotenv(override=False)

    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()

    if provider == "local":
        try:
            embedder = LocalEmbedder(
                model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
            )
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(
                model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
            )
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(
        collection_name="manual_test_store",
        embedding_fn=embedder,
    )

    store.add_documents(docs)

    print(f"Stored {store.get_collection_size()} documents")

    print("\n=== Search Test ===")
    search_results = store.search(query, top_k=3)

    for i, r in enumerate(search_results, 1):
        print(f"{i}. score={r['score']:.3f} source={r['metadata'].get('source')}")
        print(f"   preview: {r['content'][:100].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent ===")

    agent = KnowledgeBaseAgent(
        store=store,
        llm_fn=openai_llm,  
    )

    print(f"Question: {query}")

    answer = agent.answer(query, top_k=3)

    print("\nAnswer:\n")
    print(answer)

    return 0

def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())