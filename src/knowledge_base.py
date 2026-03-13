"""
KnowledgeBase
ChromaDB vector store lifecycle — create, load, rebuild.
Pattern mirrors InsightForge's knowledge_base.py exactly.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src import config


class KnowledgeBase:
    """
    Manages a persistent ChromaDB vector store for real estate market documents.

    Usage
    -----
    kb = KnowledgeBase(api_key)
    vector_store = kb.get_or_create(documents)
    retriever    = kb.get_retriever(k=5)
    """

    def __init__(
        self,
        api_key: str,
        embedding_model: str = config.EMBEDDING_MODEL,
        persist_directory: str | Path = config.CHROMA_DIR,
    ) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key,
        )
        self._persist_dir = Path(persist_directory)
        self.vector_store: Chroma | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """Embed documents and persist to disk. Overwrites any existing store."""
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            persist_directory=str(self._persist_dir),
        )
        return self.vector_store

    def load_existing(self) -> Chroma:
        """Load a previously persisted vector store from disk."""
        self.vector_store = Chroma(
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
        )
        return self.vector_store

    def get_or_create(self, documents: List[Document]) -> Chroma:
        """Load from disk if a store exists; otherwise create and persist."""
        if self._persist_dir.exists() and any(self._persist_dir.iterdir()):
            return self.load_existing()
        return self.create_from_documents(documents)

    def rebuild(self, documents: List[Document]) -> Chroma:
        """Delete the existing store and rebuild from scratch."""
        if self._persist_dir.exists():
            shutil.rmtree(self._persist_dir)
        return self.create_from_documents(documents)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_retriever(self, k: int = config.TOP_K_RETRIEVAL):
        """Return a standard LangChain retriever backed by this vector store."""
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialised. Call get_or_create() first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})
