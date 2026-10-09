"""Embedding model selection."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.config import settings


def build_embeddings() -> Embeddings:
    if settings.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.resolve_api_key(),
        )

    if settings.provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.resolve_api_key(),
        )

    raise ValueError(f"Unknown provider: {settings.provider!r}")
