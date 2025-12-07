"""Configuration manager for the multi-agent workflow.

This module centralizes all configuration for LLM and embedding services,
allowing easy switching between different providers via environment variables.
"""

import os


class Config:
    """Configuration manager for the multi-agent workflow."""

    # LLM Configuration
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://69.48.159.10:30000/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # Embedding Configuration
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://69.48.159.10:30001/v1")
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "not-needed")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Nexus-bge-m3-opensearch-embeddings")

    # Model token limits (adjusted for 64k context length local LLM)
    # Using ~80% of context for output to leave room for input
    MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS_DEFAULT", "50000"))
    MAX_TOKENS_WRITER = int(os.getenv("MAX_TOKENS_WRITER", "55000"))

    # Context length for the local LLM
    LLM_CONTEXT_LENGTH = int(os.getenv("LLM_CONTEXT_LENGTH", "64000"))


def get_model_config() -> dict:
    """Get the base configuration for initializing chat models.

    Returns:
        Dictionary with model configuration parameters.
    """
    return {
        "model": f"openai:{Config.LLM_MODEL}",
        "base_url": Config.LLM_BASE_URL,
        "api_key": Config.LLM_API_KEY,
        "temperature": Config.LLM_TEMPERATURE,
    }


def get_embedding_config() -> dict:
    """Get the configuration for embedding models.

    Returns:
        Dictionary with embedding configuration parameters.
    """
    return {
        "model": Config.EMBEDDING_MODEL,
        "base_url": Config.EMBEDDING_BASE_URL,
        "api_key": Config.EMBEDDING_API_KEY,
    }
