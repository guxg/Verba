import os
import json
from typing import List
import io

import aiohttp
from wasabi import msg

from goldenverba.components.interfaces import Embedding
from goldenverba.components.types import InputConfig
from goldenverba.components.util import get_environment, get_token

from langchain_community.embeddings import DashScopeEmbeddings

class DashscopeEmbedder(Embedding):
    """DashscopeEmbedder for Verba."""

    def __init__(self):
        super().__init__()
        self.name = "Dashscope"
        self.description = "Vectorizes documents and queries using Dashscope"

        # Fetch available models
        api_key = get_token("DASHSCOPE_API_KEY")
        models = self.get_models(api_key)

        # Set up configuration
        self.config = {
            "Model": InputConfig(
                type="dropdown",
                value="text-embedding-v3",
                description="Select an Dashscope Embedding Model",
                values=models,
            )
        }

        # Add API Key and URL configs if not set in environment
        if api_key is None:
            self.config["API Key"] = InputConfig(
                type="password",
                value="",
                description="Dashscope API Key (or set DASHSCOPE_API_KEY env var)",
                values=[],
            )
        

    async def vectorize(self, config: dict, content: List[str]) -> List[List[float]]:
        """Vectorize the input content using Dashscope API."""
        model = config.get("Model", {"value": "text-embedding-v3"}).value
        api_key = get_environment(
            config, "API Key", "DASHSCOPE_API_KEY", "No Dashscope API Key found"
        )

        embeddings = DashScopeEmbeddings(
            model=model, 
            dashscope_api_key=api_key
        )

        try:
            return embeddings.embed_documents(content)
        except Exception as e:
            msg.fail(f"Unexpected error: {type(e).__name__} - {str(e)}")
            raise

    @staticmethod
    def get_models(token: str) -> List[str]:
        """Fetch available embedding models from Dashscope API."""
        return [
            "text-embedding-v3",
            "text-embedding-v2"
        ]
