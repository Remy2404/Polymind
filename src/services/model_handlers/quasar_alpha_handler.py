import json
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from services.model_handlers import ModelHandler
from services.openrouter_api import OpenRouterAPI


class QuasarAlphaHandler(ModelHandler):
    """Handler for Quasar Alpha model via OpenRouter API."""

    def __init__(self, openrouter_api: OpenRouterAPI):
        """Initialize the Quasar Alpha model handler."""
        self.openrouter_api = openrouter_api

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Generate a text response using the Quasar Alpha model via OpenRouter."""
        response = await asyncio.wait_for(
            self.openrouter_api.generate_response(
                prompt=prompt,
                context=context,
                model="openrouter/quasar-alpha",
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=300.0,
        )
        return response

    def get_system_message(self) -> str:
        """Get the system message for the Quasar Alpha model."""
        return "You are Quasar Alpha, an advanced AI assistant that helps users with tasks and answers questions helpfully, accurately, and ethically."

    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name."""
        return "🌀 Quasar Alpha"
