import asyncio
from typing import List, Dict, Any, Optional
from services.model_handlers import ModelHandler
from services.gemini_api import GeminiAPI


class GeminiHandler(ModelHandler):
    """Handler for the Gemini AI model."""

    def __init__(self, gemini_api: GeminiAPI):
        """Initialize the Gemini model handler."""
        self.gemini_api = gemini_api

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Generate a text response using the Gemini model."""
        response = await asyncio.wait_for(
            self.gemini_api.generate_response(
                prompt=prompt,
                context=context,
            ),
            timeout=300.0,
        )
        return response

    def get_system_message(self) -> str:
        """Get the system message for the Gemini model."""
        return "You are an AI assistant that helps users with tasks and answers questions helpfully, accurately, and ethically."

    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name."""
        return "🧠 Gemini"
