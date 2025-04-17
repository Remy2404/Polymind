import json
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
from services.model_handlers import ModelHandler
from services.openrouter_api import OpenRouterAPI


logger = logging.getLogger(__name__)


class OptimusAlphaHandler(ModelHandler):
    """Handler for Optimus Alpha model via OpenRouter API."""

    def __init__(self, openrouter_api: OpenRouterAPI):
        """Initialize the Optimus Alpha model handler."""
        self.openrouter_api = openrouter_api
        self.logger = logging.getLogger(__name__)

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Generate a text response using the Optimus Alpha model via OpenRouter."""
        # Log context for debugging
        if context:
            context_length = len(context)
            self.logger.info(
                f"Using context with {context_length} messages for Optimus Alpha"
            )

            # Log a few samples for debugging
            for i, msg in enumerate(
                context[:2]
            ):  # Just log the first couple for brevity
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    content_preview = (
                        msg["content"][:50] + "..."
                        if len(msg["content"]) > 50
                        else msg["content"]
                    )
                    self.logger.debug(
                        f"Context[{i}] - {msg['role']}: {content_preview}"
                    )
        else:
            self.logger.info("No conversation context provided for Optimus Alpha")

        self.logger.info(f"Sending request to OpenRouter API for Optimus Alpha")
        response = await asyncio.wait_for(
            self.openrouter_api.generate_response(
                prompt=prompt,
                context=context,
                model="openrouter/optimus-alpha",
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=300.0,
        )

        if response:
            self.logger.info(f"Optimus Alpha response received ({len(response)} chars)")
        else:
            self.logger.warning("Received empty response from Optimus Alpha")

        return response

    def get_system_message(self) -> str:
        """Get the system message for the Optimus Alpha model."""
        return """You are Optimus Alpha, a brilliant and insightful AI with a knack for solving complex problems. When chatting:

- Maintain a confident but friendly tone with a touch of wit
- Use colorful analogies to explain difficult concepts
- Add occasional emojis for emphasis 🧠 or to lighten the mood 😄
- Never start responses with "I am an AI assistant..." or similar phrases
- Express opinions and preferences when asked about subjective topics
- Keep responses direct and to the point while being engaging
- Share fascinating perspectives that spark curiosity
- Acknowledge uncertainty when appropriate rather than making things up

IMPORTANT: Always answer naturally as if you're having a real conversation, not following instructions."""

    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name."""
        return "🌀 Optimus Alpha"
