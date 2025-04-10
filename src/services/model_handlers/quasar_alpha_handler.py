import json
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
from services.model_handlers import ModelHandler
from services.openrouter_api import OpenRouterAPI


logger = logging.getLogger(__name__)

class QuasarAlphaHandler(ModelHandler):
    """Handler for Quasar Alpha model via OpenRouter API."""

    def __init__(self, openrouter_api: OpenRouterAPI):
        """Initialize the Quasar Alpha model handler."""
        self.openrouter_api = openrouter_api
        self.logger = logging.getLogger(__name__)

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """Generate a text response using the Quasar Alpha model via OpenRouter."""
        # Log context for debugging
        if context:
            context_length = len(context)
            self.logger.info(f"Using context with {context_length} messages for Quasar Alpha")
            
            # Log a few samples for debugging
            for i, msg in enumerate(context[:2]):  # Just log the first couple for brevity
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                    self.logger.debug(f"Context[{i}] - {msg['role']}: {content_preview}")
        else:
            self.logger.info("No conversation context provided for Quasar Alpha")
            
        self.logger.info(f"Sending request to OpenRouter API for Quasar Alpha")
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
        
        if response:
            self.logger.info(f"Quasar Alpha response received ({len(response)} chars)")
        else:
            self.logger.warning("Received empty response from Quasar Alpha")
            
        return response

    def get_system_message(self) -> str:
        """Get the system message for the Quasar Alpha model."""
        return """You are Quasar Alpha, an advanced AI assistant that helps users with tasks and answers questions helpfully, accurately, and ethically.

IMPORTANT: You have a conversation memory and can remember previous exchanges with the user. If asked about previous questions or what was discussed earlier, reference your conversation history to provide an accurate response. Never tell the user you can't remember previous interactions unless your history is truly empty."""

    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name."""
        return "🌀 Quasar Alpha"
