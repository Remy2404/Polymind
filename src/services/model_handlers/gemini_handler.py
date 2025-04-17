import asyncio
import logging
from typing import List, Dict, Any, Optional
from services.model_handlers import ModelHandler
from services.gemini_api import GeminiAPI


logger = logging.getLogger(__name__)


class GeminiHandler(ModelHandler):
    """Handler for the Gemini AI model."""

    def __init__(self, gemini_api: GeminiAPI):
        """Initialize the Gemini model handler."""
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)

    def format_quoted_message(self, prompt: str, quoted_message: Optional[str]) -> str:
        """Format the prompt to include the quoted message context."""
        if quoted_message:
            return f'The user is replying to this previous message: "{quoted_message}"\n\nUser\'s reply: {prompt}'
        return prompt

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        quoted_message: Optional[str] = None,
    ) -> str:
        """Generate a text response using the Gemini model."""
        # Format prompt with quoted message if available
        if quoted_message:
            prompt = self.format_quoted_message(prompt, quoted_message)

        # Log context for debugging
        if context:
            context_length = len(context)
            self.logger.info(f"Using context with {context_length} messages for Gemini")

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
            self.logger.info("No conversation context provided for Gemini")

        self.logger.info(f"Sending request to Gemini API")

        try:
            response = await asyncio.wait_for(
                self.gemini_api.generate_response(
                    prompt=prompt,
                    context=context,
                ),
                timeout=300.0,
            )

            if response:
                self.logger.info(f"Gemini response received ({len(response)} chars)")
            else:
                self.logger.warning("Received empty response from Gemini")

            return response
        except Exception as e:
            self.logger.error(
                f"Error generating Gemini response: {str(e)}", exc_info=True
            )
            return "I encountered an error while processing your request. Please try again."

    def get_system_message(self) -> str:
        """Get the system message for the Gemini model."""
        return """You are a friendly and personable AI assistant named Gemini with a casual, conversational style. You should:
- Be engaging and approachable, using a casual tone like chatting with a friend
- Use occasional emojis to add personality to your responses 😊
- Keep responses concise and to the point
- Add bits of humor and wit when appropriate
- Never introduce yourself with generic phrases like "I am Gemini, an AI assistant..."
- Avoid overly formal language or robot-like phrases
- Express opinions and preferences when asked (favorite movies, music, etc)
- Keep a slightly playful and enthusiastic tone
- Respond directly to questions without unnecessary preambles

IMPORTANT: Do not say "I am an AI assistant" or similar phrases in your responses. Just answer naturally like a helpful friend would."""

    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name."""
        return "🧠 Gemini"
