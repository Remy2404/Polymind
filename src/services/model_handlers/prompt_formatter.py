import os
import logging
import aiofiles
from typing import Dict, Optional, Any


class PromptFormatter:
    """Handles formatting prompts for AI model consumption."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def apply_response_guidelines(
        self, prompt: str, model_handler, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Apply appropriate response style guidelines based on the selected model."""
        try:
            # Get system message from the model handler
            system_message = model_handler.get_system_message()

            # Create enhanced prompt with system message and user query
            enhanced_prompt = f"{system_message}\n\nUser query: {prompt}"

            return enhanced_prompt
        except Exception as e:
            self.logger.error(f"Error applying response guidelines: {str(e)}")
            return prompt

    def add_context(self, prompt: str, context_type: str, context_text: str) -> str:
        """Add specific types of context to the prompt."""
        if context_type == "image":
            return f"The user is referring to previously shared images. Here's the context of those images:\n\n{context_text}\n\nUser's question: {prompt}"
        elif context_type == "document":
            return f"The user is referring to previously processed documents. Here's the context of those documents:\n\n{context_text}\n\nUser's question: {prompt}"
        elif context_type == "quote":
            return f'The user is replying to this message: "{context_text}"\n\nUser\'s reply: {prompt}'

        # Default case
        return prompt
