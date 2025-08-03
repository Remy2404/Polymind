import logging
from typing import Dict, Optional, Any


class PromptFormatter:
    """Handles formatting prompts for AI model consumption."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def apply_response_guidelines(
        self, prompt: str, model_handler, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Optionally apply response style guidelines based on the selected model."""
        try:
            # Get system message from the model handler
            system_message = model_handler.get_system_message()

            # Detect if user is asking for long-form content
            long_form_indicators = [
                "100",
                "list",
                "q&a",
                "qcm",
                "questions",
                "examples",
                "write me",
                "generate",
                "create",
                "explain in detail",
                "step by step",
                "tutorial",
                "guide",
                "comprehensive",
            ]

            is_long_form_request = any(
                indicator in prompt.lower() for indicator in long_form_indicators
            )

            # Optionally enhance prompt for long-form requests
            if is_long_form_request:
                enhanced_prompt = (
                    f"{system_message}\n\n"
                    "Please provide a detailed, comprehensive, and well-structured response as requested by the user.\n\n"
                    f"User query: {prompt}"
                )
            else:
                enhanced_prompt = f"{system_message}\n\nUser query: {prompt}"

            return enhanced_prompt
        except Exception as e:
            self.logger.error(f"Error applying response guidelines: {str(e)}")
            return prompt

    def add_context(self, prompt: str, context_type: str, context_text: str) -> str:
        """Optionally add specific types of context to the prompt."""
        if context_type == "image":
            return f"The user is referring to previously shared images. Here's the context of those images:\n\n{context_text}\n\nUser's question: {prompt}"
        elif context_type == "document":
            return f"The user is referring to previously processed documents. Here's the context of those documents:\n\n{context_text}\n\nUser's question: {prompt}"
        elif context_type == "quote":
            return f'The user is replying to this message: "{context_text}"\n\nUser\'s reply: {prompt}'

        # Default case
        return prompt
