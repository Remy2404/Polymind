from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
class ModelHandler(ABC):
    """Abstract base class for AI model handlers."""
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        quoted_message: Optional[str] = None,
    ) -> str:
        """Generate a text response using the AI model."""
        pass
    @abstractmethod
    def get_system_message(self) -> str:
        """Get the system message for the model."""
        pass
    @abstractmethod
    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name."""
        pass
    def format_quoted_message(self, prompt: str, quoted_message: Optional[str]) -> str:
        """Format the prompt to include the quoted message context."""
        if quoted_message:
            return f'The user is replying to this message: "{quoted_message}"\n\nUser\'s reply: {prompt}'
        return prompt
