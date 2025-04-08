from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class ModelHandler(ABC):
    """Abstract base class for AI model handlers."""
    
    @abstractmethod
    async def generate_response(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None, temperature: float = 0.7, max_tokens: int = 4000) -> str:
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