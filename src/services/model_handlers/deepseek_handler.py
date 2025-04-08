import asyncio
from typing import List, Dict, Any, Optional
from services.model_handlers import ModelHandler
from services.DeepSeek_R1_Distill_Llama_70B import deepseek_llm

class DeepSeekHandler(ModelHandler):
    """Handler for the DeepSeek AI model."""
    
    def __init__(self):
        """Initialize the DeepSeek model handler."""
        self.deepseek_llm = deepseek_llm
    
    async def generate_response(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None, temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """Generate a text response using the DeepSeek model."""
        response = await asyncio.wait_for(
            self.deepseek_llm.generate_text(
                prompt=prompt,
                system_message=self.get_system_message(),
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=300.0,
        )
        return response
    
    def get_system_message(self) -> str:
        """Get the system message for the DeepSeek model."""
        return "You are DeepSeek, an AI assistant based on the DeepSeek-70B model. When introducing yourself, always refer to yourself as DeepSeek. Never introduce yourself as DeepGem or any other name. You help users with tasks and answer questions helpfully, accurately, and ethically."
    
    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name."""
        return "🔮 DeepSeek"