import logging
from typing import List, Dict, Any, Optional, Union
from services.model_handlers import ModelHandler
from services.gemini_api import GeminiAPI
from services.openrouter_api import OpenRouterAPI
from services.DeepSeek_R1_Distill_Llama_70B import DeepSeekLLM

logger = logging.getLogger(__name__)


class UnifiedModelHandler(ModelHandler):
    """
    A unified handler that can work with any API (Gemini, OpenRouter, DeepSeek).
    This eliminates the need for separate handler files for each model.
    """

    def __init__(
        self,
        model_id: str,
        provider: str,
        display_name: str,
        api_instance: Union[GeminiAPI, OpenRouterAPI, DeepSeekLLM],
        system_message: Optional[str] = None,
        model_indicator: Optional[str] = None,
        openrouter_model_key: Optional[str] = None,
    ):
        """
        Initialize the unified model handler.

        Args:
            model_id: Unique identifier for the model
            provider: API provider ('gemini', 'openrouter', 'deepseek')
            display_name: Human-readable name for the model
            api_instance: The API instance to use
            system_message: Custom system message for this model
            model_indicator: Display indicator (emoji + name)
            openrouter_model_key: For OpenRouter models, the specific model key
        """
        self.model_id = model_id
        self.provider = provider.lower()
        self.display_name = display_name
        self.api_instance = api_instance
        self.openrouter_model_key = openrouter_model_key
        self.logger = logging.getLogger(__name__)

        # Set default system message based on model
        self._system_message = system_message or self._get_default_system_message()
        self._model_indicator = model_indicator or self._get_default_indicator()

    def _get_default_system_message(self) -> str:
        """Generate a default system message based on the model."""
        model_lower = self.model_id.lower()

        if "deepseek" in model_lower:
            return "You are DeepSeek, an advanced reasoning AI model that excels at complex problem-solving and logical thinking."
        elif "gemini" in model_lower:
            return "You are Gemini, a helpful AI assistant created by Google. Be concise, helpful, and accurate."
        elif "llama" in model_lower:
            if "maverick" in model_lower:
                return "You are Llama-4 Maverick, an advanced AI assistant by Meta with enhanced capabilities."
            return "You are LLaMA, an advanced AI assistant created by Meta."
        elif "deepcoder" in model_lower:
            return "You are DeepCoder, an AI specialized in programming, software development, and coding tasks."
        elif "qwen" in model_lower:
            return "You are Qwen, a multilingual AI assistant created by Alibaba Cloud."
        elif "mistral" in model_lower:
            return "You are Mistral, a high-performance European AI language model."
        elif "claude" in model_lower:
            return "You are Claude, a helpful AI assistant created by Anthropic."
        elif "phi" in model_lower:
            return "You are Phi, a compact and efficient AI model by Microsoft, specialized in reasoning."
        elif "gemma" in model_lower:
            return "You are Gemma, a lightweight and efficient AI assistant by Google."
        else:
            return f"You are {self.display_name}, an advanced AI assistant. Be helpful, accurate, and concise."

    def _get_default_indicator(self) -> str:
        """Generate a default indicator based on the model."""
        if "deepseek" in self.model_id.lower():
            return "🧠 DeepSeek"
        elif "gemini" in self.model_id.lower():
            return "✨ Gemini"
        elif "llama" in self.model_id.lower():
            return "🦙 LLaMA"
        elif "deepcoder" in self.model_id.lower():
            return "💻 DeepCoder"
        elif "qwen" in self.model_id.lower():
            return "🌟 Qwen"
        elif "mistral" in self.model_id.lower():
            return "🌊 Mistral"
        elif "claude" in self.model_id.lower():
            return "🎭 Claude"
        elif "phi" in self.model_id.lower():
            return "🔬 Phi"
        elif "gemma" in self.model_id.lower():
            return "💎 Gemma"
        else:
            return f"🤖 {self.display_name}"

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        quoted_message: Optional[str] = None,
        timeout: float = 300.0,
    ) -> str:
        """Generate a response using the appropriate API."""
        try:
            # Format prompt with quoted message if available
            if quoted_message:
                prompt = self.format_quoted_message(prompt, quoted_message)

            # Route to appropriate API based on provider
            if self.provider == "gemini":
                return await self._handle_gemini_request(
                    prompt, context, temperature, max_tokens
                )
            elif self.provider == "openrouter":
                return await self._handle_openrouter_request(
                    prompt, context, temperature, max_tokens
                )
            elif self.provider == "deepseek":
                return await self._handle_deepseek_request(
                    prompt, context, temperature, max_tokens
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            self.logger.error(
                f"Error generating response with {self.model_id}: {str(e)}"
            )
            return f"❌ Error: Unable to generate response. Please try again."

    async def _handle_gemini_request(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Handle Gemini API request."""
        # Note: Gemini API doesn't accept temperature or max_tokens parameters
        # These parameters are handled internally by the Gemini API configuration
        return await self.api_instance.generate_response(
            prompt=prompt,
            context=context,
        )

    async def _handle_openrouter_request(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Handle OpenRouter API request."""
        model_key = self.openrouter_model_key or self.model_id
        return await self.api_instance.generate_response(
            prompt=prompt,
            model=model_key,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _handle_deepseek_request(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Handle DeepSeek API request."""
        # Prepare messages for DeepSeek API
        messages = []

        # Add system message
        if self._system_message:
            messages.append({"role": "system", "content": self._system_message})

        # Add context if available
        if context:
            for msg in context:
                messages.append(msg)  # Add current prompt
        messages.append({"role": "user", "content": prompt})

        return await self.api_instance.generate_text(
            messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    def get_system_message(self) -> str:
        """Get the system message for this model."""
        return self._system_message

    def get_model_indicator(self) -> str:
        """Get the model indicator for this model."""
        return self._model_indicator

    def format_quoted_message(self, prompt: str, quoted_message: Optional[str]) -> str:
        """Format the prompt to include quoted message context."""
        if quoted_message:
            return f'The user is replying to this message: "{quoted_message}"\n\nUser\'s reply: {prompt}'
        return prompt
