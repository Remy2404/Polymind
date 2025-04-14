from services.model_handlers import ModelHandler
from services.openrouter_api import OpenRouterAPI
from typing import List, Dict, Any, Optional
import logging


class DeepCoderHandler(ModelHandler):
    """Handler for the DeepCoder 14B model from Agentica on OpenRouter."""

    def __init__(self, openrouter_api: OpenRouterAPI):
        """
        Initialize the DeepCoder model handler.

        Args:
            openrouter_api: OpenRouterAPI instance for making API calls.
        """
        self.openrouter_api = openrouter_api
        self.logger = logging.getLogger(__name__)
        self.model_name = "agentica-org/deepcoder-14b-preview:free"

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """
        Generate a text response using the DeepCoder model.

        Args:
            prompt: The text prompt to generate a response for.
            context: Optional chat history for context.
            temperature: Controls randomness. Higher values mean more random completions.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            A string containing the generated response.
        """
        self.logger.info(f"Generating response using {self.model_name}")

        # Format messages for OpenRouter API
        messages = []

        # Add conversation history if provided
        if context:
            messages.extend(context)

        # Add the current prompt as the final user message
        messages.append({"role": "user", "content": prompt})

        # Call OpenRouter API with the DeepCoder model
        try:
            response = await self.openrouter_api.generate_response(
                prompt=prompt,
                context=context,
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response:
                self.logger.info(f"DeepCoder response received ({len(response)} chars)")
                return response
            else:
                self.logger.error("No valid response from DeepCoder")
                return "I'm sorry, I encountered an error generating a response."

        except Exception as e:
            self.logger.error(f"Error generating DeepCoder response: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def get_system_message(self) -> str:
        """Get the system message for the model."""
        return """You are DeepCoder, a specialized AI assistant focused on programming and software development.
You excel at:
- Writing clean, efficient code in multiple languages
- Explaining complex programming concepts
- Debugging and fixing issues in code
- Designing software architecture
- Interpreting technical documentation

Always provide context about your thinking process when solving coding problems."""

    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name."""
        return "🧑‍💻 DeepCoder 14B"


class Llama4MaverickHandler(ModelHandler):
    """Handler for the Llama-4-Maverick model from Meta on OpenRouter."""

    def __init__(self, openrouter_api: OpenRouterAPI):
        """
        Initialize the Llama-4-Maverick model handler.

        Args:
            openrouter_api: OpenRouterAPI instance for making API calls.
        """
        self.openrouter_api = openrouter_api
        self.logger = logging.getLogger(__name__)
        self.model_name = "meta-llama/llama-4-maverick:free"

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """
        Generate a text response using the Llama-4-Maverick model.

        Args:
            prompt: The text prompt to generate a response for.
            context: Optional chat history for context.
            temperature: Controls randomness. Higher values mean more random completions.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            A string containing the generated response.
        """
        self.logger.info(f"Generating response using {self.model_name}")

        # Format messages for OpenRouter API
        messages = []

        # Add conversation history if provided
        if context:
            messages.extend(context)

        # Add the current prompt as the final user message
        messages.append({"role": "user", "content": prompt})

        # Call OpenRouter API with the Llama-4-Maverick model
        try:
            response = await self.openrouter_api.generate_response(
                prompt=prompt,
                context=context,
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response:
                self.logger.info(
                    f"Llama-4-Maverick response received ({len(response)} chars)"
                )
                return response
            else:
                self.logger.error("No valid response from Llama-4-Maverick")
                return "I'm sorry, I encountered an error generating a response."

        except Exception as e:
            self.logger.error(f"Error generating Llama-4-Maverick response: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def get_system_message(self) -> str:
        """Get the system message for the model."""
        return """You are Maverick, powered by Llama 4, a helpful, harmless, and honest AI assistant. 
You have expertise across a wide range of topics and can assist with creative writing, 
information retrieval, thoughtful advice, and engaging conversations.
Always aim to provide the most relevant and accurate information, while acknowledging
the limits of your knowledge when appropriate."""

    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name."""
        return "🦙 Llama-4 Maverick"


class OptimusAlphaHandler(ModelHandler):
    """Handler for the Optimus Alpha model on OpenRouter."""

    def __init__(self, openrouter_api: OpenRouterAPI):
        """
        Initialize the Optimus Alpha model handler.

        Args:
            openrouter_api: OpenRouterAPI instance for making API calls.
        """
        self.openrouter_api = openrouter_api
        self.logger = logging.getLogger(__name__)
        self.model_name = "openrouter/optimus-alpha"

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> str:
        """
        Generate a text response using the Optimus Alpha model.

        Args:
            prompt: The text prompt to generate a response for.
            context: Optional chat history for context.
            temperature: Controls randomness. Higher values mean more random completions.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            A string containing the generated response.
        """
        self.logger.info(f"Generating response using {self.model_name}")

        # Format messages for OpenRouter API
        messages = []

        # Add conversation history if provided
        if context:
            messages.extend(context)

        # Add the current prompt as the final user message
        messages.append({"role": "user", "content": prompt})

        # Call OpenRouter API with the Optimus Alpha model
        try:
            response = await self.openrouter_api.generate_completion(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                response_text = response["choices"][0]["message"]["content"]
                self.logger.info(
                    f"Optimus Alpha response received ({len(response_text)} chars)"
                )
                return response_text
            else:
                self.logger.error(
                    f"Invalid response format from Optimus Alpha: {response}"
                )
                return "I'm sorry, I encountered an error generating a response."

        except Exception as e:
            self.logger.error(f"Error generating Optimus Alpha response: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def get_system_message(self) -> str:
        """Get the system message for the model."""
        return """You are Optimus Alpha, an advanced AI assistant optimized for complex problem-solving.
You excel at:
- Answering difficult questions with nuanced, accurate information
- Providing holistic analysis across multiple domains
- Synthesizing information from various sources
- Helping users break down complex problems
- Delivering thoughtful explanations with clarity and precision

Always consider multiple perspectives and acknowledge limitations in your knowledge."""

    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name."""
        return "🌀 Optimus Alpha"
