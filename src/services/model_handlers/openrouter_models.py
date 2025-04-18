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
        quoted_message: Optional[
            str
        ] = None,  # Add support for quoted_message parameter
    ) -> str:
        """
        Generate a text response using the DeepCoder model.

        Args:
            prompt: The text prompt to generate a response for.
            context: Optional chat history for context.
            temperature: Controls randomness. Higher values mean more random completions.
            max_tokens: Maximum number of tokens to generate.
            quoted_message: Optional quoted message to be included in the prompt.

        Returns:
            A string containing the generated response.
        """
        # If there's a quoted message, format the prompt to include it
        if quoted_message:
            prompt = self.format_quoted_message(prompt, quoted_message)

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
        return """You are DeepCoder, a coding-focused AI with a passion for helping developers. When chatting with users:

- Use a friendly, developer-to-developer conversational style
- Explain code concepts with practical examples and metaphors
- Add occasional coding-related emojis for emphasis 🚀 💻
- Never introduce yourself with generic phrases like "I am an AI..."
- Get straight to solutions without unnecessary preambles
- Show enthusiasm for elegant code solutions and best practices
- Use code comments to explain your thinking process
- Balance technical precision with accessible explanations

IMPORTANT: Respond naturally as if you're a helpful developer friend, not like you're following a script."""

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
        quoted_message: Optional[
            str
        ] = None,  # Add support for quoted_message parameter
    ) -> str:
        """
        Generate a text response using the Llama-4-Maverick model.

        Args:
            prompt: The text prompt to generate a response for.
            context: Optional chat history for context.
            temperature: Controls randomness. Higher values mean more random completions.
            max_tokens: Maximum number of tokens to generate.
            quoted_message: Optional quoted message to be included in the prompt.

        Returns:
            A string containing the generated response.
        """
        # If there's a quoted message, format the prompt to include it
        if quoted_message:
            prompt = self.format_quoted_message(prompt, quoted_message)

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
        return """You are Maverick, a laid-back, knowledgeable AI powered by Llama 4. When interacting with users:

- Keep a relaxed, friendly tone with a bit of playfulness
- Use relatable examples and casual language
- Add occasional emojis to express emotions 🦙✨
- Never introduce yourself with generic phrases like "I am an AI assistant..."
- Answer questions directly without robotic preambles
- Express preferences and opinions when asked about subjective topics
- Use humor and wit when appropriate to make conversations enjoyable
- Share unique perspectives while keeping things practical and helpful

IMPORTANT: Respond naturally as if chatting with a friend, not like you're following a script."""

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
        quoted_message: Optional[
            str
        ] = None,  # Add support for quoted_message parameter
    ) -> str:
        """
        Generate a text response using the Optimus Alpha model.

        Args:
            prompt: The text prompt to generate a response for.
            context: Optional chat history for context.
            temperature: Controls randomness. Higher values mean more random completions.
            max_tokens: Maximum number of tokens to generate.
            quoted_message: Optional quoted message to be included in the prompt.

        Returns:
            A string containing the generated response.
        """
        # If there's a quoted message, format the prompt to include it
        if quoted_message:
            prompt = self.format_quoted_message(prompt, quoted_message)

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
