import os
import logging
import asyncio
import re
from typing import List, Dict, Optional, AsyncGenerator
from dotenv import load_dotenv
from together import Together
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class DeepSeekLLM:
    """Handler for DeepSeek LLM using the Together AI API."""
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        max_concurrent_requests: int = 20,
    ):
        """Initialize the DeepSeek LLM client."""
        self.model_name = model_name
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            logger.warning("TOGETHER_API_KEY not found in environment variables")
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.client = Together(api_key=self.api_key) if self.api_key else None
        self.recent_conversations = {}
        logger.info(f"Initialized DeepSeekLLM with model '{self.model_name}'")
    def _remove_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from the model output."""
        pattern = r"<think>.*?</think>"
        cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        return cleaned_text.strip()
    def _add_anti_thinking_instruction(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Add instruction to prevent thinking tags in output."""
        modified_messages = messages.copy()
        system_message_idx = -1
        for i, msg in enumerate(modified_messages):
            if msg["role"] == "system":
                system_message_idx = i
                break
        anti_thinking_instruction = (
            "Important: DO NOT use <think> or </think> tags in your response. "
            "Provide your answer directly without showing your reasoning process."
        )
        if system_message_idx >= 0:
            modified_messages[system_message_idx][
                "content"
            ] += f"\n\n{anti_thinking_instruction}"
        else:
            modified_messages.insert(
                0, {"role": "system", "content": anti_thinking_instruction}
            )
        return modified_messages
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        timeout: float = 180.0,
    ) -> str:
        """Generate text from a prompt using the provided message list."""
        response = await self.generate_chat_response(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> Optional[str]:
        """Generate a response to a conversation."""
        if not self.client:
            logger.error("Together AI client not initialized - missing API key")
            return None
        logger.info(f"Generating response using {self.model_name}")
        modified_messages = self._add_anti_thinking_instruction(messages)
        async with self.semaphore:
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=modified_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                generated_text = response.choices[0].message.content
                cleaned_text = self._remove_thinking_tags(generated_text)
                logger.info(
                    f"Successfully generated response ({len(cleaned_text)} chars)"
                )
                return cleaned_text
            except Exception as e:
                logger.error(f"Error generating text with DeepSeek LLM: {str(e)}")
                return None
    async def stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        """Stream a response from the model."""
        if not self.client:
            logger.error("Together AI client not initialized - missing API key")
            yield "Error: API client not initialized"
            return
        logger.info(f"Streaming response using {self.model_name}")
        modified_messages = self._add_anti_thinking_instruction(messages)
        async with self.semaphore:
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=modified_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True,
                )
                buffer = ""
                in_think_tag = False
                think_buffer = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        for char in content:
                            buffer += char
                            if buffer.endswith("<think>"):
                                in_think_tag = True
                                buffer = buffer[:-7]
                            if in_think_tag and buffer.endswith("</think>"):
                                in_think_tag = False
                                buffer = buffer[:-8]
                            if in_think_tag:
                                think_buffer += char
                            else:
                                if (
                                    buffer
                                    and not buffer.endswith("<")
                                    and not buffer.endswith("<t")
                                    and not buffer.endswith("<th")
                                    and not buffer.endswith("<thi")
                                    and not buffer.endswith("<thin")
                                    and not buffer.endswith("<think")
                                ):
                                    yield buffer
                                    buffer = ""
                if buffer and not in_think_tag:
                    yield buffer
                logger.info("Successfully completed streaming response")
            except Exception as e:
                logger.error(f"Error streaming text with DeepSeek LLM: {str(e)}")
                yield f"Error: {str(e)}"
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs,
    ) -> Optional[str]:
        """
        Generate response method that matches the expected interface for fallback handler.
        This method converts the prompt and context into the expected message format.
        """
        messages = []
        if context:
            for msg in context:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if not messages:
            return None
        return await self.generate_chat_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name for DeepSeek models."""
        return "🧠 DeepSeek"
    def get_system_message(self) -> str:
        """
        Return the system message for DeepSeek models.
        This is used by the prompt formatter for consistent system prompts.
        """
        return (
            "You are DeepSeek, an advanced reasoning AI model that excels at complex problem-solving. "
            "Be concise, helpful, and accurate."
        )
