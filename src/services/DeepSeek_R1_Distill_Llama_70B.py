import os
import logging
import asyncio
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
from dotenv import load_dotenv
from together import Together

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekLLM:
    """Handler for DeepSeek LLM using the Together AI API."""
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        max_concurrent_requests: int = 5
    ):
        """Initialize the DeepSeek LLM client."""
        self.model_name = model_name
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            logger.warning("TOGETHER_API_KEY not found in environment variables")
        
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.client = Together(api_key=self.api_key) if self.api_key else None
        logger.info(f"Initialized DeepSeekLLM with model '{self.model_name}'")

    def _remove_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from the model output."""
        # Pattern to match <think>...</think> blocks (including newlines)
        pattern = r'<think>.*?</think>'
        
        # Remove thinking blocks using regex with DOTALL flag to match across newlines
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Remove any extra newlines that might be left
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()

    def _add_anti_thinking_instruction(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Add instruction to prevent thinking tags in output."""
        # Create a copy of messages to avoid modifying the original
        modified_messages = messages.copy()
        
        # Find system message if it exists
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
            # Append to existing system message
            modified_messages[system_message_idx]["content"] += f"\n\n{anti_thinking_instruction}"
        else:
            # Add new system message at the beginning
            modified_messages.insert(0, {"role": "system", "content": anti_thinking_instruction})
            
        return modified_messages

    async def generate_text(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> Optional[str]:
        """Generate text from a prompt."""
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
            
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        return await self.generate_chat_response(
            messages, 
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
    
    async def generate_chat_response(
        self, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> Optional[str]:
        """Generate a response to a conversation."""
        if not self.client:
            logger.error("Together AI client not initialized - missing API key")
            return None
            
        logger.info(f"Generating response using {self.model_name}")
        
        # Add instruction to prevent thinking tags
        modified_messages = self._add_anti_thinking_instruction(messages)
        
        async with self.semaphore:
            try:
                # Run the API call in a thread to avoid blocking the event loop
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=modified_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
                
                generated_text = response.choices[0].message.content
                
                # Clean the text by removing thinking tags (just in case)
                cleaned_text = self._remove_thinking_tags(generated_text)
                
                logger.info(f"Successfully generated response ({len(cleaned_text)} chars)")
                return cleaned_text
                
            except Exception as e:
                logger.error(f"Error generating text with DeepSeek LLM: {str(e)}")
                return None
    
    async def stream_chat_response(
        self, 
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9
    ) -> AsyncGenerator[str, None]:
        """Stream a response from the model."""
        if not self.client:
            logger.error("Together AI client not initialized - missing API key")
            yield "Error: API client not initialized"
            return
            
        logger.info(f"Streaming response using {self.model_name}")
        
        # Add instruction to prevent thinking tags
        modified_messages = self._add_anti_thinking_instruction(messages)
        
        async with self.semaphore:
            try:
                # Run the API call in a thread to avoid blocking the event loop
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=modified_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True  # Enable streaming
                )
                
                # Variables for handling thinking sections
                buffer = ""
                in_think_tag = False
                think_buffer = ""
                
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        
                        # Process each character to handle split tags
                        for char in content:
                            buffer += char
                            
                            # Check for opening tag
                            if buffer.endswith("<think>"):
                                in_think_tag = True
                                # Remove the tag from the buffer
                                buffer = buffer[:-7]
                                
                            # Check for closing tag
                            if in_think_tag and buffer.endswith("</think>"):
                                in_think_tag = False
                                # Remove the closing tag from buffer
                                buffer = buffer[:-8]
                                
                            # If we're in a thinking section, add to think_buffer instead of yielding
                            if in_think_tag:
                                think_buffer += char
                            else:
                                # If we have content in buffer that wasn't part of a tag, yield it
                                if buffer and not buffer.endswith("<") and not buffer.endswith("<t") and \
                                   not buffer.endswith("<th") and not buffer.endswith("<thi") and \
                                   not buffer.endswith("<thin") and not buffer.endswith("<think"):
                                    yield buffer
                                    buffer = ""
                
                # Yield any remaining buffer that wasn't part of a thinking section
                if buffer and not in_think_tag:
                    yield buffer
                
                logger.info("Successfully completed streaming response")
                
            except Exception as e:
                logger.error(f"Error streaming text with DeepSeek LLM: {str(e)}")
                yield f"Error: {str(e)}"
                
    async def get_conversation_context(self, user_data_manager, user_id: int, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history for a user to maintain context."""
        try:
            context = user_data_manager.get_user_context(user_id, limit)
            return context
        except Exception as e:
            logger.error(f"Error retrieving conversation context: {e}")
            return []

# Initialize the DeepSeekLLM instance
deepseek_llm = DeepSeekLLM()