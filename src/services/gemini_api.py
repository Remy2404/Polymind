import logging
import google.generativeai as genai
from typing import Optional, List, Dict
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import GEMINI_API_KEY
from src.services.rate_limiter import RateLimiter
from src.utils.telegramlog import telegram_logger

class GeminiAPI:
    def __init__(self):
        # Initialize API configuration and logger
        telegram_logger.log_message("Initializing Gemini API", 0)
        
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found or empty")
            
        genai.configure(api_key=GEMINI_API_KEY)
        
        self.generation_config = {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        try:
            self.model = genai.GenerativeModel(
                model_name="gemini-pro",
                generation_config=self.generation_config
            )
            self.vision_model = genai.GenerativeModel('gemini-pro-vision')
            self.rate_limiter = RateLimiter(requests_per_minute=60)
            telegram_logger.log_message("Gemini API initialized successfully", 0)
        except Exception as e:
            telegram_logger.log_error(f"Failed to initialize Gemini API: {str(e)}", 0)

    async def format_message(self, text: str) -> str:
        """Format the message for better Telegram display"""
        # Escape special characters for MarkdownV2
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        formatted = text
        for char in special_chars:
            formatted = formatted.replace(char, f'\\{char}')
        
        # Handle code blocks specially
        lines = formatted.split('\n')
        in_code_block = False
        formatted_lines = []
        code_block = []
        
        for line in lines:
            if line.strip().startswith('\\`\\`\\`'):
                if in_code_block:
                    # End code block
                    if code_block:
                        lang = code_block[0].replace('\\`\\`\\`', '').strip()
                        code = '\n'.join(code_block[1:])
                        # Don't escape characters inside code blocks
                        formatted_lines.append(f"```{lang}\n{code}\n```")
                    code_block = []
                    in_code_block = False
                else:
                    # Start code block
                    in_code_block = True
                    code_block = [line]
            elif in_code_block:
                # Don't escape characters inside code blocks
                code_block.append(line.replace('\\', ''))
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    async def generate_response(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        await self.rate_limiter.acquire()
        try:
            chat = self.model.start_chat(history=context or [])
            response = await asyncio.to_thread(
                chat.send_message,
                prompt
            )
            # Format the response for better display
            formatted_response = await self.format_message(response.text)
            return formatted_response
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I'm sorry, but I encountered an error processing your request\\. Please try again\\."

    async def analyze_image(self, image_data: bytes, prompt: str) -> str:
        await self.rate_limiter.acquire()
        try:
            response = await asyncio.to_thread(
                self.vision_model.generate_content,
                [prompt, image_data],
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            )
            telegram_logger.log_message("Image analysis completed successfully", 0)
            return response.text
        except Exception as e:
            telegram_logger.log_error(f"Image analysis error: {str(e)}", 0)
            return "I'm processing that image. Let me get back to you."

    async def generate_code(self, language: str, prompt: str, include_explanations: bool = True) -> Dict[str, str]:
        await self.rate_limiter.acquire()
        try:
            formatted_prompt = f"Generate {language} code for: {prompt}"
            if include_explanations:
                formatted_prompt += "\nPlease include explanations."

            response = await self.generate_response(formatted_prompt)
            
            # Parse response to separate code and explanation
            # This is a simple implementation - you might want to make it more robust
            code_parts = response.split("Explanation:", 1)
            result = {
                "code": code_parts[0].strip(),
                "explanation": code_parts[1].strip() if len(code_parts) > 1 else ""
            }
            
            return result
        except Exception as e:
            logging.error(f"Error generating code: {str(e)}")
            return {
                "code": "",
                "explanation": "Error generating code. Please try again."
            }