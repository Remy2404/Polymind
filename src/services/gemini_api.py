import logging
import google.generativeai as genai
from typing import Optional, List, Dict
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import GEMINI_API_KEY
from src.services.rate_limiter import RateLimiter
from src.utils.telegramlog import telegram_logger

class GeminiAPI:
    def __init__(self):
        telegram_logger.log_message(0, "Initializing Gemini API")
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Enhanced generation config for better responses
        self.generation_config = {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        # Initialize models with proper error handling
        try:
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config=self.generation_config
            )
            self.vision_model = genai.GenerativeModel('gemini-pro-vision')
            self.rate_limiter = RateLimiter(requests_per_minute=60)
            self._initialize_safety_settings()
            telegram_logger.log_message(0, "Gemini API initialized successfully")
        except Exception as e:
            telegram_logger.log_error(0, f"Failed to initialize Gemini API: {str(e)}")
            raise

    def _initialize_safety_settings(self):
        self.safety_settings = [
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

    async def get_text_response(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        await self.rate_limiter.acquire()
        try:
            chat = self.model.start_chat(history=context or [], safety_settings=self.safety_settings)
            response = await asyncio.to_thread(chat.send_message, prompt)
            telegram_logger.log_message(0, f"Generated response for prompt: {prompt[:50]}...")
            return response.text
        except Exception as e:
            telegram_logger.log_error(0, f"Text response generation error: {str(e)}")
            return "I encountered an error processing your request. Please try again."

    async def generate_code(self, language: str, prompt: str, include_explanations: bool = True) -> Dict[str, str]:
        code_prompt = f"""
        Generate {language} code for: {prompt}
        Requirements:
        - Include error handling
        - Follow best practices
        - Add comments for clarity
        - Make it production-ready
        {' Include detailed explanations.' if include_explanations else ''}
        """
        
        await self.rate_limiter.acquire()
        try:
            response = await asyncio.to_thread(self.model.generate_content, code_prompt)
            result = {
                'code': self._extract_code(response.text),
                'explanation': self._extract_explanation(response.text)
            }
            telegram_logger.log_message(0, f"Generated code for language: {language}")
            return result
        except Exception as e:
            telegram_logger.log_error(0, f"Code generation error: {str(e)}")
            return {
                'code': '# Error generating code',
                'explanation': 'An error occurred while generating the code.'
            }

    async def analyze_image(self, image_data: bytes, prompt: str) -> str:
        await self.rate_limiter.acquire()
        try:
            response = await asyncio.to_thread(
                self.vision_model.generate_content,
                [prompt, image_data],
                safety_settings=self.safety_settings
            )
            telegram_logger.log_message(0, "Image analysis completed successfully")
            return response.text
        except Exception as e:
            telegram_logger.log_error(0, f"Image analysis error: {str(e)}")
            return "I encountered an error analyzing the image. Please try again."

    def _extract_code(self, response: str) -> str:
        code_blocks = []
        lines = response.split('\n')
        in_code_block = False
        current_block = []
        
        for line in lines:
            if '```' in line:
                if in_code_block:
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                in_code_block = not in_code_block
                continue
            if in_code_block:
                current_block.append(line)
                
        return '\n\n'.join(code_blocks) if code_blocks else response

    def _extract_explanation(self, response: str) -> str:
        explanation = []
        lines = response.split('\n')
        in_code_block = False
        
        for line in lines:
            if '```' in line:
                in_code_block = not in_code_block
                continue
            if not in_code_block and line.strip():
                explanation.append(line)
                
        return '\n'.join(explanation)
