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
        genai.configure(api_key=GEMINI_API_KEY)
        
        self.generation_config = {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self._initialize_safety_settings()

        try:
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=self.generation_config)
            self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
            self.rate_limiter = RateLimiter(requests_per_minute=60)
            telegram_logger.log_message("Gemini API initialized successfully", 0)
        except Exception as e:
            telegram_logger.log_error(f"Failed to initialize Gemini API: {str(e)}", 0)

    def _initialize_safety_settings(self):
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]

    async def generate_response(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        await self.rate_limiter.acquire()
        try:
            chat = self.model.start_chat(history=context or [], safety_settings=self.safety_settings)
            response = await asyncio.to_thread(chat.send_message, prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I'm sorry, but I encountered an error processing your request. Please try again."

    async def analyze_image(self, image_data: bytes, prompt: str) -> str:
        await self.rate_limiter.acquire()
        try:
            response = await asyncio.to_thread(
                self.vision_model.generate_content,
                [prompt, image_data],
                safety_settings=self.safety_settings
            )
            telegram_logger.log_message("Image analysis completed successfully", 0)
            return response.text
        except Exception as e:
            telegram_logger.log_error(f"Image analysis error: {str(e)}", 0)
            return "I'm processing that image. Let me get back to you."