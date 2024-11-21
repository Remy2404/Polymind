import logging
import google.generativeai as genai
from typing import Optional, List, Dict
import asyncio
import sys
import os
from PIL import Image, UnidentifiedImageError
import io
from services.rate_limiter import RateLimiter
from utils.telegramlog import telegram_logger
from dotenv import load_dotenv
from services.image_processing import ImageProcessor

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    telegram_logger.log_error("GEMINI_API_KEY not found in environment variables.", 0)
    sys.exit(1)

class GeminiAPI:
    def __init__(self):
        telegram_logger.log_message("Initializing Gemini API", 0)

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found or empty")

        genai.configure(api_key=GEMINI_API_KEY)

        # Generation configuration
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": 4096,
        }

        try:
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=self.generation_config,
            )
            self.vision_model = genai.GenerativeModel("gemini-1.5-flash")
            self.rate_limiter = RateLimiter(requests_per_minute=20)
            telegram_logger.log_message("Gemini API initialized successfully", 0)
        except Exception as e:
            telegram_logger.log_error(f"Failed to initialize Gemini API: {str(e)}", 0)
            raise

    async def format_message(self, text: str) -> str:
        """Format text for initial processing before Telegram Markdown formatting."""
        try:
            # Don't escape special characters here, just clean up the text
            # Remove any null characters or other problematic characters
            cleaned_text = text.replace('\x00', '').strip()
            return cleaned_text
        except Exception as e:
            logging.error(f"Error formatting message: {str(e)}")
            return text

    async def analyze_image(self, image_data: bytes, prompt: str) -> str:
        """Analyze an image and generate a response based on the prompt."""
        await self.rate_limiter.acquire()
        try:
            # Validate image first
            if not ImageProcessor.validate_image(image_data):
                return "Sorry, the image format is not supported. Please send a JPEG or PNG image."

            # Process the image
            processed_image = await ImageProcessor.prepare_image(image_data)
            
            # Generate response with proper error handling
            try:
                response = await asyncio.to_thread(
                    self.vision_model.generate_content,
                    [
                        prompt,  # First element is the text prompt
                        {"mime_type": "image/jpeg", "data": processed_image}  # Second element is the image data
                    ],
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    ],
                )
                
                if response and hasattr(response, 'text'):
                    formatted_response = await self.format_message(response.text)
                    return formatted_response
                else:
                    raise ValueError("Empty response from vision model")
                
            except Exception as e:
                logging.error(f"Vision model error: {str(e)}")
                return "I'm sorry, there was an error processing your image. Please try again later."
            
        except UnidentifiedImageError:
            logging.error("The provided image could not be identified.")
            return "Sorry, the image format is not supported. Please send a valid image."
        except Exception as e:
            telegram_logger.log_error(f"Image analysis error: {str(e)}", 0)
            return "I'm sorry, I encountered an error processing your image. Please try again with a different image."

    async def generate_response(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        """Generate a text response based on a given prompt and optional context."""
        await self.rate_limiter.acquire()
        try:
            if context:
                context = context[-5:]

            chat = self.model.start_chat(history=context or [])
            response = await asyncio.to_thread(
                chat.send_message,
                prompt,
                generation_config=self.generation_config,
                safety_settings=[
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ],
            )
            return await self.format_message(response.text)
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I encountered an error generating the response. Please try again."
