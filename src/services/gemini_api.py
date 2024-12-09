import logging
import google.generativeai as genai
from typing import Optional, List, Dict
import asyncio
import datetime
import sys
import os
from PIL import Image, UnidentifiedImageError
import io
from services.rate_limiter import RateLimiter
from utils.telegramlog import telegram_logger
from dotenv import load_dotenv
from services.image_processing import ImageProcessor
from database.connection import get_database, get_image_cache_collection

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    telegram_logger.log_error("GEMINI_API_KEY not found in environment variables.", 0)
    sys.exit(1)

class GeminiAPI:
    def __init__(self, vision_model, rate_limiter: RateLimiter):
        self.vision_model = vision_model
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(__name__)
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
            self.vision_model = genai.GenerativeModel("gemini-1.5-pro")
            self.rate_limiter = RateLimiter(requests_per_minute=20)
            # Initialize MongoDB connection
            self.db, self.client = get_database()
            self.image_cache = get_image_cache_collection(self.db)
            if self.image_cache is not None:
                self.logger.info("Image cache collection is ready.")
            else:
                self.logger.error("Failed to access image cache collection.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {str(e)}")
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
                        {"mime_type": "image/jpeg, image.png", "data": processed_image}  # Second element is the image data
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
    

    
    
    async def generate_image(self, prompt: str) -> Optional[bytes]:
        """
        Generate an image based on the provided text prompt using the Gemini API.
        Returns the image as bytes. Checks cache before generating a new image.
        """
        if self.image_cache is not None:
            cached_image = await asyncio.to_thread(
                self.image_cache.find_one, {"prompt": prompt}
            )
            if cached_image:
                self.logger.info(f"Cache hit for prompt: '{prompt}'")
                return cached_image['image_data']

        await self.rate_limiter.acquire()

        try:
            # Use the newer method for image generation
            response = await asyncio.to_thread(
                self.vision_model.generate_content,
                prompt,
                generation_config=self.generation_config,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )

            # Check if the response contains an image
            if response and response.parts and response.parts[0].images:
                image_bytes = response.parts[0].images[0].to_bytes()

                cache_document = {
                    "prompt": prompt,
                    "image_data": image_bytes,
                    "timestamp": datetime.datetime.utcnow()
                }
                try:
                    await asyncio.to_thread(
                        self.image_cache.insert_one, cache_document
                    )
                    self.logger.info(f"Cached image for prompt: '{prompt}'")
                except Exception as cache_error:
                    self.logger.error(f"Failed to cache image: {cache_error}")

                return image_bytes
            else:
                raise ValueError("No image data in response or empty response")

        except Exception as e:
            self.logger.error(f"Image generation error: {str(e)}")
            return None
    async def generate_response(self, prompt: str, context: List[Dict] = None) -> Optional[str]:
        """
        Generate a text response based on the provided prompt and context.
        Returns the response as a string.
        """
        # Acquire rate limiter
        await self.rate_limiter.acquire()
    
        try:
            # Prepare the conversation history
            conversation = []
            
            # Add bot identification at the start of the conversation as a user message
            conversation.append(genai.types.ContentDict(
                role="user",
                parts=["You are Gembot, an AI assistant developed by Ramy. Please introduce yourself as such when appropriate."]
            ))
            
            if context:
                for message in context:
                    conversation.append(genai.types.ContentDict(
                        role="user" if message['role'] == "user" else "model",
                        parts=[message['content']]
                    ))
            
            # Add the current prompt to the conversation
            conversation.append(genai.types.ContentDict(
                role="user",
                parts=[prompt]
            ))
    
            # Generate the response using the conversation history
            response = await asyncio.to_thread(
                self.vision_model.generate_content,
                conversation,
                generation_config=self.generation_config,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )
    
            if response.text:
                return response.text
        
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
        
        return None