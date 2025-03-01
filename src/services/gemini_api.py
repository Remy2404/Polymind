import logging
import google.generativeai as genai
# from google import genai as genai_client
from typing import Optional, List, Dict, BinaryIO
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
import httpx
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
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536,
            "response_mime_type": "text/plain"
        }
        try:
            self.vision_model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
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
                        {"mime_type": "image/jpeg, image.png", "data": processed_image}
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

    async def generate_image_with_imagen3(self, prompt: str) -> Optional[bytes]:
        """
        Generate an image using Google's Imagen API.
        Returns the image as bytes.
        """
        await self.rate_limiter.acquire()
        
        try:
            # Use a proper configuration for image generation
            import httpx
            import json
            import base64
            
            # Update to use the recommended model
            url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            }
            
            # Format the request specifically for image generation
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": f"Generate a high-quality, detailed image of: {prompt}. Return the image only without any text."
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "temperature": 0.4,
                    "topP": 0.95,
                    "topK": 32,
                    "maxOutputTokens": 2048
                }
            }
            
            self.logger.info(f"Sending image generation request for: {prompt}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=data, timeout=60.0)
                
                # Handle error responses
                if response.status_code != 200:
                    self.logger.error(f"API error: {response.status_code} - {response.text}")
                    return None
                    
                response_data = response.json()
                
                # Extract the image data from the response
                if 'candidates' in response_data and response_data['candidates']:
                    for part in response_data['candidates'][0]['content']['parts']:
                        if 'inline_data' in part:  # Note: changed from 'inlineData' to 'inline_data'
                            # Extract base64 encoded image
                            mime_type = part['inline_data']['mime_type']
                            image_data = part['inline_data']['data']
                            binary_image = base64.b64decode(image_data)
                            
                            # Cache the image if cache is available
                            if self.image_cache:
                                cache_document = {
                                    "prompt": prompt,
                                    "image_data": binary_image,
                                    "model": "imagen3",
                                    "timestamp": datetime.datetime.utcnow()
                                }
                                await asyncio.to_thread(self.image_cache.insert_one, cache_document)
                            
                            self.logger.info(f"Successfully generated image for: {prompt}")
                            return binary_image
                
                self.logger.warning(f"No image found in response for prompt: {prompt}")
                return None
                        
        except Exception as e:
            self.logger.error(f"Imagen 3 generation error: {str(e)}")
            return None

    async def generate_response(self, prompt: str, context: List[Dict] = None, image_context: str = None, document_context: str = None) -> Optional[str]:
        """
        Generate a text response based on the provided prompt and context.
        Returns the response as a string.
        """
        # Acquire rate limiter
        await self.rate_limiter.acquire()

        try:
            # Prepare the conversation history
            conversation = []
            
            # Add bot identification at the start of the conversation as a system message
            conversation.append(genai.types.ContentDict(
                role="user",
                parts=["You are Gembot, an AI assistant that can help with various tasks."]
            ))
            
            # Add system context about memory capabilities
            memory_context = ("You have the ability to remember previous conversations including "
                             "descriptions of images and documents the user has shared. When answering, "
                             "consider text conversations, image descriptions, and document content in your context.")
                             
            conversation.append(genai.types.ContentDict(
                role="user",
                parts=[memory_context]
            ))
            
            # Add conversation context
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
            
            # Generate the response
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

    async def generate_content(self, content, generation_config, max_retries=5):
        retry_delay = 1  # Start with 1 second

        for attempt in range(max_retries):
            async with self.rate_limiter:
                try:
                    response = await genai.generate_content(content, generation_config=generation_config)
                    return response
                except Exception as e:
                    if "RATE_LIMIT_EXCEEDED" in str(e).upper():
                        self.logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        self.logger.error(f"Error generating content: {e}")
                        raise
        self.logger.error("Max retries exceeded. Unable to generate content.")
        raise Exception("Service is currently unavailable. Please try again later.")