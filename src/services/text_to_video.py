import os
import logging
import asyncio
import time
from typing import Optional
import aiohttp
import json
from dotenv import load_dotenv
import tempfile
from pathlib import Path
import io
import atexit
import base64
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToVideoGenerator:
    """Handles text-to-video generation using Hugging Face Inference API."""
    
    def __init__(self, 
                 image_model_name: str = "runwayml/stable-diffusion-v1-5",
                 video_model_name: str = "stabilityai/stable-video-diffusion-img2vid-xt",
                 max_concurrent_requests: int = 2):
        """
        Initialize the TextToVideoGenerator.
        
        Args:
            image_model_name: The model ID for text-to-image generation
            video_model_name: The model ID for image-to-video generation
            max_concurrent_requests: Maximum number of concurrent API requests
        """
        self.image_model_name = image_model_name
        self.video_model_name = video_model_name
        self.api_key = os.getenv("TEXT_TO_VIDEO_API_KEY")
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None
        
        if not self.api_key:
            logger.warning("TEXT_TO_VIDEO_API_KEY not found in environment variables")
        else:
            logger.info(f"Initialized TextToVideoGenerator with models '{self.image_model_name}' and '{self.video_model_name}'")

    async def init_session(self):
        """Initialize the aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            logger.debug("Created new aiohttp session for video generation")
    
    async def generate_video(
        self, 
        prompt: str, 
        negative_prompt: str = None,
        num_frames: int = 16,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5
    ) -> Optional[bytes]:
        """
        Generate a video based on the given text prompt.
        
        Args:
            prompt (str): The text description for the video
            negative_prompt (str, optional): Text to discourage in the generation
            num_frames (int): Number of frames to generate
            height (int): Height of the video
            width (int): Width of the video
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Higher value means more adherence to prompt
            
        Returns:
            Optional[bytes]: The generated video as bytes or None if generation failed
        """
        if not self.api_key:
            logger.error("Cannot generate video: API key is missing")
            return None
            
        await self.init_session()
        logger.info(f"Generating video for prompt: '{prompt}'")
        start_time = time.time()
        
        try:
            async with self.semaphore:
                # Step 1: Generate an image from the prompt
                image_bytes = await self._generate_image_from_text(prompt, negative_prompt, width, height, num_inference_steps, guidance_scale)
                if not image_bytes:
                    return None
                    
                # Step 2: Convert the image to a video
                video_bytes = await self._generate_video_from_image(
                    image_bytes, 
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps
                )
                
                if video_bytes:
                    generation_time = time.time() - start_time
                    logger.info(f"Video generated successfully in {generation_time:.2f}s")
                    return video_bytes
                else:
                    logger.error("Failed to generate video from image")
                    return None
                
        except Exception as e:
            logger.error(f"Video generation error: {str(e)}")
            return None

    async def _generate_image_from_text(self, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale):
        """Generate an image from text using Hugging Face API."""
        try:
            # Prepare payload for text-to-image API call
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": negative_prompt if negative_prompt else "",
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }
            }
            
            # Make API call to text-to-image model
            image_api_url = f"https://api-inference.huggingface.co/models/{self.image_model_name}"
            
            logger.info(f"Calling text-to-image API with prompt: {prompt}")
            async with self.session.post(image_api_url, json=payload) as response:
                if response.status == 200:
                    image_bytes = await response.read()
                    logger.info("Image generated successfully")
                    return image_bytes
                else:
                    error_text = await response.text()
                    logger.error(f"Error generating image: HTTP {response.status}, {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            return None

    async def _generate_video_from_image(self, image_bytes, num_frames, num_inference_steps):
        """Generate a video from an image using Hugging Face API."""
        try:
            # Encode image as base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare payload for image-to-video API call
            payload = {
                "inputs": image_base64,
                "parameters": {
                    "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps,
                    "motion_bucket_id": 127,  # Higher = more motion
                    "decode_chunk_size": 8,
                }
            }
            
            # Make API call to image-to-video model
            video_api_url = f"https://api-inference.huggingface.co/models/{self.video_model_name}"
            
            logger.info("Calling image-to-video API")
            async with self.session.post(video_api_url, json=payload) as response:
                if response.status == 200:
                    video_bytes = await response.read()
                    logger.info("Video generated successfully")
                    return video_bytes
                else:
                    error_text = await response.text()
                    logger.error(f"Error generating video: HTTP {response.status}, {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in video generation: {e}")
            return None

    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("TextToVideoGenerator resources cleaned up")

# Initialize the TextToVideoGenerator
text_to_video_generator = TextToVideoGenerator()

# Register the close method to run at exit
def shutdown():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(text_to_video_generator.close())
        else:
            try:
                loop.run_until_complete(text_to_video_generator.close())
            except RuntimeError:
                # If there's no event loop, create a new one
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(text_to_video_generator.close())
    except Exception as e:
        logger.warning(f"Error during shutdown: {e}")

atexit.register(shutdown)