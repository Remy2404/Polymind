import os
import logging
import asyncio
import time
import io
from typing import Optional, List
from PIL import Image
import requests
from dotenv import load_dotenv
from together import Together

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TogetherAIImageGenerator:
    """Image generation using the Together AI API."""
    
    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-schnell-Free",
        max_concurrent_requests: int = 20,
        request_timeout: int = 120
    ):
        """
        Initialize the TogetherAIImageGenerator.
        
        Args:
            model_name (str): The Together AI model to use for image generation
            max_concurrent_requests (int): Maximum number of concurrent API requests
            request_timeout (int): Timeout in seconds for API requests
        """
        self.model_name = model_name
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            logger.warning("TOGETHER_API_KEY not found in environment variables")
        
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.timeout = request_timeout
        self.client = Together(api_key=self.api_key) if self.api_key else None
        logger.info(f"Initialized TogetherAIImageGenerator with model '{self.model_name}'")

    async def generate_image(
        self, 
        prompt: str,
        negative_prompt: str = None,
        num_steps: int = 4,  # Changed default to 4 (max for Together)
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 7.5,
        seed: int = None
    ) -> Optional[Image.Image]:
        """
        Generate an image from the given prompt using Together AI.
        
        Args:
            prompt (str): The text description for image generation
            negative_prompt (str): Things to avoid in the image
            num_steps (int): Number of diffusion steps (1-4 for Together AI)
            width (int): Width of the generated image
            height (int): Height of the generated image
            guidance_scale (float): How closely to follow the prompt
            seed (int): Random seed for reproducibility
            
        Returns:
            Optional[Image.Image]: The generated PIL Image or None if generation failed
        """
        if not self.client:
            logger.error("Together AI client not initialized - missing API key")
            return None
            
        # Use a random seed if not provided
        if seed is None:
            seed = int(time.time()) % 1000000
            
        # Ensure steps is within Together AI's valid range (1-4)
        steps = max(1, min(4, num_steps))
        
        logger.info(f"Generating image with prompt: '{prompt}'")
        start_time = time.time()
        
        async with self.semaphore:
            try:
                # Use asyncio.to_thread to run the synchronous API call in a separate thread
                response = await asyncio.to_thread(
                    self.client.images.generate,
                    prompt=prompt,
                    model=self.model_name,
                    steps=steps,  # Use the validated steps value
                    width=width,
                    height=height,
                    seed=seed,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale
                )
                
                # Get the image URL from the response
                image_url = response.data[0].url
                
                # Download the image
                image_response = requests.get(image_url, timeout=self.timeout)
                image_response.raise_for_status()
                
                # Create PIL image from the response content
                image = Image.open(io.BytesIO(image_response.content))
                
                generation_time = time.time() - start_time
                logger.info(f"Image generated successfully in {generation_time:.2f}s")
                return image
                
            except Exception as e:
                logger.error(f"Error generating image with Together AI: {str(e)}")
                return None
    
    async def generate_images(
        self,
        prompt: str,
        num_images: int = 1,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate multiple images from the same prompt.
        
        Args:
            prompt (str): The text description for image generation
            num_images (int): Number of images to generate
            **kwargs: Additional parameters for image generation
            
        Returns:
            List[Image.Image]: List of generated PIL Images
        """
        tasks = []
        for i in range(num_images):
            # Use different seeds for different images
            seed = kwargs.get('seed', None)
            if seed is not None:
                kwargs['seed'] = seed + i
            tasks.append(self.generate_image(prompt, **kwargs))
            
        results = await asyncio.gather(*tasks)
        # Filter out None results
        return [img for img in results if img is not None]

# Initialize the TogetherAIImageGenerator instance
together_ai_image_generator = TogetherAIImageGenerator()