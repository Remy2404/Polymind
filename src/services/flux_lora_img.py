import os
import aiohttp
import asyncio
from PIL import Image
from io import BytesIO
import logging
from dotenv import load_dotenv
import base64
import time
import atexit

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FluxLoraImageGenerator:
    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        api_endpoint: str = "https://api-inference.huggingface.co",
        max_concurrent_requests: int = 5,
        timeout: int = 300  # in seconds
    ):
        """
        Initializes the FluxLoraImageGenerator.

        Args:
            model_name (str): Name of the Hugging Face model.
            api_key (str): Hugging Face API key.
            api_endpoint (str): Base URL for the Hugging Face Inference API.
            max_concurrent_requests (int): Maximum number of concurrent API requests.
            timeout (int): Request timeout in seconds.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.full_url = f"{self.api_endpoint}/models/{self.model_name}"
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.cache = {}  # Simple in-memory cache
        self.session = None
        self.timeout = timeout
        logger.info(f"Initialized FluxLoraImageGenerator with model '{self.model_name}'.")

    async def init_session(self):
        """Initialize the aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            logger.info("aiohttp ClientSession initialized.")

    async def close(self):
        """Closes the aiohttp session."""
        if self.session:
            await self.session.close()
            logger.info("Closed aiohttp ClientSession.")

    async def text_to_image(
        self, 
        prompt: str, 
        num_images: int = 4, 
        num_inference_steps: int = 30,  
        width: int = 1024,                
        height: int = 1024,               
        guidance_scale: float = 8.5   
    ) -> list[Image.Image]:
        """
        Asynchronously generates images from a text prompt using the Hugging Face Inference API.

        Args:
            prompt (str): The text prompt for image generation.
            num_images (int): Number of images to generate.
            num_inference_steps (int): Number of inference steps.
            width (int): Width of the generated image.
            height (int): Height of the generated image.
            guidance_scale (float): Guidance scale for the model.

        Returns:
            list[Image.Image]: List of generated PIL Image objects.
        """
        await self.init_session()
        
        if prompt in self.cache:
            logger.info("Fetching images from cache.")
            return self.cache[prompt]
        
        tasks = [
            self._generate_single_image(
                prompt, 
                num_inference_steps, 
                width, 
                height, 
                guidance_scale,
                seed=int(time.time() * 1000) + i  # Different seed for each image
            )
            for i in range(num_images)
        ]

        images = await asyncio.gather(*tasks)
        valid_images = [img for img in images if isinstance(img, Image.Image)]

        if valid_images:
            self.cache[prompt] = valid_images

        return valid_images

    async def _generate_single_image(
        self, 
        prompt: str, 
        num_inference_steps: int, 
        width: int, 
        height: int, 
        guidance_scale: float,
        seed: int = None
    ) -> Image.Image:
        # Generate a random seed if none is provided
        if seed is None:
            seed = int(time.time() * 1000) % 1000000 + id(prompt) % 1000
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "seed": seed  # Add seed parameter
            }
        }

        retries = 3
        backoff = 2  # initial backoff in seconds

        for attempt in range(1, retries + 1):
            try:
                async with self.semaphore:
                    async with self.session.post(self.full_url, json=payload) as response:
                        if response.status == 200:
                            content_type = response.headers.get('Content-Type')
                            if content_type.startswith('image/'):
                                image_data = await response.read()
                                image = Image.open(BytesIO(image_data)).convert("RGB")
                                logger.info(f"Image generated successfully on attempt {attempt}.")
                                return image
                            else:
                                data = await response.json()
                                logger.error(f"Unexpected response format: {data}")
                                return None
                        elif response.status in {500, 502, 503, 504}:
                            logger.warning(f"Server error {response.status}. Attempt {attempt} of {retries}. Retrying in {backoff} seconds...")
                            await asyncio.sleep(backoff)
                            backoff *= 2
                        else:
                            error_detail = await response.text()
                            logger.error(f"Failed to generate image: {response.status} {error_detail}")
                            return None
            except asyncio.TimeoutError:
                logger.warning(f"Request timed out. Attempt {attempt} of {retries}. Retrying in {backoff} seconds...")
                await asyncio.sleep(backoff)
                backoff *= 2
            except Exception as e:
                logger.error(f"Unexpected error during image generation: {e}")
                return None

        logger.error("Max retries exceeded. Could not generate image.")
        return None

    def _process_response(self, data) -> Image.Image:
        """
        Processes the API response and converts it to a PIL Image.

        Args:
            data: The JSON response from the API.

        Returns:
            Image.Image: Generated PIL Image object or None if processing fails.
        """
        try:
            if isinstance(data, list) and len(data) > 0:
                # Adjust this based on the actual response structure
                # Example assumes the image is base64 encoded in 'generated_image' key
                if 'generated_image' in data[0]:
                    img_base64 = data[0]['generated_image']
                else:
                    # Fallback if the image is directly the first element
                    img_base64 = data[0]
                
                img_bytes = BytesIO(base64.b64decode(img_base64))
                image = Image.open(img_bytes).convert("RGB")
                return image
            else:
                logger.error(f"Unexpected response format: {data}")
                return None
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return None

# Initialize the FluxLoraImageGenerator instance
flux_lora_image_generator = FluxLoraImageGenerator(
    model_name="black-forest-labs/FLUX.1-schnell",
    api_key=os.getenv("TEXT_TO_IMAGE_API_KEY"),
    api_endpoint="https://api-inference.huggingface.co",
    max_concurrent_requests=5,
    timeout=300 
)
def shutdown():
    """Safe shutdown function that doesn't rely on event loop"""
    try:
        logging.info("Marking flux-lora resources for cleanup")
        # Just mark for cleanup - don't attempt async operations
        if hasattr(flux_lora_image_generator, 'session') and flux_lora_image_generator.session:
            flux_lora_image_generator.session = None
    except Exception as e:
        logging.warning(f"Error during shutdown: {e}")