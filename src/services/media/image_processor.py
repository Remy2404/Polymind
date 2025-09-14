import io
import logging
from typing import Optional, Union
from PIL import Image
import re


class ImageProcessor:
    """
    Handles image processing, analysis, and generation operations.
    Supports both bytes and BytesIO objects consistently.
    """

    def __init__(self, ai_client=None):
        """Initialize with optional AI client for image analysis"""
        self.ai_client = ai_client
        self.logger = logging.getLogger(__name__)
        self.max_image_size = 4096
        self.image_quality = 95

    async def generate_image(self, prompt: str) -> Optional[bytes]:
        """
        Generate an image based on the provided text prompt.
        Args:
            prompt: Text description for image generation
        Returns:
            Optional[bytes]: Generated image as bytes if successful, None otherwise
        """
        if not self.ai_client:
            self.logger.error("No AI client provided for image generation")
            return None
        try:
            if hasattr(self.ai_client, "generate_image"):
                self.logger.info(f"Generating image with prompt: {prompt}")
                return await self.ai_client.generate_image(prompt)
            else:
                self.logger.error("AI client does not support image generation")
                return None
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            return None

    async def analyze_image(
        self,
        image_data: Union[bytes, io.BytesIO],
        prompt: str = "Describe this image in detail",
    ) -> str:
        """
        Analyze an image using the AI client and return a text description.
        Args:
            image_data: Image as bytes or BytesIO
            prompt: Text prompt for the AI
        Returns:
            str: Analysis result from AI
        """
        if not self.ai_client:
            self.logger.error("No AI client provided for image analysis")
            return "Image analysis unavailable. AI client not configured."
        try:
            image_bytes_io = self._ensure_bytesio(image_data)
            if not self.validate_image(image_bytes_io):
                return "Sorry, the image format is not supported. Please send a JPEG or PNG image."
            processed_image = await self.prepare_image(image_bytes_io)
            if hasattr(self.ai_client, "analyze_image"):
                return await self.ai_client.analyze_image(processed_image, prompt)
            else:
                self.logger.error("AI client does not have analyze_image method")
                return "Sorry, image analysis is not available right now."
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            return f"Sorry, I couldn't analyze this image: {str(e)}"

    def _ensure_bytesio(self, image_data: Union[bytes, io.BytesIO]) -> io.BytesIO:
        """
        Ensure the image data is in BytesIO format for consistent processing.
        Args:
            image_data: Image as bytes or BytesIO
        Returns:
            io.BytesIO: Image in BytesIO format
        """
        if isinstance(image_data, io.BytesIO):
            image_data.seek(0)
            return image_data
        elif isinstance(image_data, bytes):
            return io.BytesIO(image_data)
        else:
            raise TypeError(f"Unsupported image data type: {type(image_data)}")

    def validate_image(self, image_data: Union[bytes, io.BytesIO]) -> bool:
        """
        Validate if the image data is in a supported format.
        Args:
            image_data: Image as bytes or BytesIO
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            image_io = self._ensure_bytesio(image_data)
            with Image.open(image_io) as img:
                supported_formats = ["JPEG", "JPG", "PNG", "WEBP", "GIF"]
                if img.format not in supported_formats:
                    self.logger.warning(f"Unsupported image format: {img.format}")
                    return False
                if img.size[0] * img.size[1] > 25000000:
                    self.logger.warning(f"Image too large: {img.size[0]}x{img.size[1]}")
                    return False
                image_io.seek(0)
                return True
        except Exception as e:
            self.logger.error(f"Image validation failed: {str(e)}")
            return False

    def get_mime_type(self, image_data: Union[bytes, io.BytesIO]) -> str:
        """
        Determine the MIME type of an image.
        Args:
            image_data: Image as bytes or BytesIO
        Returns:
            str: MIME type, defaulting to image/jpeg if detection fails
        """
        try:
            image_io = self._ensure_bytesio(image_data)
            with Image.open(image_io) as img:
                fmt = img.format.lower() if img.format else "jpeg"
                image_io.seek(0)
                return f"image/{fmt}"
        except Exception as e:
            self.logger.error(f"Error determining MIME type: {str(e)}")
            return "image/jpeg"

    async def prepare_image(self, image_data: Union[bytes, io.BytesIO]) -> io.BytesIO:
        """
        Prepare an image for AI processing by optimizing size and format.
        Args:
            image_data: Image as bytes or BytesIO
        Returns:
            io.BytesIO: Processed image
        """
        try:
            image_io = self._ensure_bytesio(image_data)
            with Image.open(image_io) as image:
                if image.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    if image.mode == "RGBA":
                        background.paste(image, mask=image.split()[3])
                    else:
                        background.paste(image, mask=image.split()[1])
                    image = background
                if max(image.size) > self.max_image_size:
                    ratio = self.max_image_size / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.LANCZOS)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                output = io.BytesIO()
                image.save(
                    output,
                    format="JPEG",
                    quality=self.image_quality,
                    optimize=True,
                )
                output.seek(0)
                return output
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise ValueError(f"Image processing failed: {str(e)}")

    def is_image_generation_request(self, message: str) -> bool:
        """
        Detect if a text message is requesting image generation.
        Args:
            message: Text message to analyze
        Returns:
            bool: True if it seems like an image generation request
        """
        image_request_patterns = [
            r"(?i)generate\s+(?:an?|some)\s+image",
            r"(?i)create\s+(?:an?|some)\s+image",
            r"(?i)make\s+(?:an?|some)\s+image",
            r"(?i)draw\s+(?:an?|some)",
            r"(?i)show\s+(?:me)?\s+(?:an?|some)\s+image",
            r"(?i)visualize\s+(?:an?|some)",
            r"(?i)picture\s+of",
            r"(?i)image\s+of",
            r"(?i)\(generating\s+(?:an?|some)\s+image",
            r"(?i)^generating\s+(?:an?|some)\s+image",
            r"(?i)\(.*image of.*\)",
        ]
        for pattern in image_request_patterns:
            if re.search(pattern, message):
                return True
        return False

    async def detect_image_generation_request(self, message: str) -> tuple[bool, str]:
        """
        Detect if a message is requesting image generation and extract the prompt.
        Args:
            message: Text message to analyze
        Returns:
            tuple: (is_request, image_prompt)
                - is_request: True if this is an image generation request
                - image_prompt: The extracted image prompt or empty string
        """
        if not self.is_image_generation_request(message):
            return False, ""
        prompt_patterns = [
            r"(?i)generate\s+(?:an?|some)\s+image\s+(?:of|showing|with|about|depicting)?\s*(.*)",
            r"(?i)create\s+(?:an?|some)\s+image\s+(?:of|showing|with|about|depicting)?\s*(.*)",
            r"(?i)make\s+(?:an?|some)\s+image\s+(?:of|showing|with|about|depicting)?\s*(.*)",
            r"(?i)draw\s+(?:an?|some)\s+(.*)",
            r"(?i)show\s+(?:me)?\s+(?:an?|some)\s+image\s+(?:of|showing|with|about|depicting)?\s*(.*)",
            r"(?i)visualize\s+(?:an?|some)\s+(.*)",
            r"(?i)picture\s+of\s+(.*)",
            r"(?i)image\s+of\s+(.*)",
            r"(?i)\(generating\s+(?:an?|some)\s+image\s+(?:of|showing|with|about|depicting)?\s*(.*?)(?:\)|$)",
            r"(?i)^generating\s+(?:an?|some)\s+image\s+(?:of|showing|with|about|depicting)?\s*(.*)",
        ]
        for pattern in prompt_patterns:
            match = re.search(pattern, message)
            if match and match.group(1).strip():
                return True, match.group(1).strip()
        if message.startswith("(") and ")" in message:
            content = message.strip("()")
            if re.search(r"(?i)image|picture|draw|generate|creating|showing", content):
                return True, content
        for command in [
            "generate image",
            "create image",
            "make image",
            "draw",
            "show image",
            "visualize",
        ]:
            if command.lower() in message.lower():
                prompt = message.lower().replace(command.lower(), "").strip()
                if prompt:
                    return True, prompt
        return True, message.strip()
