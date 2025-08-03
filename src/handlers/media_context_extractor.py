import logging
from typing import Dict, Any
import re


class MediaContextExtractor:
    """Extracts context from previously shared media messages."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def is_referring_to_image(self, message: str) -> bool:
        """
        Detects if a message is referring to a previously shared image.

        Args:
            message: User's message text

        Returns:
            bool: True if the message appears to refer to a previously shared image
        """
        # Patterns that indicate reference to an image
        image_reference_patterns = [
            r"(?i)(?:the|this|that|previous|last|first|second|recent|earlier|above|shared|sent|uploaded) image",
            r"(?i)(?:image|photo|picture) (?:you|i|we|they) (?:saw|see|shared|showed|uploaded|sent)",
            r"(?i)(?:what|tell me|describe) (?:about|is in|do you see in) (?:the|this|that) (?:image|photo|picture)",
            r"(?i)what (?:is|was) (?:in|on) (?:the|this|that) (?:image|photo|picture)",
            r"(?i)(?:can you|could you|please) (?:tell me more|explain|describe) (?:about|what's in) (?:the|this|that) (?:image|photo|picture)",
            r"(?i)(?:the|this|that) (?:image|photo|picture) (?:i|we|you) (?:shared|sent|uploaded|talked about)",
            r"(?i)(?:remember|recall) (?:the|this|that) (?:image|photo|picture)",
        ]

        for pattern in image_reference_patterns:
            if re.search(pattern, message):
                return True

        return False

    async def get_image_context(self, user_data: Dict[str, Any]) -> str:
        """Generate context from previously processed images."""
        if "image_history" not in user_data or not user_data["image_history"]:
            return ""

        # Get the 3 most recent images
        recent_images = user_data["image_history"][-3:]

        image_context = "Recently analyzed images that you can refer to:\n"
        for idx, img in enumerate(recent_images):
            timestamp = img.get("timestamp", "")
            time_str = f" (shared on {timestamp})" if timestamp else ""

            image_context += f"[Image {idx + 1}]{time_str}: Caption: {img['caption']}\n"
            # Include more complete description for better context
            full_desc = img.get("description", "")
            # Use full description if it's reasonably sized, otherwise truncate
            desc_text = full_desc if len(full_desc) < 300 else f"{full_desc[:250]}..."
            image_context += f"Description: {desc_text}\n\n"

        # Add explicit instruction about referring to images
        image_context += "You can refer to these images in your responses when the user asks about them."

        return image_context

    async def get_document_context(self, user_data: Dict[str, Any]) -> str:
        """Generate context from previously processed documents."""
        if "document_history" not in user_data or not user_data["document_history"]:
            return ""

        # Get the most recent document
        most_recent = user_data["document_history"][-1]

        document_context = f"Recently analyzed document: {most_recent['file_name']}\n\n"
        document_context += f"Full content summary:\n{most_recent['full_response']}\n\n"
        document_context += "Please provide additional details or answer follow-up questions about this document."

        return document_context
