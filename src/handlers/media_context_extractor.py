import logging
from typing import Dict, Any, List, Optional

class MediaContextExtractor:
    """Extracts context from previously shared media messages."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def get_image_context(self, user_data: Dict[str, Any]) -> str:
        """Generate context from previously processed images."""
        if "image_history" not in user_data or not user_data["image_history"]:
            return ""

        # Get the 3 most recent images
        recent_images = user_data["image_history"][-3:]

        image_context = "Recently analyzed images:\n"
        for idx, img in enumerate(recent_images):
            image_context += f"[Image {idx+1}]: Caption: {img['caption']}\nDescription: {img['description'][:100]}...\n\n"

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