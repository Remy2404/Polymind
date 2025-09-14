import logging
from typing import Dict, Any
from datetime import datetime


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def format_response(data: Dict[str, Any]) -> str:
    """Format API response for telegram message"""
    return "\n".join([f"{k}: {v}" for k, v in data.items()])


def validate_image(image_data: bytes) -> bool:
    """Validate image data before processing"""
    return len(image_data) > 0


def track_usage(user_id: int, command: str):
    """Track user command usage"""
    timestamp = datetime.now().isoformat()
    logging.info(f"User {user_id} used command {command} at {timestamp}")
