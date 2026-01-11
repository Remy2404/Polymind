"""
Image generation command handlers.
Contains advanced image generation and Together AI image generation commands.
"""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
import logging
import io
import time
import asyncio
from datetime import datetime, timedelta
from cachetools import TTLCache
from typing import Optional
from PIL import Image
from dataclasses import dataclass, field


@dataclass
class ImageRequest:
    prompt: str
    width: int
    height: int
    steps: int
    timestamp: float = field(default_factory=time.time)


class ImageGenerationHandler:
    def __init__(self):
        self.request_cache = TTLCache(maxsize=100, ttl=3600)
        self.request_limiter = {}
        self.processing_queue = asyncio.Queue()
        self.rate_limit_time = 30

    def is_rate_limited(self, user_id: int) -> bool:
        if user_id in self.request_limiter:
            last_request = self.request_limiter[user_id]
            if datetime.now() - last_request < timedelta(seconds=self.rate_limit_time):
                return True
        return False

    def update_rate_limit(self, user_id: int) -> None:
        self.request_limiter[user_id] = datetime.now()

    def get_cached_image(
        self, prompt: str, width: int, height: int, steps: int
    ) -> Optional[Image.Image]:
        cache_key = f"{prompt}_{width}_{height}_{steps}"
        return self.request_cache.get(cache_key)

    def cache_image(
        self, prompt: str, width: int, height: int, steps: int, image: Image.Image
    ) -> None:
        cache_key = f"{prompt}_{width}_{height}_{steps}"
        self.request_cache[cache_key] = image


class ImageCommands:
    def __init__(
        self,
        flux_lora_image_generator,
        user_data_manager,
        telegram_logger,
        image_handler,
    ):
        self.flux_lora_image_generator = flux_lora_image_generator
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.image_handler = image_handler
        self.logger = logging.getLogger(__name__)

    async def generate_together_image(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /genimg command for image generation using Together AI."""
        user_id = update.effective_user.id
        self.telegram_logger.log_message(
            "Together AI image generation requested", user_id
        )
        if not context.args:
            await update.message.reply_text(
                "Please provide a description for the image you want to generate.\n"
                "Example: `/genimg a sunset over a calm lake with mountains in the background`",
                parse_mode="Markdown",
            )
            return
        prompt = " ".join(context.args)
        status_message = await update.message.reply_text(
            "🎨 Generating image with Together AI... This may take a moment."
        )
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
        )
        try:
            from services.together_ai_img import together_ai_image_generator

            image = await together_ai_image_generator.generate_image(
                prompt=prompt, num_steps=4, width=1024, height=1024
            )
            if image:
                await status_message.delete()
                with io.BytesIO() as output:
                    image.save(output, format="PNG")
                    output.seek(0)
                    await update.message.reply_photo(
                        photo=output,
                        caption=f"🖼️ Generated image based on: '{prompt}'",
                        parse_mode="Markdown",
                    )
                if self.user_data_manager:
                    self.user_data_manager.update_stats(user_id, image_generation=True)
            else:
                await status_message.edit_text(
                    "❌ Sorry, I couldn't generate the image. Please try a different description or try again later."
                )
        except Exception as e:
            self.telegram_logger.log_error(f"Image generation error: {str(e)}", user_id)
            await status_message.edit_text(
                "❌ An error occurred while generating your image. Please try again later."
            )
