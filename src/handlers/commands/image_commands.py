"""
Image generation command handlers.
Contains advanced image generation and Together AI image generation commands.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
import logging
import io


class ImageCommands:
    def __init__(self, flux_lora_image_generator, user_data_manager, telegram_logger, image_handler):
        self.flux_lora_image_generator = flux_lora_image_generator
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.image_handler = image_handler
        self.logger = logging.getLogger(__name__)

    async def generate_image_advanced(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /imagen3 command for advanced image generation."""
        user_id = update.effective_user.id
        self.telegram_logger.log_message("Imagen 3 image generation requested", user_id)

        if not context.args:
            await update.message.reply_text(
                "Please provide a description for the image you want to generate.\n"
                "Example: `/imagen3 a surreal landscape with floating islands and waterfalls`",
                parse_mode="Markdown",
            )
            return

        # Join all arguments to form the prompt
        prompt = " ".join(context.args)

        # Send a status message
        status_message = await update.message.reply_text(
            "Generating image with AI... This may take a moment."
        )

        try:
            # Use the correct method name: text_to_image instead of generate_images
            images = await self.flux_lora_image_generator.text_to_image(
                prompt=prompt,
                num_images=1,
                num_inference_steps=30,  # Higher quality setting
                width=768,
                height=768,
                guidance_scale=7.5,
            )

            if images and len(images) > 0:
                # Delete the status message
                await status_message.delete()

                # Convert PIL Image to bytes
                with io.BytesIO() as output:
                    images[0].save(output, format="PNG")
                    output.seek(0)
                    image_bytes = output.getvalue()

                # Send the generated image
                await update.message.reply_photo(
                    photo=image_bytes,
                    caption=f"Generated image based on: '{prompt}'",
                    parse_mode="Markdown",
                )

                # Update user stats
                if self.user_data_manager:
                    self.user_data_manager.update_stats(user_id, image_generation=True)
            else:
                await status_message.edit_text(
                    "Sorry, I couldn't generate that image. Please try a different description or try again later."
                )
        except Exception as e:
            self.logger.error(f"Image generation error: {str(e)}")
            await status_message.edit_text(
                "Sorry, there was an error generating your image. Please try a different description."
            )

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

        # Join all arguments to form the prompt
        prompt = " ".join(context.args)

        # Send a status message
        status_message = await update.message.reply_text(
            "🎨 Generating image with Together AI... This may take a moment."
        )

        # Send typing action to indicate processing
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
        )

        try:
            # Import the generator
            from services.together_ai_img import together_ai_image_generator

            # Generate the image
            image = await together_ai_image_generator.generate_image(
                prompt=prompt, num_steps=4, width=1024, height=1024
            )

            if image:
                # Delete the status message
                await status_message.delete()

                # Send the generated image
                with io.BytesIO() as output:
                    image.save(output, format="PNG")
                    output.seek(0)

                    await update.message.reply_photo(
                        photo=output,
                        caption=f"🖼️ Generated image based on: '{prompt}'",
                        parse_mode="Markdown",
                    )

                # Update user stats if available
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

    async def handle_image_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
    ) -> None:
        """Handle image-related callback settings"""
        query = update.callback_query
        await query.answer()
        
        # Placeholder implementation for image settings
        await query.edit_message_text(
            "Image settings not implemented yet."
        )
