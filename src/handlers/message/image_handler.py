"""
Image message handling.
"""
import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from src.handlers.text_handlers import TextHandler


class ImageMessageHandlerMixin:
    """Mixin for image message handling."""

    async def _handle_image_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming image messages using unified conversation system."""
        try:
            user_id = update.effective_user.id
            self.logger.info(f"Processing image from user {user_id}")
            self.telegram_logger.log_message("Received image message", user_id)

            if not update.message or not update.message.photo or len(update.message.photo) == 0:
                await update.message.reply_text("Sorry, I couldn't process this image.")
                return

            # Check if user's model supports images
            supports_images, model_id, error_message = await self.check_model_supports_media(user_id, "image")
            if not supports_images:
                await update.message.reply_text(error_message, parse_mode="Markdown")
                return

            # Route through TextHandler for unified conversation processing
            text_handler = TextHandler(
                self.gemini_api,
                self.user_data_manager,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )

            # Extract media files
            has_attached_media, media_files, media_type = await text_handler._extract_media_files(update, context)

            if has_attached_media and media_files:
                processing_message = await update.message.reply_text("Processing your image. Please wait...")
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

                try:
                    await text_handler._handle_media_analysis(
                        update,
                        context,
                        processing_message,
                        media_files,
                        media_type,
                        update.message.caption or "Analyze this image",
                        user_id,
                        model_id,
                    )
                    self.telegram_logger.log_message("Image processed successfully via unified system", user_id)

                except Exception as e:
                    self.logger.error(f"Image processing error: {str(e)}")
                    try:
                        await processing_message.edit_text(
                            "Sorry, there was an error processing your image. Please try again."
                        )
                    except Exception:
                        await update.message.reply_text(
                            "Sorry, there was an error processing your image. Please try again."
                        )
            else:
                await update.message.reply_text("Sorry, I couldn't extract the image data.")

        except Exception as e:
            self.logger.error(f"Error in image message handler: {str(e)}")
            await self._error_handler(update, context)
