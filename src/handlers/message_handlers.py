import os , io
import tempfile
import logging
import speech_recognition as sr
from pydub import AudioSegment
from telegram.constants import ChatAction
from handlers.text_handlers import TextHandler
from telegram.ext import MessageHandler, filters
from services.user_data_manager import UserDataManager
from telegram import Update
from services.gemini_api import GeminiAPI
from telegram import (
    Update, 
)
from telegram.ext import (
    ContextTypes,
    MessageHandler,
    filters
)
from utils.telegramlog import TelegramLogger as telegram_logger


class MessageHandlers:
    def __init__(self, gemini_api : GeminiAPI, user_data_manager : UserDataManager, telegram_logger : telegram_logger, pdf_handler):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.pdf_handler = pdf_handler
        self.logger = logging.getLogger(__name__)


    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages."""
        try:
            user_id = update.effective_user.id
            message_text = update.message.text
            self.logger.info(f"Received text message from user {user_id}: {message_text}")

            # Check if the bot is mentioned
            bot_username = "@Gemini_AIAssistBot"
            if bot_username in message_text:
                self.logger.info(f"Bot mentioned by user {user_id}")
                await update.message.reply_text("Hello! How can I assist you today?")

            # Initialize user data if not already initialized
            self.user_data_manager.initialize_user(user_id)

            # Create text handler instance
            text_handler = TextHandler(self.gemini_api, self.user_data_manager)

            # Process the message
            await text_handler.handle_text_message(update, context)
            await self.user_data_manager.update_user_stats(user_id, {'text_messages': 1, 'total_messages': 1})
        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            await self._error_handler(update, context)
        self.user_data_manager.update_stats(user_id, text_message=True)

    async def _handle_image_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming image messages."""
        try:
            user_id = update.effective_user.id
            self.telegram_logger.log_message(user_id, "Received image message")

            # Show upload photo action
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)

            # Check if the bot is mentioned in the caption
            if update.message.caption and "@Gemini_AIAssistBot" in update.message.caption:
                self.logger.info(f"Bot mentioned by user {user_id} in image caption")
                
                # Show typing action
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
                await update.message.reply_text("Hello! How can I assist you with this image?")

            # Initialize user data if not already initialized 
            self.user_data_manager.initialize_user(user_id)
            
            # Create text handler instance (which also handles images)
            text_handler = TextHandler(self.gemini_api, self.user_data_manager)

            # Process the image
            await text_handler.handle_image(update, context)

        except Exception as e:
            self.logger.error(f"Error processing image message: {str(e)}")
            await self._error_handler(update, context)
            self.user_data_manager.update_stats(user_id, image_message=True)

    async def _handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming voice messages."""
        user_id = update.effective_user.id
        self.telegram_logger.log_message(user_id, "Received voice message")

        await update.message.reply_text("I'm processing your voice message. Please wait...")

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Download the voice file
                file = await context.bot.get_file(update.message.voice.file_id)
                ogg_file_path = os.path.join(temp_dir, f"{user_id}_voice.ogg")
                await file.download_to_drive(ogg_file_path)

                # Convert OGG to WAV
                wav_file_path = os.path.join(temp_dir, f"{user_id}_voice.wav")
                audio = AudioSegment.from_ogg(ogg_file_path)
                audio.export(wav_file_path, format="wav")

                # Convert the voice file to text
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_file_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)

                # Log the transcribed text
                self.telegram_logger.log_message(user_id, f"Transcribed text: {text}")

                # Initialize user data if not already initialized
                self.user_data_manager.initialize_user(user_id)

                # Create text handler instance
                text_handler =TextHandler(self.gemini_api, self.user_data_manager)

                # Create a new Update object with the transcribed text
                new_update = Update.de_json({
                    'update_id': update.update_id,
                    'message': {
                        'message_id': update.message.message_id,
                        'date': update.message.date.timestamp(),
                        'chat': update.message.chat.to_dict(),
                        'from': update.message.from_user.to_dict(),
                        'text': text
                    }
                }, context.bot)

                # Process the transcribed text as if it were a regular text message
                await text_handler.handle_text_message(new_update, context)

            except sr.UnknownValueError:
                await update.message.reply_text("Sorry, I couldn't understand the audio. Could you please try again?")
            except sr.RequestError as e:
                self.logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                await update.message.reply_text("Sorry, there was an error processing your voice message. Please try again later.")
            except Exception as e:
                self.logger.error(f"Error processing voice message: {str(e)}")
                await update.message.reply_text("An error occurred while processing your voice message. Please try again.")

        # Update user stats
        self.user_data_manager.update_stats(user_id, voice_message=True)
   
    async def _handle_pdf_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle PDF documents."""
        user_id = update.effective_user.id
        try:
            self.logger.info(f"Received PDF document from user {user_id}")
            
            if not update.message.document or update.message.document.mime_type != 'application/pdf':
                await update.message.reply_text("Please send a valid PDF file.")
                return

            await self.user_data_manager.initialize_user(user_id)
            
            file = await context.bot.get_file(update.message.document.file_id)
            file_content = io.BytesIO()
            await file.download(out=file_content)
            file_content.seek(0)

            # Process the PDF using the pdf_handler
            await self.pdf_handler.handle_pdf(update, context)
            
            await self.user_data_manager.update_stats(user_id, pdf_document=True)
            await update.message.reply_text("PDF received and processed. You can now ask questions about it.")
        except Exception as e:
            self.logger.error(f"Error handling PDF: {str(e)}")
            await update.message.reply_text("An error occurred while processing your PDF. Please try again.")

  
    async def _handle_edited_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle edited messages sent by the bot."""
        try:
            edited_message = update.edited_message
            if not edited_message:
                return  # Not an edited message

            user_id = edited_message.from_user.id
            chat_id = edited_message.chat_id
            message_id = edited_message.message_id

            self.logger.info(f"Received edited message from user {user_id} in chat {chat_id}")

            # Check if the edited message has text or caption
            if edited_message.text:
                edited_content = edited_message.text
            elif edited_message.caption:
                edited_content = edited_message.caption
            else:
                self.logger.warning(f"Edited message {message_id} has no text or caption")
                return

            # Delete the previous message if it was sent by the bot
            await context.bot.delete_message(chat_id=chat_id, message_id=message_id)

            # Send a new response with the updated content
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"I noticed you edited your message. The new content is: {edited_content}"
            )
        except Exception as e:
            self.logger.error(f"Error handling edited message: {str(e)}")
            await self._error_handler(update, context)
                # You might want to send an error message to the user here
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle errors occurring in the dispatcher."""
            self.logger.error(f"Update {update} caused error: {context.error}")
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "An error occurred while processing your request. Please try again later. 🥹💔"

                )
    async def some_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /some_command command."""
        user_id = update.effective_user.id
        self.logger.info(f"Received /some_command from user {user_id}")
        await update.message.reply_text("You triggered /some_command! 🚀")
        await self.user_data_manager.update_user_data(user_id, {'some_command': 1})
    def register_handlers(self, application):
        """Register message handlers with the application."""
        try:
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
            application.add_handler(MessageHandler(filters.PHOTO, self._handle_image_message))
            application.add_handler(MessageHandler(filters.VOICE, self._handle_voice_message))
            application.add_handler(MessageHandler(filters.Document.MimeType('application/pdf'), self._handle_pdf_document))
            application.add_handler(MessageHandler(filters.UpdateType.EDITED_MESSAGE, self._handle_edited_message))
            application.add_error_handler(self._error_handler)
            self.logger.info("Message handlers registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register message handlers: {str(e)}")
            raise