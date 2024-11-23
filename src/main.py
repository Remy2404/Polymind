import os, sys
import tempfile
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from pydub import AudioSegment
import speech_recognition as sr
from database.connection import get_database, close_database_connection
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from utils.telegramlog import telegram_logger
from utils.pdf_handler import PDFHandler
import handlers.text_handlers as text_handlers
from handlers.text_handlers import TextHandler
from handlers.command_handlers import CommandHandlers
from telegram.ext import ConversationHandler

# Load environment variables
load_dotenv()

# Update logging configuration to handle Unicode
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler(sys.stdout)]
)

file_handler = logging.FileHandler('your_log_file.log', encoding='utf-8')
logging.getLogger().addHandler(file_handler)

class TelegramBot:
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Establish database connection
        self.db, self.client = get_database()
        if self.db is None:
            self.logger.error("Failed to connect to the database")
            raise ConnectionError("Failed to connect to the database")
        self.logger.info("Connected to MongoDB successfully")

        # Get tokens from .env file
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

        # Verify Gemini API key is present
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY not found in .env file")

        # Initialize components
        self.gemini_api = GeminiAPI()
        self.user_data_manager = UserDataManager(self.db)
        self.telegram_logger = telegram_logger
        
        # Initialize command handler
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api, 
            user_data_manager=self.user_data_manager
        )
        # Initialize TextHandler
        self.text_handler = TextHandler(self.gemini_api, self.user_data_manager)
        
        # Initialize PDFHandler with TextHandler
        self.pdf_handler = PDFHandler(self.gemini_api, self.text_handler)

    def shutdown(self):
        close_database_connection(self.client)

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages."""
        try:
            user_id = update.effective_user.id
            self.telegram_logger.log_message(user_id, f"Received text message: {update.message.text}")

            # Initialize user data if not already initialized
            self.user_data_manager.initialize_user(user_id)
            # Create text handler instance
            text_handler = text_handlers.TextHandler(self.gemini_api, self.user_data_manager)

            # Process the message
            await self.text_handler.handle_text_message(update, context)

        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            await self._error_handler(update, context)

    async def _handle_image_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming image messages."""
        try:
            user_id = update.effective_user.id
            self.telegram_logger.log_message(user_id, "Received image message")

            # Initialize user data if not already initialized
            self.user_data_manager.initialize_user(user_id)
            
            # Create text handler instance (which also handles images)
            text_handler = text_handlers.TextHandler(self.gemini_api, self.user_data_manager)

            # Process the image
            await self.text_handler.handle_image(update, context)

        except Exception as e:
            self.logger.error(f"Error processing image message: {str(e)}")
            await self._error_handler(update, context)

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
                text_handler = text_handlers.TextHandler(self.gemini_api, self.user_data_manager)

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


    # In the TelegramBot class:
    async def _handle_pdf_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle PDF documents."""
        user_id = update.effective_user.id
        try:
            await self.pdf_handler.handle_pdf(update, context)
            await self.user_data_manager.update_stats(user_id, pdf_document=True)
        except Exception as e:
            self.logger.error(f"Error handling PDF: {str(e)}")
            await update.message.reply_text("An error occurred while processing your PDF. Please try again.")

    async def _handle_pdf_followup(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle follow-up questions about the PDF content."""
        try:
            await self.pdf_handler.handle_pdf_followup(update, context)
        except Exception as e:
            self.logger.error(f"Error handling PDF followup: {str(e)}")
            await update.message.reply_text("An error occurred while processing your question. Please try again.")


   
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors occurring in the dispatcher."""
        self.logger.error(f"Update {update} caused error: {context.error}")
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "An error occurred while processing your request. Please try again later."
            )

    def run(self) -> None:
        """Start the bot."""
        try:
            application = Application.builder().token(self.token).build()
    
            # Register command handlers
            self.command_handler.register_handlers(application)
    
            # Add message handlers for text, voice, and images
            for handler in self.text_handler.get_handlers():
                application.add_handler(handler)
    
            application.add_handler(MessageHandler(filters.VOICE, self._handle_voice_message))
            application.add_handler(MessageHandler(filters.Document.PDF, self._handle_pdf_document))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_pdf_followup))
            application.add_handler(CommandHandler("pdf_info", self.pdf_handler.handle_pdf_info))
            application.add_handler(self.pdf_handler.get_conversation_handler())
            application.add_handler(CommandHandler("history", self.text_handler.show_history))
    
            # Add global error handler
            application.add_error_handler(self._error_handler)
    
            self.logger.info("Starting bot")
            application.run_polling(allowed_updates=Update.ALL_TYPES)
    
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}")
            raise

if __name__ == '__main__':
    main_bot = TelegramBot()
    try:
        main_bot.run()
    finally:
        main_bot.shutdown()