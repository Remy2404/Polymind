import os , sys
import tempfile
import logging
from dotenv import load_dotenv
from telegram import Update, Message ,InputFile
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
import handlers.text_handlers as text_handlers

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
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')  # Changed to match your .env variable name
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

        # Verify Gemini API key is present
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY not found in .env file")

        # Initialize components
        self.gemini_api = GeminiAPI()
        self.user_data_manager = UserDataManager(self.db)
        self.telegram_logger = telegram_logger

    def shutdown(self):
        close_database_connection(self.client)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        help_text = (
            "Here's how to use me:\n\n"
            "1. Send text messages for AI-powered responses\n"
            "2. Send voice messages for transcription\n"
            "3. Send images for analysis\n\n"
            "Commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/reset - Reset our conversation\n"
            "/stats - View your usage statistics"
        )
        await update.message.reply_text(help_text)

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reset the conversation history when /reset is issued."""
        user_id = update.effective_user.id
        self.user_data_manager.clear_history(user_id)
        await update.message.reply_text("Conversation history has been reset!")

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages."""
        try:
            user_id = update.effective_user.id
            message_text = update.message.text
            self.logger.info(f"Received text message from user {user_id}: {message_text}")

            # Check if the bot is mentioned
            bot_username = context.bot.username
            if f"@{bot_username}" in message_text:
                self.logger.info(f"Bot mentioned by user {user_id}")
                await update.message.reply_text("Hello! How can I assist you today?")

            # Initialize user data if not already initialized
            self.user_data_manager.initialize_user(user_id)

            # Create text handler instance
            text_handler = text_handlers.TextHandler(self.gemini_api, self.user_data_manager)

            # Process the message
            await text_handler.handle_text_message(update, context)
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
            await text_handler.handle_image(update, context)

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


    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors occurring in the dispatcher."""
        self.logger.error(f"Update {update} caused error: {context.error}")
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "An error occurred while processing your request. Please try again later."
            )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            user_id = update.effective_user.id
            stats = self.user_data_manager.get_user_stats(user_id)
            if stats:
                await update.message.reply_text(
                    f"Here are your stats:\n"
                    f"• Total Messages Sent: {stats.get('total_messages', 0)}\n"
                    f"• Text Messages: {stats.get('text_messages', 0)}\n"
                    f"• Voice Messages: {stats.get('voice_messages', 0)}\n"
                    f"• Images Sent: {stats.get('images', 0)}"
                )
            else:
                await update.message.reply_text("No statistics available yet.")
        except Exception as e:
            self.logger.error(f"Error fetching user stats: {str(e)}")
            await self._error_handler(update, context)

    async def broadcast_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Broadcast a message to all users (admin only)."""
        try:
            admin_user_id = int(os.getenv('ADMIN_USER_ID', '0'))
            if update.effective_user.id != admin_user_id:
                await update.message.reply_text("You are not authorized to use this command.")
                return

            if not context.args:
                await update.message.reply_text("Please provide a message to broadcast.")
                return

            broadcast_message = ' '.join(context.args)
            all_users = self.user_data_manager.get_all_user_ids()
            successful_sends = 0
            for user_id in all_users:
                try:
                    await context.bot.send_message(chat_id=user_id, text=broadcast_message)
                    successful_sends += 1
                except Exception as e:
                    self.logger.error(f"Failed to send message to user {user_id}: {str(e)}")

            await update.message.reply_text(f"Broadcast message sent successfully to {successful_sends} users.")
        except Exception as e:
            self.logger.error(f"Error during broadcast: {str(e)}")
            await self._error_handler(update, context)
    def run(self) -> None:
        """Start the bot."""
        try:
            application = Application.builder().token(self.token).build()

            application.add_handler(CommandHandler('start', self.stats_command))
            application.add_handler(CommandHandler('help', self.help_command))
            application.add_handler(CommandHandler('reset', self.reset_command))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
            application.add_handler(MessageHandler(filters.VOICE, self._handle_voice_message))
            application.add_handler(MessageHandler(filters.PHOTO, self._handle_image_message))
            application.add_handler(CommandHandler('stats', self.stats_command))
            application.add_handler(CommandHandler(
                'broadcast',
                self.broadcast_command,
                filters=filters.User(int(os.getenv('ADMIN_USER_ID', '0')))
            ))

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