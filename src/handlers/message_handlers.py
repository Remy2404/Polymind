import os , io
import tempfile
import logging
import speech_recognition as sr
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from pydub import AudioSegment
from handlers.text_handlers import TextHandler
from services.user_data_manager import UserDataManager
from telegram.ext import MessageHandler, filters
import datetime
from services.gemini_api import GeminiAPI
import asyncio


class MessageHandlers:
    def __init__(self, gemini_api, user_data_manager, telegram_logger , document_processor ,text_handler):
        self.gemini_api = gemini_api
        self.text_handler = text_handler
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.document_processor = document_processor
        self.logger = logging.getLogger(__name__)


    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                """Handle incoming text messages."""
                try:
                    if update.message is None and update.callback_query is None:
                        self.logger.error("Received update with no message or callback query")
                        return

                    if update.callback_query:
                        user_id = update.callback_query.from_user.id
                        message_text = update.callback_query.data
                        await update.callback_query.answer()
                    else:
                        user_id = update.effective_user.id
                        message_text = update.message.text

                    self.logger.info(f"Received text message from user {user_id}: {message_text}")

                    # Check if the bot is mentioned
                    bot_username = "@Gemini_AIAssistBot"
                    if bot_username in message_text:
                        self.logger.info(f"Bot mentioned by user {user_id}")
                        if update.callback_query:
                            await update.callback_query.edit_message_text("Hello! How can I assist you today?")
                        else:
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

            # Check if the bot is mentioned in the image caption
            bot_username = "@Gemini_AIAssistBot"
            if update.message.caption and bot_username in update.message.caption:
                self.logger.info(f"Bot mentioned by user {user_id} in image caption")
                await update.message.reply_text("I see you sent an image mentioning me. How can I assist you?")

            # Initialize user data if not already initialized
            self.user_data_manager.initialize_user(user_id)
            
            # Create text handler instance (which also handles images)
            text_handler = TextHandler(self.gemini_api, self.user_data_manager)

            # Process the image
            await text_handler.handle_image(update, context)
            await self.user_data_manager.update_user_stats(user_id, {'images': 1, 'total_messages': 1})
        except Exception as e:
            self.logger.error(f"Error processing image message: {str(e)}")
            await self._error_handler(update, context)
        self.user_data_manager.update_stats(user_id, image=True)
            

    async def _handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming voice messages."""
        user_id = update.effective_user.id
        self.telegram_logger.log_message(user_id, "Received voice message")
        
        try:
            await update.message.reply_text("I'm processing your voice message. Please wait...")
        
            with tempfile.TemporaryDirectory() as temp_dir:
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
                text_handler = TextHandler(self.gemini_api, self.user_data_manager)
        
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
                await self.user_data_manager.update_user_stats(user_id, {'voice_messages': 1, 'total_messages': 1})
    
        except sr.UnknownValueError:
            await update.message.reply_text("Sorry, I couldn't understand the audio. Could you please try again?")
        except sr.RequestError as e:
            self.logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            await update.message.reply_text("Sorry, there was an error processing your voice message. Please try again later.")
        except Exception as e:
            self.logger.error(f"Error processing voice message: {str(e)}")
            await self._error_handler(update, context)

    async def _handle_document_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming document messages."""
        user_id = update.effective_user.id
        self.logger.info(f"Processing document for user: {user_id}")

        try:
            document = update.message.document
            file = await context.bot.get_file(document.file_id)
            file_extension = document.file_name.split('.')[-1]

            response = await self.document_processor.process_document_from_file(
                file=await file.download_as_bytearray(),
                file_extension=file_extension,
                prompt="Analyze this document."
            )

            formatted_response = await self.text_handler.format_telegram_markdown(response)
            await update.message.reply_text(
                formatted_response,
                parse_mode='MarkdownV2',
                disable_web_page_preview=True
            )

            self.user_data_manager.update_stats(user_id, document=True)
            self.telegram_logger.log_message("Document processed successfully.", user_id)

        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            if "RATE_LIMIT_EXCEEDED" in str(e).upper():
                await update.message.reply_text(
                    "The service is currently experiencing high demand. Please try again later."
                )
            else:
                await self._error_handler(update, context)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        self.telegram_logger.log_message("Processing document", user_id)

        try:
            # Check if the message is in a group chat
            if update.effective_chat.type in ['group', 'supergroup']:
                # Process only if the bot is mentioned in the caption
                bot_username = '@' + context.bot.username
                caption = update.message.caption or ""
                if bot_username not in caption:
                    return
                else:
                    # Remove bot mention
                    caption = caption.replace(bot_username, '').strip()
            else:
                caption = update.message.caption or "Please analyze this document."
            
            # Get basic document information
            document = update.message.document
            file_name = document.file_name
            file_id = document.file_id
            file_extension = os.path.splitext(file_name)[1][1:] if '.' in file_name else ''
            
            # Send typing action and status message
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            status_message = await update.message.reply_text(
                f"Processing your {file_extension.upper()} document... This might take a moment."
            )
            
            # Download and process the document
            document_file = await context.bot.get_file(file_id)
            file_content = await document_file.download_as_bytearray()
            document_file_obj = io.BytesIO(file_content)
            
            # Default prompt if caption is empty
            prompt = caption or f"Please analyze this {file_extension.upper()} file and provide a detailed summary."
            
            # Use enhanced document processing for PDFs
            if file_extension.lower() == 'pdf':
                response = await self.document_processor.process_document_enhanced(
                    file=document_file_obj,
                    file_extension=file_extension,
                    prompt=prompt
                )
            else:
                response = await self.document_processor.process_document_from_file(
                    file=document_file_obj,
                    file_extension=file_extension,
                    prompt=prompt
                )
            
            # Delete status message
            await status_message.delete()
            
            if response:
                # Split long messages
                response_chunks = await self.text_handler.split_long_message(response)
                sent_messages = []
                
                # Send each chunk
                for chunk in response_chunks:
                    try:
                        formatted_chunk = await self.text_handler.format_telegram_markdown(chunk)
                        sent_message = await update.message.reply_text(
                            formatted_chunk,
                            parse_mode='MarkdownV2',
                            disable_web_page_preview=True
                        )
                        sent_messages.append(sent_message.message_id)
                    except Exception as markdown_error:
                        self.logger.warning(f"Markdown formatting failed: {markdown_error}")
                        sent_message = await update.message.reply_text(chunk, parse_mode=None)
                        sent_messages.append(sent_message.message_id)
                
                # Store document info in user context
                self.user_data_manager.add_to_context(
                    user_id, 
                    {"role": "user", "content": f"[Document: {file_name} with prompt: {prompt}]"}
                )
                self.user_data_manager.add_to_context(
                    user_id, 
                    {"role": "assistant", "content": response}
                )
                
                # Store document reference in user data (NEW)
                if 'document_history' not in context.user_data:
                    context.user_data['document_history'] = []
    
                # Store document info in user data (OLD)
                context.user_data['document_history'].append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'file_id': file_id,
                    'file_name': file_name, 
                    'file_extension': file_extension,
                    'prompt': prompt,
                    'summary': response[:300] + "..." if len(response) > 300 else response,
                    'full_response': response,  # Critical for follow-up questions
                    'message_id': update.message.message_id,
                    'response_message_ids': [msg.message_id for msg in sent_messages]
                })
                
                # Update user stats
                if self.user_data_manager:
                    self.user_data_manager.update_stats(user_id, document=True)
                
                self.telegram_logger.log_message(f"Document analysis completed successfully", user_id)
            else:
                await update.message.reply_text("Sorry, I couldn't analyze the document. Please try again.")
        
        except ValueError as ve:
            await update.message.reply_text(f"Error: {str(ve)}")
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            await update.message.reply_text(
                "Sorry, I couldn't process your document. Please ensure it's in a supported format."
            )

    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors occurring in the dispatcher."""
        self.logger.error(f"Update {update} caused error: {context.error}")
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "An error occurred while processing your request. Please try again later."
            )


    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors occurring in the dispatcher."""
        self.logger.error(f"Update {update} caused error: {context.error}")
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "An error occurred while processing your request. Please try again later."
            )
    def register_handlers(self, application):
        """Register message handlers with the application."""
        try:
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
            application.add_handler(MessageHandler(filters.PHOTO, self._handle_image_message))
            application.add_handler(MessageHandler(filters.VOICE, self._handle_voice_message))
            application.add_handler(MessageHandler(filters.Document.ALL, self._handle_document_message))

            application.add_error_handler(self._error_handler)
            self.logger.info("Message handlers registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register message handlers: {str(e)}")
            raise Exception("Failed to register message handlers") from e