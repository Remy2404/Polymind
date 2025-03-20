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
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Union
from functools import partial
import traceback
import gc
import time
import weakref
from telegram import Update, Message, Document
from telegram.ext import (
    MessageHandler, 
    filters, 
    ContextTypes, 
    CallbackContext
)
from src.services.gemini_api import GeminiAPI
from src.services.user_data_manager import user_data_manager
from src.utils.telegramlog import TelegramLogger
from src.services.document_processing import DocumentProcessor
from src.handlers.text_handlers import TextHandler

logger = logging.getLogger(__name__)

class MessageHandlers:
    def __init__(
        self, 
        gemini_api: GeminiAPI, 
        user_data_manager: user_data_manager,
        telegram_logger: TelegramLogger,
        document_processor: DocumentProcessor,
        text_handler: TextHandler
    ):
        """Initialize the MessageHandlers with required services."""
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.document_processor = document_processor
        self.text_handler = text_handler
        
        # Use a thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Active requests tracking with weak references to avoid memory leaks
        self.active_requests = weakref.WeakSet()
        self.request_limiter = asyncio.Semaphore(20)  # Limit concurrent requests
        
        logger.info("MessageHandlers initialized with optimized concurrency settings")
        
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle document uploads with improved memory management."""
        if not update.message or not update.message.document:
            return
            
        user_id = update.effective_user.id
        self.telegram_logger.log_message(f"Document received: {update.message.document.file_name}", user_id)
        
        async def process_document():
            try:
                # Acquire semaphore to limit concurrent processing
                async with self.request_limiter:
                    # Set typing status
                    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
                    
                    # First send acknowledgment message
                    status_message = await update.message.reply_text("Processing your document...")
                    
                    # Process document in background with timeout protection
                    try:
                        document = update.message.document
                        start_time = time.time()
                        
                        # Download file with timeout
                        file = await asyncio.wait_for(
                            context.bot.get_file(document.file_id),
                            timeout=30.0
                        )
                        
                        # Process document
                        file_bytes = await file.download_as_bytearray()
                        
                        # Get document content using the document processor
                        content = await self.document_processor.process_document(
                            file_bytes, 
                            document.file_name,
                            document.mime_type
                        )
                        
                        processing_time = time.time() - start_time
                        logger.info(f"Document processed in {processing_time:.2f}s: {document.file_name}")
                        
                        if content:
                            # Save document to user's history
                            await self.user_data_manager.save_document_to_history(
                                user_id, 
                                document.file_name, 
                                document.file_unique_id, 
                                document.mime_type, 
                                content[:1000]  # Store truncated preview
                            )
                            
                            # Send summary and generate response
                            await status_message.edit_text(f"📄 Document '{document.file_name}' processed successfully!")
                            
                            # Let the user know document content is available via history
                            instruction_msg = (
                                "I've saved your document content. You can now ask me questions about it, "
                                "and I'll analyze its contents."
                            )
                            await update.message.reply_text(instruction_msg)
                            
                        else:
                            await status_message.edit_text(
                                f"❌ Sorry, I couldn't process '{document.file_name}'. "
                                "The file may be too large, corrupted, or in an unsupported format."
                            )
                    except asyncio.TimeoutError:
                        await status_message.edit_text("⏱️ Document processing timed out. The file may be too large.")
                        logger.warning(f"Document processing timed out for user {user_id}: {document.file_name}")
            except Exception as e:
                error_message = f"Error processing document: {str(e)}"
                logger.error(error_message)
                logger.error(traceback.format_exc())
                self.telegram_logger.log_error(e, user_id)
                
                try:
                    await update.message.reply_text(
                        "Sorry, I couldn't process your document. Please try a different format or a smaller file."
                    )
                except Exception:
                    pass
            finally:
                # Force garbage collection to free memory from large documents
                gc.collect()
        
        # Create background task for processing
        task = asyncio.create_task(process_document())
        self.active_requests.add(task)
        
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle image messages with improved error handling and memory management."""
        if not update.message or not update.message.photo:
            return
            
        user_id = update.effective_user.id
        self.telegram_logger.log_message("Image received", user_id)
        
        async def process_image():
            try:
                async with self.request_limiter:
                    # Set typing action
                    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
                    
                    # Get the highest resolution image
                    photo = update.message.photo[-1]
                    
                    # Extract caption or use default prompt
                    caption = update.message.caption or "Analyze this image in detail"
                    
                    # Get image file with timeout
                    try:
                        file = await asyncio.wait_for(
                            context.bot.get_file(photo.file_id),
                            timeout=15.0  # 15 second timeout for file retrieval
                        )
                        
                        # Download image as bytes
                        image_bytes = await file.download_as_bytearray()
                        
                        # Process the image and generate a response with the gemini_api
                        response = await asyncio.wait_for(
                            self.gemini_api.analyze_image(image_bytes, caption),
                            timeout=45.0  # 45 second timeout for processing
                        )
                        
                        if response:
                            # Send response in chunks if needed
                            if len(response) > 4000:
                                for i in range(0, len(response), 4000):
                                    chunk = response[i:i+4000]
                                    await update.message.reply_text(chunk)
                            else:
                                await update.message.reply_text(response)
                                
                            # Save interaction to user history
                            await self.user_data_manager.save_image_analysis_to_history(
                                user_id,
                                photo.file_unique_id,
                                caption,
                                response[:500]  # Save truncated response
                            )
                        else:
                            await update.message.reply_text(
                                "Sorry, I couldn't analyze this image. Please try a different image."
                            )
                            
                    except asyncio.TimeoutError:
                        await update.message.reply_text("The operation timed out. Please try again with a smaller image.")
                        
            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                logger.error(error_message)
                logger.error(traceback.format_exc())
                self.telegram_logger.log_error(e, user_id)
                
                try:
                    await update.message.reply_text("Sorry, there was an error processing your image.")
                except Exception:
                    pass
            finally:
                # Clean up resources
                gc.collect()
                
        # Create background task for processing
        task = asyncio.create_task(process_image())
        self.active_requests.add(task)
        
    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors with detailed logging and graceful user communication."""
        try:
            if update and isinstance(update, Update) and update.effective_chat:
                user_id = update.effective_user.id if update.effective_user else 0
                chat_id = update.effective_chat.id
                
                # Log the error
                logger.error(f"Error for user {user_id}: {context.error}")
                logger.error(traceback.format_exc())
                self.telegram_logger.log_error(context.error, user_id)
                
                # Send user-friendly error message
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="Sorry, something went wrong processing your request. Please try again later."
                    )
                except Exception as send_error:
                    logger.error(f"Failed to send error message: {send_error}")
            else:
                logger.error(f"Update caused error without chat context: {context.error}")
                logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            logger.error(traceback.format_exc())
            
    def register_handlers(self, application) -> None:
        """Register all message handlers with the application."""
        # Handle documents (PDFs, DOCs, etc.)
        application.add_handler(MessageHandler(
            filters.Document.ALL & ~filters.COMMAND, 
            self.handle_document
        ))
        
        # Handle images
        application.add_handler(MessageHandler(
            filters.PHOTO & ~filters.COMMAND, 
            self.handle_image
        ))
        
    async def cleanup(self):
        """Clean up resources and cancel pending requests."""
        try:
            # Cancel all active tasks
            active_tasks = list(self.active_requests)
            if active_tasks:
                logger.info(f"Cancelling {len(active_tasks)} pending message handler tasks")
                for task in active_tasks:
                    if not task.done():
                        task.cancel()
                        
                # Wait for tasks to be cancelled
                await asyncio.gather(*active_tasks, return_exceptions=True)
                
            # Shutdown thread pool
            self.executor.shutdown(wait=False)
            logger.info("Message handlers cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during message handler cleanup: {e}")
            logger.error(traceback.format_exc())