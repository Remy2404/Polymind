import io
import os
import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from src.utils.log.telegramlog import telegram_logger
from src.services.gemini_api import GeminiAPI
from src.services.user_data_manager import UserDataManager
import asyncio
from .message_context_handler import MessageContextHandler
from .response_formatter import ResponseFormatter
from src.services.memory_context.memory_manager import MemoryManager
from src.services.memory_context.model_history_manager import ModelHistoryManager
from src.services.model_handlers.factory import ModelHandlerFactory
from src.services.model_handlers.prompt_formatter import PromptFormatter
from src.services.memory_context.conversation_manager import ConversationManager
from .text_processing.media_analyzer import MediaAnalyzer
from .text_processing.utilities import MediaUtilities
from .model_fallback_handler import ModelFallbackHandler
from src.services.ai_command_router import EnhancedIntentDetector
from src.services.mcp_bot_integration import (
    generate_mcp_response,
    is_model_mcp_compatible,
)
from src.utils.bot_username_helper import BotUsernameHelper
from .document_sender import DocumentSender
class TextHandler:
    def __init__(
        self,
        gemini_api: GeminiAPI,
        user_data_manager: UserDataManager,
        openrouter_api=None,
        deepseek_api=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.openrouter_api = openrouter_api
        self.deepseek_api = deepseek_api
        self.max_context_length = 9
        self.memory_manager = MemoryManager(
            db=user_data_manager.db if hasattr(user_data_manager, "db") else None,
        )
        self.memory_manager.short_term_limit = 15
        self.memory_manager.token_limit = 8192
        self.model_history_manager = ModelHistoryManager(self.memory_manager)
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()
        self.prompt_formatter = PromptFormatter()
        self.conversation_manager = ConversationManager(
            self.memory_manager, self.model_history_manager
        )
        self.media_analyzer = MediaAnalyzer(gemini_api, openrouter_api)
        self.model_fallback_handler = ModelFallbackHandler(self.response_formatter)
        self.intent_detector = EnhancedIntentDetector()
        self.user_model_manager = None
        
        # Initialize document sender for MCP-generated documents
        self.document_sender = DocumentSender()
    class MockMessage:
        def __init__(self, bot, chat_id):
            self.bot = bot
            self.chat_id = chat_id
        async def reply_text(self, text, **kwargs):
            return await self.bot.send_message(
                chat_id=self.chat_id, text=text, **kwargs
            )
    async def handle_text_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Main handler for text messages.
        Processes user messages, detects intent, and generates appropriate responses.
        """
        if not update.message and not update.edited_message:
            return
        user_id = update.effective_user.id
        message = update.message or update.edited_message
        message_text = message.text
        # --- NEW: Save user question to history ---
        if "user_questions" not in context.user_data:
            context.user_data["user_questions"] = []
        # Only save non-empty, non-command messages
        if message_text and not message_text.startswith("/"):
            context.user_data["user_questions"].append(message_text)
            # Limit to last 10 questions
            context.user_data["user_questions"] = context.user_data["user_questions"][-10:]
        # --- NEW: Detect "previous question" intent ---
        if self._is_previous_question_intent(message_text):
            prev_questions = context.user_data.get("user_questions", [])
            if prev_questions:
                response = "Here are your last questions:\n\n" + "\n".join(
                    [f"{i+1}. {q}" for i, q in enumerate(prev_questions[:-1])]
                )
            else:
                response = "I don't have any previous questions from you yet."
            await self.response_formatter.safe_send_message(message, response)
            return
        chat = update.effective_chat
        is_group = chat and chat.type in ["group", "supergroup"]
        if (
            is_group
            and hasattr(self, "_group_chat_integration")
            and self._group_chat_integration
        ):
            enhanced_message = await self._group_chat_integration.process_message(
                update, context
            )
            if enhanced_message:
                if enhanced_message.get("enhanced_text"):
                    message_text = enhanced_message["enhanced_text"]
        if "enhanced_message" in context.user_data:
            enhanced_message_text = context.user_data["enhanced_message"]
            group_metadata = context.user_data.get("group_context", {})
            if update.effective_chat.type in ["group", "supergroup"]:
                message_text = enhanced_message_text
                self.logger.info("Using enhanced group message for processing")
        quoted_text, quoted_message_id = self.context_handler.extract_reply_context(
            message
        )
        conversation_id = f"user_{user_id}"
        try:
            if update.edited_message and "bot_messages" in context.user_data:
                await self._handle_edited_message(update, context)
            if update.effective_chat.type in ["group", "supergroup"]:
                entities = []
                if message.entities:
                    entities = [
                        {
                            "type": entity.type,
                            "offset": entity.offset,
                            "length": entity.length,
                            "url": getattr(entity, "url", None),
                            "user": getattr(entity, "user", None),
                        }
                        for entity in message.entities
                    ]
                self.logger.debug(f"Group chat message entities: {entities}")
                if not BotUsernameHelper.is_bot_mentioned(
                    message_text, context, entities=entities
                ):
                    self.logger.debug(
                        f"Bot not mentioned in group chat message: '{message_text}'"
                    )
                    return
                else:
                    self.logger.debug("Bot mentioned in group chat, processing message")
                    message_text = BotUsernameHelper.remove_bot_mention(
                        message_text, context
                    )
            (
                has_attached_media,
                media_files,
                media_type,
            ) = await self._extract_media_files(update, context)
            thinking_message = await message.reply_text("Processing your request...🧠")
            await self._send_appropriate_chat_action(
                update, context, has_attached_media, media_type
            )
            intent_result = await self.intent_detector.detect_intent(message_text)
            user_intent = (intent_result.intent, intent_result.confidence)
            preferred_model = await self._get_user_preferred_model(user_id)
            await self.memory_manager.extract_and_save_user_info(user_id, message_text)
            history_context = await self.conversation_manager.get_conversation_history(
                user_id, max_messages=self.max_context_length, model=preferred_model
            )
            user_context = await self._load_user_context(user_id, update)
            if user_context and history_context:
                user_context_message = {
                    "role": "system",
                    "content": f"User information: {user_context}",
                }
                history_context.insert(0, user_context_message)
            if has_attached_media and user_intent == "analyze":
                await self._handle_media_analysis(
                    update,
                    context,
                    thinking_message,
                    media_files,
                    media_type,
                    message_text,
                    user_id,
                    preferred_model,
                )
                return
            await self._handle_text_conversation(
                update,
                context,
                thinking_message,
                message_text,
                quoted_text,
                quoted_message_id,
                history_context,
                user_id,
                preferred_model,
            )
        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            if "thinking_message" in locals() and thinking_message is not None:
                await thinking_message.delete()
            await self.response_formatter.safe_send_message(
                update.message, "Sorry, I encountered an error. Please try again later."
            )
    async def _handle_edited_message(self, update, context):
        """Handle when a user edits their previous message"""
        original_message_id = update.edited_message.message_id
        if original_message_id in context.user_data["bot_messages"]:
            for msg_id in context.user_data["bot_messages"][original_message_id]:
                if msg_id:
                    try:
                        await context.bot.delete_message(
                            chat_id=update.effective_chat.id, message_id=msg_id
                        )
                    except Exception as e:
                        self.logger.error(f"Error deleting old message: {str(e)}")
            del context.user_data["bot_messages"][original_message_id]
    async def _extract_media_files(self, update, context):
        """Extract media files from the update"""
        has_attached_media = False
        media_files = []
        media_type = None
        if update.message:
            if update.message.photo:
                has_attached_media = True
                media_type = "photo"
                photo = update.message.photo[-1]
                photo_file = await context.bot.get_file(photo.file_id)
                photo_bytes = await photo_file.download_as_bytearray()
                media_files.append(
                    {
                        "type": "photo",
                        "data": io.BytesIO(photo_bytes),
                        "mime": "image/jpeg",
                        "filename": f"photo_{photo.file_id}.jpg",
                    }
                )
            elif update.message.video:
                has_attached_media = True
                media_type = "video"
                video = update.message.video
                video_file = await context.bot.get_file(video.file_id)
                video_bytes = await video_file.download_as_bytearray()
                media_files.append(
                    {
                        "type": "video",
                        "data": io.BytesIO(video_bytes),
                        "mime": "video/mp4",
                        "filename": (
                            video.file_name
                            if hasattr(video, "file_name")
                            else f"video_{video.file_id}.mp4"
                        ),
                    }
                )
            elif update.message.voice or update.message.audio:
                has_attached_media = True
                media_type = "audio"
                audio = update.message.voice or update.message.audio
                audio_file = await context.bot.get_file(audio.file_id)
                audio_bytes = await audio_file.download_as_bytearray()
                file_name = (
                    getattr(audio, "file_name", None) or f"audio_{audio.file_id}.ogg"
                )
                media_files.append(
                    {
                        "type": "audio",
                        "data": io.BytesIO(audio_bytes),
                        "mime": "audio/ogg",
                        "filename": file_name,
                    }
                )
            elif update.message.document:
                has_attached_media = True
                media_type = "document"
                document = update.message.document
                document_file = await context.bot.get_file(document.file_id)
                document_bytes = await document_file.download_as_bytearray()
                
                # Debug logging for MIME type detection
                self.logger.info(f"Document filename: {document.file_name}")
                file_ext = os.path.splitext(document.file_name)[1].lower() if document.file_name else ""
                self.logger.info(f"Extracted file extension: '{file_ext}'")
                mime_type = MediaUtilities.get_mime_type(file_ext)
                self.logger.info(f"MIME type from extension: {mime_type}")
                
                # Fallback to content-based detection if extension detection fails
                if mime_type == "application/octet-stream" and document_bytes:
                    content_mime = MediaUtilities.detect_mime_from_content(document_bytes[:50])
                    if content_mime != "application/octet-stream":
                        mime_type = content_mime
                        self.logger.info(f"MIME type from content: {mime_type}")
                
                self.logger.info(f"Final MIME type: {mime_type}")
                
                # Special handling for image documents
                if MediaUtilities.is_image_file(file_ext):
                    self.logger.info("Document is an image, reclassifying as photo")
                    media_type = "photo"  # Reclassify image documents as photos
                
                media_files.append(
                    {
                        "type": media_type,
                        "data": io.BytesIO(document_bytes),
                        "mime": mime_type,
                        "filename": document.file_name or f"document_{document.file_id}",
                    }
                )
            elif update.message.media_group_id:
                has_attached_media = True
                media_type = "media_group"
                if "media_groups" not in context.bot_data:
                    context.bot_data["media_groups"] = {}
                media_group_id = update.message.media_group_id
                if media_group_id in context.bot_data["media_groups"]:
                    if update.message.photo:
                        photo = update.message.photo[-1]
                        photo_file = await context.bot.get_file(photo.file_id)
                        photo_bytes = await photo_file.download_as_bytearray()
                        context.bot_data["media_groups"][media_group_id].append(
                            {
                                "type": "photo",
                                "data": io.BytesIO(photo_bytes),
                                "mime": "image/jpeg",
                                "filename": f"photo_{photo.file_id}.jpg",
                            }
                        )
                    elif update.message.document:
                        document = update.message.document
                        document_file = await context.bot.get_file(document.file_id)
                        document_bytes = await document_file.download_as_bytearray()
                        file_ext = os.path.splitext(document.file_name)[1].lower() if document.file_name else ""
                        mime_type = MediaUtilities.get_mime_type(file_ext)
                        
                        # Special handling for image documents in media groups
                        doc_type = "photo" if MediaUtilities.is_image_file(file_ext) else "document"
                        
                        context.bot_data["media_groups"][media_group_id].append(
                            {
                                "type": doc_type,
                                "data": io.BytesIO(document_bytes),
                                "mime": mime_type,
                                "filename": document.file_name or f"document_{document.file_id}",
                            }
                        )
                    return False, [], None
                else:
                    context.bot_data["media_groups"][media_group_id] = []
                    if update.message.photo:
                        photo = update.message.photo[-1]
                        photo_file = await context.bot.get_file(photo.file_id)
                        photo_bytes = await photo_file.download_as_bytearray()
                        context.bot_data["media_groups"][media_group_id].append(
                            {
                                "type": "photo",
                                "data": io.BytesIO(photo_bytes),
                                "mime": "image/jpeg",
                                "filename": f"photo_{photo.file_id}.jpg",
                            }
                        )
                    elif update.message.document:
                        document = update.message.document
                        document_file = await context.bot.get_file(document.file_id)
                        document_bytes = await document_file.download_as_bytearray()
                        file_ext = os.path.splitext(document.file_name)[1].lower() if document.file_name else ""
                        mime_type = MediaUtilities.get_mime_type(file_ext)
                        
                        # Special handling for image documents in media groups (initialization)
                        doc_type = "photo" if MediaUtilities.is_image_file(file_ext) else "document"
                        
                        context.bot_data["media_groups"][media_group_id].append(
                            {
                                "type": doc_type,
                                "data": io.BytesIO(document_bytes),
                                "mime": mime_type,
                                "filename": document.file_name or f"document_{document.file_id}",
                            }
                        )
                    asyncio.create_task(
                        self._process_complete_media_group(
                            media_group_id,
                            update.effective_chat.id,
                            update.effective_user.id,
                            update.message.caption or "",
                            context,
                        )
                    )
                    return False, [], None
        return has_attached_media, media_files, media_type
    async def _process_complete_media_group(
        self, media_group_id, chat_id, user_id, caption, context
    ):
        """
        Process a complete media group after a delay to ensure all files are received
        """
        await asyncio.sleep(1.5)
        if (
            "media_groups" in context.bot_data
            and media_group_id in context.bot_data["media_groups"]
        ):
            media_files = context.bot_data["media_groups"][media_group_id]
            del context.bot_data["media_groups"][media_group_id]
            if media_files:
                thinking_message = await context.bot.send_message(
                    chat_id=chat_id, text="Processing multiple files... 🧠"
                )
                try:
                    from src.services.user_preferences_manager import UserPreferencesManager
                    preferences_manager = UserPreferencesManager(self.user_data_manager)
                    preferred_model = (
                        await preferences_manager.get_user_model_preference(user_id)
                    )
                    from src.services.media.multi_file_processor import MultiFileProcessor
                    multi_processor = MultiFileProcessor(self.gemini_api)
                    result = await multi_processor.process_multiple_files(
                        media_files, caption or "Analyze these files"
                    )
                    try:
                        await thinking_message.delete()
                    except Exception:
                        pass
                    model_indicator = (
                        "🧠 Gemini"
                        if preferred_model == "gemini"
                        else f"🤖 {preferred_model.capitalize()}"
                    )
                    if "intent" in result:
                        intent_message = f"I'm processing these files with intent: {result['intent']}\n\n"
                    else:
                        intent_message = ""
                    if "results" in result:
                        formatted_results = []
                        for filename, content in result["results"].items():
                            if isinstance(content, str):
                                header = f"📄 *{filename}*:"
                                formatted_results.append(f"{header}\n{content}")
                        if formatted_results:
                            response = (
                                f"{intent_message}{model_indicator}\n\n"
                                + "\n\n".join(formatted_results)
                            )
                            chunks = await self.response_formatter.split_long_message(
                                response
                            )
                            for chunk in chunks:
                                mock_message = self.MockMessage(context.bot, chat_id)
                                await self.response_formatter.safe_send_message(
                                    mock_message, chunk
                                )
                        else:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text="Sorry, I couldn't process these files properly. Please try again.",
                            )
                    else:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text="Sorry, I couldn't process these files. Please try again with a clearer prompt.",
                        )
                except Exception as e:
                    self.logger.error(f"Error processing media group: {e}")
                    try:
                        await thinking_message.delete()
                    except Exception:
                        pass
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="Sorry, there was an error processing your files. Please try again later.",
                    )
    async def _handle_media_analysis(
        self,
        update,
        context,
        thinking_message,
        media_files,
        media_type,
        message_text,
        user_id,
        preferred_model,
    ):
        """Handle analysis of media files"""
        if len(media_files) > 1:
            from src.services.media.multi_file_processor import MultiFileProcessor
            multi_processor = MultiFileProcessor(self.gemini_api)
            result = await multi_processor.process_multiple_files(
                media_files, message_text or "Analyze these files"
            )
            if thinking_message is not None:
                try:
                    await thinking_message.delete()
                    thinking_message = None
                except Exception:
                    pass
            model_indicator = (
                "🧠 Gemini"
                if preferred_model == "gemini"
                else f"🤖 {preferred_model.capitalize()}"
            )
            if "intent" in result:
                intent_message = (
                    f"I'm processing these files with intent: {result['intent']}\n\n"
                )
            else:
                intent_message = ""
            if "results" in result:
                formatted_results = []
                for filename, content in result["results"].items():
                    if isinstance(content, str):
                        header = f"📄 *{filename}*:"
                        formatted_results.append(f"{header}\n{content}")
                if formatted_results:
                    response = f"{intent_message}{model_indicator}\n\n" + "\n\n".join(
                        formatted_results
                    )
                    chunks = await self.response_formatter.split_long_message(response)
                    for chunk in chunks:
                        await self.response_formatter.safe_send_message(
                            update.message, chunk
                        )
                    if self.user_data_manager:
                        await self.user_data_manager.update_stats(
                            user_id, multi_file_analysis=True
                        )
                    media_description = "[Multiple files analysis request]"
                    await self.conversation_manager.save_media_interaction(
                        user_id,
                        "multi_files",
                        media_description,
                        response,
                        preferred_model,
                    )
                    
                    # Also save to main conversation history for follow-up questions  
                    await self.conversation_manager.save_message_pair(
                        user_id,
                        message_text or "[Multiple files uploaded]",
                        response,
                        preferred_model
                    )
                    return
            await update.message.reply_text(
                "Sorry, I couldn't analyze the content you provided. Please try again with a clearer prompt."
            )
            return
        result = await self.media_analyzer.analyze_media(
            media_files, message_text, preferred_model
        )
        if thinking_message is not None:
            try:
                await thinking_message.delete()
                thinking_message = None
            except Exception:
                pass
        if result:
            model_indicator = (
                "🧠 Gemini"
                if preferred_model == "gemini"
                else f"🤖 {preferred_model.capitalize()}"
            )
            text_to_send = self.response_formatter.format_with_model_indicator(
                result, model_indicator
            )
            await self.response_formatter.safe_send_message(
                update.message, text_to_send
            )
            media_description = f"[{media_type.capitalize()} analysis request]"
            await self.conversation_manager.save_media_interaction(
                user_id, media_type, media_description, result, preferred_model
            )
            
            # Also save to main conversation history for follow-up questions
            await self.conversation_manager.save_message_pair(
                user_id, 
                message_text or f"[{media_type.capitalize()} uploaded]", 
                result, 
                preferred_model
            )
            
            if self.user_data_manager:
                # Map media_type to the correct stat parameter
                stat_param = "image" if media_type == "photo" else media_type
                await self.user_data_manager.update_stats(user_id, **{stat_param: True})
        else:
            await self.response_formatter.safe_send_message(
                update.message,
                "Sorry, I couldn't analyze the content you provided. Please try again.",
            )
    async def _send_appropriate_chat_action(
        self, update, context, has_attached_media, media_type
    ):
        """Send appropriate chat action based on message type"""
        action = ChatAction.TYPING
        if has_attached_media:
            if media_type == "photo":
                action = ChatAction.UPLOAD_PHOTO
            elif media_type == "video":
                action = ChatAction.UPLOAD_VIDEO
            elif media_type == "audio":
                action = ChatAction.RECORD_VOICE
            elif media_type == "document":
                action = ChatAction.UPLOAD_DOCUMENT
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=action
        )
    async def _get_user_preferred_model(self, user_id):
        """Get user's preferred model"""
        from src.services.user_preferences_manager import UserPreferencesManager
        preferences_manager = UserPreferencesManager(self.user_data_manager)
        preferred_model = await preferences_manager.get_user_model_preference(user_id)
        self.logger.info(f"Preferred model for user {user_id}: {preferred_model}")
        return preferred_model
    async def _handle_text_conversation(
        self,
        update,
        context,
        thinking_message,
        message_text,
        quoted_text,
        quoted_message_id,
        history_context,
        user_id,
        preferred_model,
    ):
        """Handle regular text conversation"""
        message = update.message or update.edited_message
        enhanced_prompt = message_text
        if quoted_text:
            enhanced_prompt = self.prompt_formatter.add_context(
                message_text, "quote", quoted_text
            )
        
        # Get intelligent context including recent media interactions and conversation history
        intelligent_context = await self.conversation_manager.get_intelligent_context(user_id, message_text)
        
        # Format context if available
        if intelligent_context and intelligent_context.get("relevant_memory"):
            # Extract relevant text from context data
            context_texts = []
            for item in intelligent_context["relevant_memory"]:
                if isinstance(item, dict):
                    if 'content' in item:
                        context_texts.append(item['content'])
                    elif 'assistant_message' in item:
                        context_texts.append(item['assistant_message'])
                    elif 'user_message' in item:
                        context_texts.append(f"Previous: {item['user_message']}")
                elif isinstance(item, str):
                    context_texts.append(item)
            
            if context_texts:
                formatted_context = "\n".join(context_texts)
                self.logger.info(f"Adding intelligent context for user {user_id} (length: {len(formatted_context)})")
                enhanced_prompt = self.prompt_formatter.add_context(
                    enhanced_prompt, "context", formatted_context
                )
        enhanced_prompt_with_guidelines = (
            await self.prompt_formatter.apply_response_guidelines(
                enhanced_prompt,
                ModelHandlerFactory.get_model_handler(
                    preferred_model,
                    gemini_api=self.gemini_api,
                    openrouter_api=self.openrouter_api,
                    deepseek_api=self.deepseek_api,
                ),
                context,
            )
        )
        long_form_indicators = [
            "100",
            "list",
            "q&a",
            "qcm",
            "questions",
            "examples",
            "write me",
            "generate",
            "create",
            "explain in detail",
            "step by step",
            "tutorial",
            "guide",
            "comprehensive",
        ]
        is_long_form_request = any(
            indicator in message_text.lower() for indicator in long_form_indicators
        )
        from src.services.model_handlers.model_configs import ModelConfigurations, Provider
        model_config = ModelConfigurations.get_all_models().get(preferred_model)
        # Determine max_tokens based on model and request type
        # Different models have different context length limits
        if (
            model_config
            and hasattr(model_config, "max_tokens")
            and model_config.max_tokens
        ):
            base_max_tokens = model_config.max_tokens
        else:
            # Fallback to conservative default
            base_max_tokens = 6000
        
        # Adjust max_tokens based on request type and provider limits
        # DeepSeek has 8193 token limit (input + output combined)
        is_deepseek = model_config and model_config.provider == Provider.DEEPSEEK
        
        if is_long_form_request:
            # For DeepSeek, cap at 7500 to stay within 8193 total limit
            # This leaves ~693 tokens minimum for input context
            if is_deepseek:
                max_tokens = min(base_max_tokens, 7500)
            else:
                max_tokens = min(base_max_tokens, 12000)
        else:
            # For DeepSeek, cap at 7500 to stay within 8193 total limit
            # Error: "inputs tokens + max_new_tokens must be <= 8193"
            if is_deepseek:
                max_tokens = min(base_max_tokens, 7500)
            else:
                max_tokens = min(base_max_tokens, 8000)
        base_timeout = 60.0
        model_timeout = base_timeout
        complex_indicators = [
            "compare",
            "comparison",
            "vs",
            "versus",
            "difference",
            "differences",
            "analyze",
            "analysis",
            "explain",
            "detailed",
            "comprehensive",
            "performance",
            "benchmark",
            "pros and cons",
            "advantages",
            "disadvantages",
        ]
        is_complex_question = any(
            indicator in message_text.lower() for indicator in complex_indicators
        )
        if is_complex_question:
            if is_deepseek:
                model_timeout = 300.0
            else:
                model_timeout = 120.0
        elif is_long_form_request:
            if is_deepseek:
                model_timeout = 240.0
            else:
                model_timeout = 90.0
        if self.user_model_manager:
            model_config = self.user_model_manager.get_user_model_config(user_id)
            model_timeout = (
                model_config.timeout_seconds if model_config else model_timeout
            )
        self.logger.info(
            f"Using timeout {model_timeout}s for user {user_id} with model {preferred_model} "
            f"(complex: {is_complex_question}, long_form: {is_long_form_request})"
        )
        try:
            mcp_compatible = (
                is_model_mcp_compatible(preferred_model) if preferred_model else False
            )
            if mcp_compatible:
                mcp_response = await generate_mcp_response(
                    prompt=enhanced_prompt_with_guidelines,
                    user_id=user_id,
                    model=preferred_model,
                    temperature=0.7,
                    max_tokens=max_tokens,
                    context=history_context,
                )
                if mcp_response:
                    self.logger.info(f"Using MCP-enhanced response for user {user_id}")
                    response = mcp_response
                    actual_model_used = preferred_model
    
                    await self._handle_mcp_document_output(
                        context=context,
                        user_id=user_id,
                        message=message,
                        mcp_response=mcp_response
                    )
                else:
                    self.logger.debug(
                        f"MCP response failed for user {user_id}, using regular processing"
                    )
                    (
                        response,
                        actual_model_used,
                    ) = await self.model_fallback_handler.attempt_with_fallback(
                        primary_model=preferred_model,
                        model_handler_factory=ModelHandlerFactory,
                        enhanced_prompt=enhanced_prompt_with_guidelines,
                        history_context=history_context,
                        max_tokens=max_tokens,
                        model_timeout=model_timeout,
                        message=message,
                        is_complex_question=is_complex_question,
                        quoted_text=quoted_text,
                        gemini_api=self.gemini_api,
                        openrouter_api=self.openrouter_api,
                        deepseek_api=self.deepseek_api,
                    )
            else:
                self.logger.debug(
                    f"Model {preferred_model} not MCP compatible, using regular processing"
                )
                (
                    response,
                    actual_model_used,
                ) = await self.model_fallback_handler.attempt_with_fallback(
                    primary_model=preferred_model,
                    model_handler_factory=ModelHandlerFactory,
                    enhanced_prompt=enhanced_prompt_with_guidelines,
                    history_context=history_context,
                    max_tokens=max_tokens,
                    model_timeout=model_timeout,
                    message=message,
                    is_complex_question=is_complex_question,
                    quoted_text=quoted_text,
                    gemini_api=self.gemini_api,
                    openrouter_api=self.openrouter_api,
                    deepseek_api=self.deepseek_api,
                )
            if response:
                response = self._clean_response_content(response)
            if response:
                response_length = len(response)
                response_preview = (
                    response[:200] + "..." if len(response) > 200 else response
                )
                self.logger.info(
                    f"Generated response length: {response_length} characters using model: {actual_model_used}"
                )
                self.logger.debug(f"Response preview: {response_preview}")
            else:
                self.logger.warning("No response generated from fallback system")
            if thinking_message is not None:
                try:
                    await thinking_message.delete()
                    thinking_message = None
                except Exception:
                    pass
            if response is None:
                await self.response_formatter.safe_send_message(
                    message,
                    "Sorry, I couldn't generate a response. Please try rephrasing your message.",
                )
                return
            actual_model_handler = ModelHandlerFactory.get_model_handler(
                actual_model_used,
                gemini_api=self.gemini_api,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )
            await self._send_formatted_response(
                update,
                context,
                message,
                response,
                actual_model_handler.get_model_indicator(actual_model_used),
                quoted_text,
                quoted_message_id,
            )
            if response:
                await self.memory_manager.extract_and_save_user_info(
                    user_id, message_text
                )
                if quoted_text:
                    await self.conversation_manager.add_quoted_message_context(
                        user_id, quoted_text, message_text, response, actual_model_used
                    )
                else:
                    await self.conversation_manager.save_message_pair(
                        user_id, message_text, response, actual_model_used
                    )
            telegram_logger.log_message("Text response sent successfully", user_id)
        except asyncio.TimeoutError:
            if thinking_message is not None:
                await thinking_message.delete()
            if is_deepseek:
                if is_complex_question:
                    timeout_message = "⏱️ DeepSeek R1 is thoroughly analyzing your complex question but needs more time. This usually means a very detailed response is being prepared. Please try again or break the question into smaller parts."
                else:
                    timeout_message = "⏱️ DeepSeek R1 timed out. The model may be experiencing high load. Please try again in a moment."
            else:
                if is_complex_question:
                    timeout_message = "⏱️ Your complex question required more processing time than available. For detailed comparisons and analyses, try breaking it into smaller parts or asking again."
                elif is_long_form_request:
                    timeout_message = "⏱️ Your long-form request timed out. Try asking for a shorter response or break it into multiple questions."
                else:
                    timeout_message = "⏱️ Sorry, the request took too long to process. Please try again or rephrase your question."
            await self.response_formatter.safe_send_message(message, timeout_message)
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            if thinking_message is not None:
                await thinking_message.delete()
            if "timeout" in str(e).lower() or isinstance(e, asyncio.TimeoutError):
                if is_complex_question:
                    error_message = "⏱️ Your detailed question needed more time than available. Try:\n• Breaking it into simpler parts\n• Asking for a shorter comparison\n• Focusing on specific aspects"
                else:
                    error_message = "⏱️ Processing took too long. Please try rephrasing your question or try again in a moment."
            else:
                error_message = "❌ Sorry, there was an error processing your request. Please try again or rephrase your question."
            await self.response_formatter.safe_send_message(message, error_message)
    async def _send_formatted_response(
        self,
        update,
        context,
        message,
        response,
        model_indicator,
        quoted_text,
        quoted_message_id,
    ):
        """Format and send the AI response"""
        message_chunks = await self.response_formatter.split_long_message(response)
        sent_messages = []
        context.user_data["last_message_indicator"] = model_indicator
        is_reply = self.context_handler.should_use_reply_format(
            quoted_text, quoted_message_id
        )
        for i, chunk in enumerate(message_chunks):
            try:
                if i == 0:
                    text_to_send = self.response_formatter.format_with_model_indicator(
                        chunk, model_indicator, is_reply
                    )
                else:
                    text_to_send = chunk
                if i == 0 and is_reply:
                    last_message = await self.response_formatter.safe_send_message(
                        message, text_to_send, reply_to_message_id=quoted_message_id
                    )
                elif i == 0:
                    last_message = await self.response_formatter.safe_send_message(
                        message, text_to_send
                    )
                else:
                    mock_message = self.MockMessage(
                        context.bot, update.effective_chat.id
                    )
                    last_message = await self.response_formatter.safe_send_message(
                        mock_message, text_to_send
                    )
                if last_message:
                    sent_messages.append(last_message)
            except Exception as final_error:
                self.logger.error(
                    f"Failed to send message chunk {i}: {str(final_error)}"
                )
                continue
        if sent_messages:
            if "bot_messages" not in context.user_data:
                context.user_data["bot_messages"] = {}
            context.user_data["bot_messages"][message.message_id] = [
                msg.message_id for msg in sent_messages
            ]
    async def _load_user_context(self, user_id: int, update: Update) -> str:
        """Load user context including name and profile information from MongoDB."""
        try:
            user_context_parts = []
            user = update.effective_user
            if user:
                if user.first_name:
                    user_context_parts.append(f"Name: {user.first_name}")
                if user.last_name:
                    user_context_parts.append(f"Last name: {user.last_name}")
                if user.username:
                    user_context_parts.append(f"Username: @{user.username}")
            try:
                user_profile = await self.memory_manager.get_user_profile(user_id)
                if user_profile:
                    if user_profile.get("name"):
                        user_context_parts.append(
                            f"Preferred name: {user_profile['name']}"
                        )
                    if user_profile.get("conversation_count"):
                        count = user_profile["conversation_count"]
                        user_context_parts.append(f"Previous conversations: {count}")
                    for key, value in user_profile.items():
                        if (
                            key
                            not in [
                                "name",
                                "conversation_count",
                                "created_at",
                                "last_updated",
                            ]
                            and value
                        ):
                            user_context_parts.append(f"{key.capitalize()}: {value}")
                user_data = await self.user_data_manager.get_user_data(user_id)
                if user_data:
                    user_prefs = user_data.get("preferences", {})
                    if (
                        user_prefs.get("name")
                        and f"Preferred name: {user_prefs['name']}"
                        not in user_context_parts
                    ):
                        user_context_parts.append(
                            f"Preferred name: {user_prefs['name']}"
                        )
            except Exception as e:
                self.logger.debug(f"Could not load user profile from MongoDB: {e}")
            return "; ".join(user_context_parts) if user_context_parts else ""
        except Exception as e:
            self.logger.error(f"Error loading user context: {e}")
            return ""
    def _clean_response_content(self, content: str) -> str:
        """
        Clean response content by removing thinking tags and tool calls.
        Args:
            content: Raw response content from the model
        Returns:
            Cleaned content suitable for user display
        """
        if not content:
            return content
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
        content = re.sub(r"<[^>]+>.*?</[^>]+>", "", content, flags=re.DOTALL)
        content = content.strip()
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
        return content

    def _is_previous_question_intent(self, message_text: str) -> bool:
        """
        Detects if the user is asking about their previous questions.
        """
        if not message_text:
            return False
        lowered = message_text.lower()
        # Simple keyword-based check, can be improved with NLP
        return (
            "previous question" in lowered
            or "last question" in lowered
            or "which question did i ask" in lowered
            or "what did i ask" in lowered
            or "history of my questions" in lowered
        )

    async def _handle_mcp_document_output(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        user_id: int,
        message,
        mcp_response: str
    ) -> None:
        """
        Handle document outputs from MCP tools (e.g., office-word-mcp-server).
        Detects if documents were created and sends them to the user.
        
        Args:
            context: Telegram context
            user_id: User ID
            message: Original message object
            mcp_response: Response from MCP tool execution
        """
        try:
            # Check if the response indicates document creation
            document_indicators = [
                'document created',
                'saved to',
                'file created',
                '.docx',
                '.pdf',
                '.xlsx',
                '.pptx'
            ]
            
            if not any(indicator in mcp_response.lower() for indicator in document_indicators):
                self.logger.debug("No document creation indicators found in MCP response")
                return
            
            # Search for recently created documents (within last 5 minutes)
            recent_documents = self.document_sender.find_recent_documents(
                directory=".",
                max_age_seconds=300
            )
            
            if not recent_documents:
                self.logger.info("No recent documents found despite document creation indicators")
                return
            
            # Send the most recent document
            document_path = recent_documents[0]
            self.logger.info(f"Found recent document for user {user_id}: {document_path}")
            
            # Prepare caption from document name
            import os
            file_name = os.path.basename(document_path)
            caption = f"📄 {file_name}"
            
            # Send document to user
            success = await self.document_sender.send_document(
                bot=context.bot,
                chat_id=message.chat_id,
                file_path=document_path,
                caption=caption,
                reply_to_message_id=message.message_id
            )
            
            if success:
                self.logger.info(f"Successfully sent document {document_path} to user {user_id}")
                telegram_logger.log_message(
                    f"Document sent: {file_name}",
                    user_id
                )
            else:
                self.logger.warning(f"Failed to send document {document_path} to user {user_id}")
                
        except Exception as e:
            self.logger.error(f"Error handling MCP document output: {str(e)}")

