import io
import os
import logging
import re
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from src.utils.log.telegramlog import telegram_logger
from src.services.gemini_api import GeminiAPI
from src.services.user_data_manager import UserDataManager
import asyncio
from .message_context_handler import MessageContextHandler
from .response_formatter import ResponseFormatter
from .media_context_extractor import MediaContextExtractor
from src.services.memory_context.memory_manager import MemoryManager
from src.services.memory_context.model_history_manager import ModelHistoryManager
from src.services.model_handlers.factory import ModelHandlerFactory
from src.services.model_handlers.prompt_formatter import PromptFormatter
from src.services.memory_context.conversation_manager import ConversationManager
from .text_processing.media_analyzer import MediaAnalyzer
from .text_processing.utilities import MediaUtilities
from .model_fallback_handler import ModelFallbackHandler
from src.services.ai_command_router import EnhancedIntentDetector

# Import the bot username helper
from src.utils.bot_username_helper import BotUsernameHelper

# Import MCP integration for /context functionality
from src.services.mcp import get_mcp_registry


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

        # Initialize memory and history management
        self.memory_manager = MemoryManager(
            db=user_data_manager.db if hasattr(user_data_manager, "db") else None,
        )
        self.memory_manager.short_term_limit = 15
        self.memory_manager.token_limit = 8192
        self.model_history_manager = ModelHistoryManager(self.memory_manager)

        # Initialize utility classes (removed unused ones)
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()
        self.prompt_formatter = PromptFormatter()
        self.media_context_extractor = MediaContextExtractor()

        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            self.memory_manager, self.model_history_manager
        )

        # Initialize core components only
        self.media_analyzer = MediaAnalyzer(gemini_api)
        # Initialize model fallback handler
        self.model_fallback_handler = ModelFallbackHandler(self.response_formatter)
        # Initialize enhanced intent detector for user intent detection
        self.intent_detector = EnhancedIntentDetector()

        # Optional components (will be set externally if needed)
        self.user_model_manager = None
        
        # MCP integration cache for discovered tools
        self.discovered_mcp_tools = {}

    def _detect_context_command(self, text: str) -> tuple[bool, str]:
        """
        Detect context commands and MCP tool patterns.
        
        Args:
            text: Message text to check
            
        Returns:
            tuple: (is_context_command, extracted_query)
        """
        # Match /context command pattern
        context_pattern = r'^/context\s+(.+)$'
        context_match = re.match(context_pattern, text.strip(), re.IGNORECASE)
        
        if context_match:
            return True, context_match.group(1).strip()
        
        # Check for MCP tool commands that might not be handled by command handlers
        mcp_patterns = [
            r'^/sequentialthinking\s+(.+)$',
            r'^/search\s+(.+)$', 
            r'^/company\s+(.+)$',
            r'^/context7\s+(.+)$',
            r'^/docfork\s+(.+)$'
        ]
        
        for pattern in mcp_patterns:
            match = re.match(pattern, text.strip(), re.IGNORECASE)
            if match:
                return True, match.group(1).strip()
        
        return False, ""
        
    async def _discover_mcp_tools(self) -> dict:
        """Discover available MCP tools, with caching."""
        if self.discovered_mcp_tools:
            return self.discovered_mcp_tools
            
        try:
            registry = await get_mcp_registry()
            self.discovered_mcp_tools = await registry.discover_available_tools()
            return self.discovered_mcp_tools
        except Exception as e:
            self.logger.error(f"Failed to discover MCP tools: {e}")
            return {}
            
    async def _execute_mcp_context(self, query: str, user_id: int, preferred_model: str = None) -> tuple[bool, str]:
        """
        Execute MCP tools for context query.
        
        Args:
            query: User query for MCP tools
            user_id: User ID for logging
            preferred_model: User's preferred model for MCP execution
            
        Returns:
            tuple: (success, result_text)
        """
        try:
            # Discover available tools
            discovered_tools = await self._discover_mcp_tools()
            
            if not discovered_tools:
                return False, "❌ No MCP tools available for context queries."
            
            # Use user's preferred model or fallback to system default
            if preferred_model:
                model_to_use = preferred_model
            else:
                # Import and use the proper default model from model configs
                from src.services.model_handlers.model_configs import get_default_agent_model
                model_to_use = get_default_agent_model()
            
            # Use OpenRouter API with MCP integration if available
            if hasattr(self, 'openrouter_api') and self.openrouter_api:
                # Create enhanced prompt that guides the AI to use appropriate MCP tools
                enhanced_prompt = f"""I need help with: {query}

Please use the most appropriate available MCP tool to help answer this query. You have access to web search, company research, documentation search, and other specialized tools through the Model Context Protocol.

Provide a comprehensive and helpful response using the available tools."""

                # Execute using OpenRouter API with MCP tools
                response, tool_logger = await self.openrouter_api.generate_response_with_tool_logging(
                    prompt=enhanced_prompt,
                    model=model_to_use,
                    use_mcp=True,
                    timeout=120.0
                )
                
                if response:
                    # Return the response directly for unified formatting
                    return True, response
                else:
                    return False, "❌ No response generated from MCP tools."
            else:
                return False, "❌ OpenRouter API not available for MCP integration."
                
        except Exception as e:
            self.logger.error(f"Error executing MCP context: {e}")
            return False, f"❌ Error executing context query: {str(e)[:200]}..."

    # Removed delegation methods - use response_formatter directly

    async def handle_text_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Main handler for text messages.
        Processes user messages, detects intent, and generates appropriate responses.
        Includes integrated MCP command handling through the unified workflow.
        """
        if not update.message and not update.edited_message:
            return

        user_id = update.effective_user.id
        message = update.message or update.edited_message
        message_text = message.text

        # Check if this is a group chat and handle accordingly
        chat = update.effective_chat
        is_group = chat and chat.type in ["group", "supergroup"]

        if (
            is_group
            and hasattr(self, "_group_chat_integration")
            and self._group_chat_integration
        ):
            # Process message through group chat integration first
            enhanced_message = await self._group_chat_integration.process_message(
                update, context
            )
            if enhanced_message:
                # Update message text with enhanced context if available
                if enhanced_message.get("enhanced_text"):
                    message_text = enhanced_message["enhanced_text"]

        # Check for enhanced group message
        if "enhanced_message" in context.user_data:
            enhanced_message_text = context.user_data["enhanced_message"]
            group_metadata = context.user_data.get("group_context", {})

            # Use enhanced message for group processing
            if update.effective_chat.type in ["group", "supergroup"]:
                message_text = enhanced_message_text
                self.logger.info("Using enhanced group message for processing")

        # Extract quoted message context if this is a reply
        quoted_text, quoted_message_id = self.context_handler.extract_reply_context(
            message
        )

        # Define conversation ID for this user
        conversation_id = f"user_{user_id}"

        try:
            # Handle edited messages
            if update.edited_message and "bot_messages" in context.user_data:
                await self._handle_edited_message(update, context)

            # In group chats, only process messages that mention the bot
            if update.effective_chat.type in ["group", "supergroup"]:
                # Extract entities for accurate mention detection
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
                    # Bot not mentioned, ignore message
                    self.logger.debug(
                        f"Bot not mentioned in group chat message: '{message_text}'"
                    )
                    return
                else:
                    # Remove bot username from message text
                    self.logger.debug("Bot mentioned in group chat, processing message")
                    message_text = BotUsernameHelper.remove_bot_mention(
                        message_text, context
                    )

            # Extract any attached media files
            (
                has_attached_media,
                media_files,
                media_type,
            ) = await self._extract_media_files(update, context)

            # Send initial "thinking" message and appropriate chat action
            thinking_message = await message.reply_text("Processing your request...🧠")
            await self._send_appropriate_chat_action(
                update, context, has_attached_media, media_type
            )  # Detect user intent (analyze, generate image, or chat)
            intent_result = await self.intent_detector.detect_intent(message_text)
            user_intent = (intent_result.intent, intent_result.confidence)
            
            # Get user's preferred model early for consistent usage
            preferred_model = await self._get_user_preferred_model(user_id)
            
            # Check for /context command - integrate MCP tools in conversation flow
            is_context_command, context_query = self._detect_context_command(message_text)
            
            if is_context_command:
                # Execute MCP context query within the conversation workflow
                telegram_logger.log_message(f"Context command: {context_query}", user_id)
                
                # Execute MCP tools with user's preferred model
                success, mcp_result = await self._execute_mcp_context(context_query, user_id, preferred_model)
                
                if success:
                    # Instead of replacing message_text, use MCP result as the AI response
                    # This maintains the original query in memory but provides MCP-enhanced response
                    
                    # Update thinking message to show MCP processing
                    await thinking_message.edit_text("🔧 Processing with MCP tools...")
                    
                    # Use the MCP result directly as the response, but process it through normal formatting
                    response = mcp_result
                    
                    # Get model indicator for proper formatting
                    if 'qwen3-235b' in preferred_model.lower():
                        model_indicator = "🌟 Qwen3 235B A22B"
                    else:
                        model_indicator = f"🌟 {preferred_model.split('/')[-1].replace('-', ' ').title()}"
                    
                    # Send the MCP response with proper formatting and memory persistence
                    await self._send_formatted_response(
                        update, context, message, response, model_indicator,
                        quoted_text, quoted_message_id
                    )
                    
                    # Store in conversation memory
                    await self.conversation_manager.save_message_pair(
                        user_id=user_id,
                        user_message=message_text,
                        assistant_message=response,
                        model_id=preferred_model
                    )
                    
                    # Update user stats
                    self.user_data_manager.update_user_stats(user_id, {
                        'mcp_queries': 1,
                        'last_mcp_query': context_query,
                        'last_activity': datetime.now().isoformat()
                    })
                    telegram_logger.log_message("MCP context response sent successfully", user_id)
                    return
                else:
                    # Handle MCP failure gracefully with proper formatting
                    await thinking_message.edit_text(mcp_result, parse_mode='Markdown')
                    return

            # Extract user information using memory manager directly
            await self.memory_manager.extract_and_save_user_info(user_id, message_text)

            # Get conversation history for context
            history_context = await self.conversation_manager.get_conversation_history(
                user_id, max_messages=self.max_context_length, model=preferred_model
            )

            # Load user context and personal information to enhance conversation
            user_context = await self._load_user_context(
                user_id, update
            )  # Enhance conversation history with user context if available
            if user_context and history_context:
                # Add user context to the beginning of history to maintain continuity
                user_context_message = {
                    "role": "system",
                    "content": f"User information: {user_context}",
                }
                history_context.insert(0, user_context_message)

            # Handle media analysis if media is attached and intent is analyze
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

            # If we're here, this is a regular text conversation
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

            # Use safe message sending with automatic fallback
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
                file_ext = os.path.splitext(document.file_name)[1].lower()
                mime_type = MediaUtilities.get_mime_type(file_ext)
                media_files.append(
                    {
                        "type": "document",
                        "data": io.BytesIO(document_bytes),
                        "mime": mime_type,
                        "filename": document.file_name,
                    }
                )

            # Check for media group (multiple files sent together)
            elif update.message.media_group_id:
                has_attached_media = True
                media_type = "media_group"

                # Store the media group ID in context for tracking
                if "media_groups" not in context.bot_data:
                    context.bot_data["media_groups"] = {}

                media_group_id = update.message.media_group_id

                # Check if we've already processed this media group partially
                if media_group_id in context.bot_data["media_groups"]:
                    # Add this media item to the existing group
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
                        file_ext = os.path.splitext(document.file_name)[1].lower()
                        mime_type = MediaUtilities.get_mime_type(file_ext)
                        context.bot_data["media_groups"][media_group_id].append(
                            {
                                "type": "document",
                                "data": io.BytesIO(document_bytes),
                                "mime": mime_type,
                                "filename": document.file_name,
                            }
                        )

                    # Do not process each media file individually - return empty for now
                    # The complete group will be processed once the caption message is received
                    # or after a short delay
                    return False, [], None
                else:
                    # Start tracking this new media group
                    context.bot_data["media_groups"][media_group_id] = []

                    # Check the current message's type
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
                        file_ext = os.path.splitext(document.file_name)[1].lower()
                        mime_type = MediaUtilities.get_mime_type(file_ext)
                        context.bot_data["media_groups"][media_group_id].append(
                            {
                                "type": "document",
                                "data": io.BytesIO(document_bytes),
                                "mime": mime_type,
                                "filename": document.file_name,
                            }
                        )

                    # Schedule a task to process the complete media group after a delay
                    # This gives time for all media files to be received
                    asyncio.create_task(
                        self._process_complete_media_group(
                            media_group_id,
                            update.effective_chat.id,
                            update.effective_user.id,
                            update.message.caption or "",
                            context,
                        )
                    )

                    # Return empty for now - processing will happen in the scheduled task
                    return False, [], None

        return has_attached_media, media_files, media_type

    async def _process_complete_media_group(
        self, media_group_id, chat_id, user_id, caption, context
    ):
        """
        Process a complete media group after a delay to ensure all files are received
        """
        # Wait a moment for all media files to be received (Telegram typically sends them in quick succession)
        await asyncio.sleep(1.5)

        if (
            "media_groups" in context.bot_data
            and media_group_id in context.bot_data["media_groups"]
        ):
            media_files = context.bot_data["media_groups"][media_group_id]

            # Clean up the stored media group to avoid memory leaks
            del context.bot_data["media_groups"][media_group_id]

            # Only process if we have files
            if media_files:
                # Send a thinking message
                thinking_message = await context.bot.send_message(
                    chat_id=chat_id, text="Processing multiple files... 🧠"
                )

                try:
                    # Get user's preferred model
                    from services.user_preferences_manager import UserPreferencesManager

                    preferences_manager = UserPreferencesManager(self.user_data_manager)
                    preferred_model = (
                        await preferences_manager.get_user_model_preference(user_id)
                    )

                    # Process multiple files using our new MultiFileProcessor
                    from services.media.multi_file_processor import MultiFileProcessor

                    multi_processor = MultiFileProcessor(self.gemini_api)

                    result = await multi_processor.process_multiple_files(
                        media_files, caption or "Analyze these files"
                    )

                    # Delete thinking message
                    try:
                        await thinking_message.delete()
                    except Exception:
                        pass

                    # Format and send the response
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
                        # Format each file result
                        formatted_results = []
                        for filename, content in result["results"].items():
                            if isinstance(content, str):
                                header = f"📄 *{filename}*:"
                                formatted_results.append(f"{header}\n{content}")

                        # Combine results
                        if formatted_results:
                            response = (
                                f"{intent_message}{model_indicator}\n\n"
                                + "\n\n".join(formatted_results)
                            )

                            # Send each chunk using safe method
                            chunks = await self.response_formatter.split_long_message(
                                response
                            )
                            for chunk in chunks:
                                # Create a mock message object to use with safe_send_message
                                class MockMessage:
                                    def __init__(self, bot, chat_id):
                                        self.bot = bot
                                        self.chat_id = chat_id

                                    async def reply_text(self, text, **kwargs):
                                        return await self.bot.send_message(
                                            chat_id=self.chat_id, text=text, **kwargs
                                        )

                                mock_message = MockMessage(context.bot, chat_id)
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
        # Check if we have multiple files to process
        if len(media_files) > 1:
            # Use MultiFileProcessor for multiple files
            from services.media.multi_file_processor import MultiFileProcessor

            multi_processor = MultiFileProcessor(self.gemini_api)
            result = await multi_processor.process_multiple_files(
                media_files, message_text or "Analyze these files"
            )

            # Delete thinking message
            if thinking_message is not None:
                try:
                    await thinking_message.delete()
                    thinking_message = None
                except Exception:
                    pass

            # Format and send the response
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
                # Format each file result
                formatted_results = []
                for filename, content in result["results"].items():
                    if isinstance(content, str):
                        header = f"📄 *{filename}*:"
                        formatted_results.append(f"{header}\n{content}")

                # Combine results
                if formatted_results:
                    response = f"{intent_message}{model_indicator}\n\n" + "\n\n".join(
                        formatted_results
                    )

                    # Split and send using safe_send_message
                    chunks = await self.response_formatter.split_long_message(response)
                    for chunk in chunks:
                        await self.response_formatter.safe_send_message(
                            update.message, chunk
                        )

                    # Update user stats
                    if self.user_data_manager:
                        await self.user_data_manager.update_stats(
                            user_id, multi_file_analysis=True
                        )

                    # Save to conversation history
                    media_description = "[Multiple files analysis request]"
                    await self.conversation_manager.save_media_interaction(
                        user_id,
                        "multi_files",
                        media_description,
                        response,
                        preferred_model,
                    )

                    return

            # If we get here, something went wrong with the processing
            await update.message.reply_text(
                "Sorry, I couldn't analyze the content you provided. Please try again with a clearer prompt."
            )
            return

        # Single file handling (existing code)
        result = await self.media_analyzer.analyze_media(
            media_files, message_text, preferred_model
        )

        # Delete thinking message
        if thinking_message is not None:
            try:
                await thinking_message.delete()
                thinking_message = None
            except Exception:
                pass

        # Format and send the response
        if result:
            # Add model indicator
            model_indicator = (
                "🧠 Gemini"
                if preferred_model == "gemini"
                else f"🤖 {preferred_model.capitalize()}"
            )
            text_to_send = self.response_formatter.format_with_model_indicator(
                result, model_indicator
            )

            # Format for Telegram and send the response using safe_send_message
            await self.response_formatter.safe_send_message(
                update.message, text_to_send
            )

            # Save to conversation history
            media_description = f"[{media_type.capitalize()} analysis request]"
            await self.conversation_manager.save_media_interaction(
                user_id, media_type, media_description, result, preferred_model
            )

            # Update user stats
            if self.user_data_manager:
                await self.user_data_manager.update_stats(user_id, **{media_type: True})
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
        from services.user_preferences_manager import UserPreferencesManager

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

        # Build enhanced prompt with context
        enhanced_prompt = message_text

        # Add quoted text context if this is a reply
        if quoted_text:
            enhanced_prompt = self.prompt_formatter.add_context(
                message_text, "quote", quoted_text
            )

        # Add any reference to previously shared media
        # Check both our detection method and the media context extractor's method
        is_referring_to_image = self.context_handler.detect_reference_to_image(
            message_text
        ) or self.media_context_extractor.is_referring_to_image(message_text)

        if is_referring_to_image and "image_history" in context.user_data:
            # Generate context from previous images
            image_context = await self.media_context_extractor.get_image_context(
                context.user_data
            )

            # Add an explicit instruction for the model about handling images
            instruction = (
                "The user is referring to an image they previously shared. "
                "Use the following image information to answer their question. "
                "DO NOT say you don't have access to images - you've previously analyzed "
                "these images and should use that analysis to answer."
            )

            # Add both the context and instruction
            enhanced_prompt = self.prompt_formatter.add_context(
                enhanced_prompt, "image", f"{instruction}\n\n{image_context}"
            )

        if (
            self.context_handler.detect_reference_to_document(message_text)
            and "document_history" in context.user_data
        ):
            document_context = await self.media_context_extractor.get_document_context(
                context.user_data
            )
            enhanced_prompt = self.prompt_formatter.add_context(
                enhanced_prompt, "document", document_context
            )

        # Apply response style guidelines
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

        # Detect if this is a long-form request and adjust max_tokens accordingly
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
        max_tokens = 32000 if is_long_form_request else 16000

        # Get model timeout - increase for complex questions
        base_timeout = 60.0
        model_timeout = base_timeout

        # Detect if this is a complex question that needs more time
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

        # Set timeout based on question complexity and model type
        if is_complex_question:
            # DeepSeek R1 needs more time for reasoning
            if "deepseek" in preferred_model.lower():
                model_timeout = 300.0
            else:
                model_timeout = 120.0
        elif is_long_form_request:
            if "deepseek" in preferred_model.lower():
                model_timeout = 240.0
            else:
                model_timeout = 90.0

        if self.user_model_manager:
            model_config = self.user_model_manager.get_user_model_config(user_id)
            model_timeout = (
                model_config.timeout_seconds if model_config else model_timeout
            )

        # Log timeout configuration for debugging
        self.logger.info(
            f"Using timeout {model_timeout}s for user {user_id} with model {preferred_model} "
            f"(complex: {is_complex_question}, long_form: {is_long_form_request})"
        )

        try:
            # Use automatic fallback system to generate response
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

            # Log response length and first part for debugging
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

            # Delete thinking message
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

            # Get model handler for the actual model used (for model indicator)
            actual_model_handler = ModelHandlerFactory.get_model_handler(
                actual_model_used,
                gemini_api=self.gemini_api,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )

            # Send the response
            await self._send_formatted_response(
                update,
                context,
                message,
                response,
                actual_model_handler.get_model_indicator(),
                quoted_text,
                quoted_message_id,
            )

            # Save to conversation history
            if response:
                # Extract and save user information from the message (like name)
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

            # Provide more specific timeout messages based on model and question type
            if "deepseek" in preferred_model.lower():
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

            # Provide context-aware error messages
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
        # Split long messages
        message_chunks = await self.response_formatter.split_long_message(response)
        sent_messages = []

        # Store indicator for editing functionality
        context.user_data["last_message_indicator"] = model_indicator

        # Determine if this is a reply
        is_reply = self.context_handler.should_use_reply_format(
            quoted_text, quoted_message_id
        )

        # Send each chunk using safe_send_message for better error handling
        for i, chunk in enumerate(message_chunks):
            try:
                # Format first chunk with model indicator
                if i == 0:
                    text_to_send = self.response_formatter.format_with_model_indicator(
                        chunk, model_indicator, is_reply
                    )
                else:
                    text_to_send = chunk

                # Use safe_send_message for automatic fallback handling
                if i == 0 and is_reply:
                    last_message = await self.response_formatter.safe_send_message(
                        message, text_to_send, reply_to_message_id=quoted_message_id
                    )
                elif i == 0:
                    last_message = await self.response_formatter.safe_send_message(
                        message, text_to_send
                    )
                else:
                    # For subsequent messages, create a mock message object
                    class MockMessage:
                        def __init__(self, bot, chat_id):
                            self.bot = bot
                            self.chat_id = chat_id

                        async def reply_text(self, text, **kwargs):
                            return await self.bot.send_message(
                                chat_id=self.chat_id, text=text, **kwargs
                            )

                    mock_message = MockMessage(context.bot, update.effective_chat.id)
                    last_message = await self.response_formatter.safe_send_message(
                        mock_message, text_to_send
                    )

                if last_message:
                    sent_messages.append(last_message)

            except Exception as final_error:
                self.logger.error(
                    f"Failed to send message chunk {i}: {str(final_error)}"
                )
                # Continue with next chunk instead of failing completely
                continue

        # Store message IDs for future editing
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

            # Get user's Telegram profile information
            user = update.effective_user
            if user:
                if user.first_name:
                    user_context_parts.append(f"Name: {user.first_name}")
                if user.last_name:
                    user_context_parts.append(f"Last name: {user.last_name}")
                if user.username:
                    user_context_parts.append(f"Username: @{user.username}")

            # Load stored user profile from MongoDB via memory manager
            try:
                user_profile = await self.memory_manager.get_user_profile(user_id)
                if user_profile:
                    # Extract relevant user information
                    if user_profile.get("name"):
                        user_context_parts.append(
                            f"Preferred name: {user_profile['name']}"
                        )

                    if user_profile.get("conversation_count"):
                        count = user_profile["conversation_count"]
                        user_context_parts.append(f"Previous conversations: {count}")

                    # Add any other stored personal information
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

                # Also load any stored user preferences from user_data_manager
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
