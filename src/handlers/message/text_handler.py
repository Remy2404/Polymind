"""
Text message handling.
"""
import logging
from telegram import Update
from telegram.ext import ContextTypes
from src.handlers.text_handlers import TextHandler
from src.services.group_chat.integration import GroupChatIntegration
from src.utils.bot_username_helper import BotUsernameHelper


class TextMessageHandlerMixin:
    """Mixin for text message handling."""

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

            # Check for awaiting document text
            if update.message and await self.handle_awaiting_doc_text(update, context):
                return

            # Check for awaiting AI doc topic
            if update.message and context.user_data.get("awaiting_aidoc_topic"):
                context.user_data["awaiting_aidoc_topic"] = False
                context.user_data["aidoc_prompt"] = update.message.text
                if hasattr(self, "command_handlers") and self.command_handlers:
                    await self.command_handlers.document_commands._show_ai_document_format_selection(
                        update, context
                    )
                else:
                    from src.handlers.commands.document_commands import DocumentCommands
                    doc_commands = DocumentCommands(
                        self.gemini_api, self.user_data_manager, self.telegram_logger
                    )
                    await doc_commands._show_ai_document_format_selection(update, context)
                return

            await self.user_data_manager.initialize_user(user_id)

            # Initialize group chat integration if needed
            if self._group_chat_integration is None and self._conversation_manager:
                self._group_chat_integration = GroupChatIntegration(
                    self.user_data_manager, self._conversation_manager
                )

            enhanced_message_text = message_text
            group_metadata = {}
            chat = update.effective_chat

            # Process group messages
            if self._group_chat_integration and chat and chat.type in ["group", "supergroup"]:
                try:
                    enhanced_message_text, group_metadata = await self._group_chat_integration.process_group_message(
                        update, context, message_text
                    )
                except Exception as e:
                    self.logger.error(f"Error processing group message: {e}")

            # Extract message entities
            message_entities = []
            if update.message and update.message.entities:
                message_entities = [
                    {
                        "type": entity.type,
                        "offset": entity.offset,
                        "length": entity.length,
                        "url": getattr(entity, "url", None),
                        "user": getattr(entity, "user", None),
                    }
                    for entity in update.message.entities
                ]

            # Check for bot mentions
            if BotUsernameHelper.is_bot_mentioned(enhanced_message_text, context, entities=message_entities):
                bot_username = BotUsernameHelper.get_bot_username(context, with_at=True)
                self.logger.info(f"Bot mentioned by user {user_id} using username: {bot_username}")

            # AI command routing
            if self.ai_command_router:
                try:
                    is_group_chat = chat and chat.type in ["group", "supergroup"]
                    is_mentioned = BotUsernameHelper.is_bot_mentioned(
                        enhanced_message_text, context, entities=message_entities
                    )
                    if not is_group_chat or is_mentioned:
                        has_attached_media = bool(
                            update.message
                            and (
                                update.message.photo
                                or update.message.video
                                or update.message.document
                                or update.message.audio
                                or update.message.voice
                            )
                        )
                        should_route = await self.ai_command_router.should_route_message(
                            message_text, has_attached_media
                        )
                        if should_route:
                            intent, confidence = await self.ai_command_router.detect_intent(
                                message_text, has_attached_media
                            )
                            command_executed = await self.ai_command_router.route_command(
                                update, context, intent, message_text
                            )
                            if command_executed:
                                await self.user_data_manager.update_stats(
                                    user_id, {"text_messages": 1, "total_messages": 1, "ai_commands": 1}
                                )
                                return
                except Exception as e:
                    self.logger.error(f"Error in AI command routing: {str(e)}")

            # Create TextHandler and process message
            text_handler = TextHandler(
                self.gemini_api,
                self.user_data_manager,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )

            if enhanced_message_text != message_text and chat and chat.type in ["group", "supergroup"]:
                context.user_data["group_context"] = group_metadata
                context.user_data["original_message"] = message_text
                context.user_data["enhanced_message"] = enhanced_message_text

            await text_handler.handle_text_message(update, context)
            await self.user_data_manager.update_stats(user_id, {"text_messages": 1, "total_messages": 1})

        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            await self._error_handler(update, context)
