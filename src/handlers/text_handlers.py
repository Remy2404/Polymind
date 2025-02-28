from telegram import Update
from telegram.ext import ContextTypes, MessageHandler, filters
from telegram.constants import ChatAction
from utils.telegramlog import telegram_logger
from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
from typing import List
import datetime
import logging

class TextHandler:
    def __init__(self, gemini_api: GeminiAPI, user_data_manager: UserDataManager):
        self.logger = logging.getLogger(__name__)
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.max_context_length = 5

    async def format_telegram_markdown(self, text: str) -> str:
        try:
            from telegramify_markdown import convert
            formatted_text = convert(text)
            return formatted_text
        except Exception as e:
            self.logger.error(f"Error formatting markdown: {str(e)}")
            return text.replace('*', '').replace('_', '').replace('`', '')

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        for line in text.split('\n'):
            if len(current_chunk) + len(line) + 1 > max_length:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line
        
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message and not update.edited_message:
            return
            
        user_id = update.effective_user.id
        message = update.message or update.edited_message
        message_text = message.text

        try:
            # Delete old message if this is an edited message
            if update.edited_message and 'bot_messages' in context.user_data:
                original_message_id = update.edited_message.message_id
                if original_message_id in context.user_data['bot_messages']:
                    for msg_id in context.user_data['bot_messages'][original_message_id]:
                        try:
                            await context.bot.delete_message(
                                chat_id=update.effective_chat.id,
                                message_id=msg_id
                            )
                        except Exception as e:
                            self.logger.error(f"Error deleting old message: {str(e)}")
                    del context.user_data['bot_messages'][original_message_id]

            # In group chats, process only messages that mention the bot
            if update.effective_chat.type in ['group', 'supergroup']:
                bot_username = '@' + context.bot.username
                if bot_username not in message_text:
                    # Bot not mentioned, ignore message
                    return
                else:
                    # Remove all mentions of bot_username from the message text
                    message_text = message_text.replace(bot_username, '').strip()

            # Check if user is referring to images
            image_related_keywords = ['image', 'picture', 'photo', 'pic', 'img', 'that image', 'the picture']
            referring_to_image = any(keyword in message_text.lower() for keyword in image_related_keywords)
            
            # Send initial "thinking" message
            thinking_message = await message.reply_text("Thinking...🧠")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            
            # Get user context
            user_context = self.user_data_manager.get_user_context(user_id)
            
            # If user is referring to images, include image context
            enhanced_prompt = message_text
            if referring_to_image and 'image_history' in context.user_data and context.user_data['image_history']:
                image_context = await self.get_image_context(user_id, context)
                enhanced_prompt = f"The user is referring to previously shared images. Here's the context of those images:\n\n{image_context}\n\nUser's question: {message_text}"
            
            # Generate response
            response = await self.gemini_api.generate_response(
                prompt=enhanced_prompt,
                context=user_context[-self.max_context_length:]
            )

            if response is None:
                await thinking_message.delete()
                await message.reply_text(
                    "Sorry, I couldn't generate a response\\. Please try rephrasing your message\\.",
                    parse_mode='MarkdownV2'
                )
                return

            # Split long messages
            message_chunks = await self.split_long_message(response)

            # Delete thinking message
            await thinking_message.delete()

            # Store the message IDs for potential editing
            sent_messages = []
            
            last_message = None
            for i, chunk in enumerate(message_chunks):
                try:
                    # Format with telegramify-markdown
                    formatted_chunk = await self.format_telegram_markdown(chunk)
                    if i == 0:
                        last_message = await message.reply_text(
                            formatted_chunk,
                            parse_mode='MarkdownV2',
                            disable_web_page_preview=True,
                        )
                    else:
                        last_message = await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=formatted_chunk,
                            parse_mode='MarkdownV2',
                            disable_web_page_preview=True,
                        )
                    sent_messages.append(last_message)
                except Exception as formatting_error:
                    self.logger.error(f"Formatting failed: {str(formatting_error)}")
                    if i == 0:
                        last_message = await message.reply_text(
                            chunk.replace('*', '').replace('_', '').replace('`', ''),
                            parse_mode=None
                        )
                    else:
                        last_message = await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=chunk.replace('*', '').replace('_', '').replace('`', ''),
                            parse_mode=None
                        )
                    sent_messages.append(last_message)

            # Update user context only if response was successful
            if response:
                self.user_data_manager.add_to_context(user_id, {"role": "user", "content": message_text})
                self.user_data_manager.add_to_context(user_id, {"role": "assistant", "content": response})

                # Store the message IDs in context for future editing
                if 'bot_messages' not in context.user_data:
                    context.user_data['bot_messages'] = {}
                context.user_data['bot_messages'][message.message_id] = [msg.message_id for msg in sent_messages]

            telegram_logger.log_message(f"Text response sent successfully", user_id)
        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            if 'thinking_message' in locals():
                await thinking_message.delete()
            await update.message.reply_text(
                "Sorry, I encountered an error\\. Please try again later\\.",
                parse_mode='MarkdownV2'
            )
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            user_id = update.effective_user.id
            telegram_logger.log_message("Processing an image", user_id)
        
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
            try:
                # In group chats, process only images that mention the bot
                if update.effective_chat.type in ['group', 'supergroup']:
                    bot_username = '@' + context.bot.username
                    caption = update.message.caption or ""
                    if bot_username not in caption:
                        # Bot not mentioned, ignore message
                        return
                    else:
                        # Remove all mentions of bot_username from the caption
                        caption = caption.replace(bot_username, '').strip()
                else:
                    caption = update.message.caption or "Please analyze this image and describe it."
        
                photo = update.message.photo[-1]
                image_file = await context.bot.get_file(photo.file_id)
                image_bytes = await image_file.download_as_bytearray()
                
                response = await self.gemini_api.analyze_image(image_bytes, caption)
        
                if response:
                    # Split the response into chunks
                    response_chunks = await self.split_long_message(response)
                    sent_messages = []
                    
                    # Send each chunk
                    for chunk in response_chunks:
                        try:
                            formatted_chunk = await self.format_telegram_markdown(chunk)
                            sent_message = await update.message.reply_text(
                                formatted_chunk,
                                parse_mode='MarkdownV2',
                                disable_web_page_preview=True
                            )
                            sent_messages.append(sent_message.message_id)
                        except Exception as markdown_error:
                            self.logger.warning(f"Markdown formatting failed: {markdown_error}")
                            # Try without markdown if formatting fails
                            sent_message = await update.message.reply_text(chunk, parse_mode=None)
                            sent_messages.append(sent_message.message_id)
        
                    # Store image info in user context
                    self.user_data_manager.add_to_context(user_id, {"role": "user", "content": f"[Image with caption: {caption}]"})
                    self.user_data_manager.add_to_context(user_id, {"role": "assistant", "content": response})
                    
                    # Store image reference in user data for future reference
                    if 'image_history' not in context.user_data:
                        context.user_data['image_history'] = []
                    
                    # Store image metadata
                    context.user_data['image_history'].append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'file_id': photo.file_id,
                        'caption': caption,
                        'description': response,
                        'message_id': update.message.message_id,
                        'response_message_ids': sent_messages  # Now storing all message IDs
                    })
        
                    # Update user stats for image
                    if self.user_data_manager:
                        self.user_data_manager.update_stats(user_id, image=True)
        
                    telegram_logger.log_message(f"Image analysis completed successfully", user_id)
                else:
                    await update.message.reply_text("Sorry, I couldn't analyze the image. Please try again.")
        
            except Exception as e:
                self.logger.error(f"Error processing image: {e}")
                await update.message.reply_text(
                    "Sorry, I couldn't process your image. The response might be too long or there might be an issue with the image format. Please try a different image or a more specific question."
                )   
    async def show_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        history = self.user_data_manager.get_user_context(user_id)
        
        if not history:
            await update.message.reply_text("You don't have any conversation history yet.")
            return

        history_text = "Your conversation history:\n\n"
        for entry in history:
            role = entry['role'].capitalize()
            content = entry['content']
            history_text += f"{role}: {content}\n\n"

        # Split long messages
        message_chunks = await self.split_long_message(history_text)

        for chunk in message_chunks:
            await update.message.reply_text(chunk)

    async def get_image_context(self, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Generate context from previously processed images"""
        if 'image_history' not in context.user_data or not context.user_data['image_history']:
            return ""
        
        # Get the 3 most recent images 
        recent_images = context.user_data['image_history'][-3:]
        
        image_context = "Recently analyzed images:\n"
        for idx, img in enumerate(recent_images):
            image_context += f"[Image {idx+1}]: Caption: {img['caption']}\nDescription: {img['description'][:100]}...\n\n"
        
        return image_context

    def get_handlers(self):
        return [
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message),
            MessageHandler(filters.PHOTO, self.handle_image),
        ]