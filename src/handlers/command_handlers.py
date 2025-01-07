# src/handlers/command_handlers.py

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler, Application
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from utils.telegramlog import  TelegramLogger as telegram_logger
import logging
import re
from io import BytesIO
import requests
from services.flux_lora_img import FluxLoraImageGenerator as flux_lora_image_generator
import time , asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cachetools import TTLCache
from typing import Optional
from PIL import Image
import io

@dataclass
class ImageRequest:
    prompt: str
    width: int
    height: int
    steps: int
    timestamp: float = field(default_factory=time.time)

class ImageGenerationHandler:
    def __init__(self):
        self.request_cache = TTLCache(maxsize=100, ttl=3600)
        self.request_limiter = {}
        self.processing_queue = asyncio.Queue()
        self.rate_limit_time = 30
        
    def is_rate_limited(self, user_id: int) -> bool:
        if user_id in self.request_limiter:
            last_request = self.request_limiter[user_id]
            if datetime.now() - last_request < timedelta(seconds=self.rate_limit_time):
                return True
        return False

    def update_rate_limit(self, user_id: int) -> None:
        self.request_limiter[user_id] = datetime.now()

    def get_cached_image(self, prompt: str, width: int, height: int, steps: int) -> Optional[Image.Image]:
        cache_key = f"{prompt}_{width}_{height}_{steps}"
        return self.request_cache.get(cache_key)

    def cache_image(self, prompt: str, width: int, height: int, steps: int, image: Image.Image) -> None:
        cache_key = f"{prompt}_{width}_{height}_{steps}"
        self.request_cache[cache_key] = image

class CommandHandlers:
    def __init__(self, gemini_api: GeminiAPI, user_data_manager: UserDataManager , telegram_logger:telegram_logger,flux_lora_image_generator: flux_lora_image_generator):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.flux_lora_image_generator = flux_lora_image_generator
        self.logger = logging.getLogger(__name__)
        self.telegram_logger = telegram_logger
        self.image_handler = ImageGenerationHandler()
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user:
            return

        user_id = update.effective_user.id
        welcome_message = (
            "👋 Welcome to GemBot! I'm your AI assistant powered by Gemini.\n\n"
            "I can help you with:\n"
            "🤖 General conversations\n"
            "📝 Code assistance\n"
            "🗣️ Voice to text conversion\n"
            "🖼️ Image analysis\n\n"
            "Feel free to start chatting or use /help to learn more!"
        )

        keyboard = [
            [
                InlineKeyboardButton("Help 📚", callback_data='help'),
                InlineKeyboardButton("Settings ⚙️", callback_data='preferences')
            ],
            [InlineKeyboardButton("Support Channel 📢", url='https://t.me/GemBotAI')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        if update.callback_query:
            await update.callback_query.message.reply_text(welcome_message, reply_markup=reply_markup)
        else:
            await update.message.reply_text(welcome_message, reply_markup=reply_markup)
            
        await self.user_data_manager.initialize_user(user_id)
        self.logger.info(f"New user started the bot: {user_id}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        help_text = (
            "🤖 Available Commands\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/reset - Reset conversation history\n"
            "/settings - Configure bot settings\n"
            "/stats - Show bot statistics\n\n"
            "/generate_image <prompt> - Generate an image from text\n"
            "/export - Export conversation history\n"
            "💡 Features\n"
            "• Send text messages for general conversation\n"
            "• Send images for analysis\n"
            "• Supports markdown formatting\n\n"
            "Need more help? Join our support channel!"
        )
        if update.callback_query:
            await update.callback_query.message.reply_text(help_text)
        else:
            await update.message.reply_text(help_text)

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        self.user_data_manager.reset_conversation(user_id)
        await update.message.reply_text("Conversation history has been reset!")

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        settings = self.user_data_manager.get_user_settings(user_id)
    
        keyboard = [
            [
                InlineKeyboardButton(
                    f"{'🔵' if settings.get('markdown_enabled', True) else '⚪'} Markdown Mode",
                    callback_data='toggle_markdown'
                )
            ],
            [
                InlineKeyboardButton(
                    f"{'🔵' if settings.get('code_suggestions', True) else '⚪'} Code Suggestions",
                    callback_data='toggle_code_suggestions'
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
    
        settings_text = "⚙️ *Bot Settings*\nCustomize your interaction preferences:"
    
        if update.callback_query:
            await update.callback_query.message.reply_text(
                settings_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                settings_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        self.telegram_logger.log_message(user_id, "Opened settings menu")
    async def handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        user_data = self.user_data_manager.get_user_data(user_id)
        stats = user_data.get('stats', {})

        stats_message = (
            "📊 Your Bot Usage Statistics:\n\n"
            f"📝 Text Messages: {stats.get('messages', 0)}\n"
            f"🎤 Voice Messages: {stats.get('voice_messages', 0)}\n"
            f"🖼 Images Processed: {stats.get('images', 0)}\n"
            f"📑 PDFs Analyzed: {stats.get('pdfs_processed', 0)}\n"
            f"Last Active: {stats.get('last_active', 'Never')}"
        )

        await update.message.reply_text(stats_message)

    async def handle_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        user_data = self.user_data_manager.get_user_data(user_id)
        history = user_data.get('conversation_history', [])

        # Create formatted export
        export_text = "💬 Conversation History:\n\n"
        for msg in history:
            export_text += f"User: {msg['user']}\n"
            export_text += f"Bot: {msg['bot']}\n\n"

        # Send as file if too long
        if len(export_text) > 4000:
            with open(f'history_{user_id}.txt', 'w') as f:
                f.write(export_text)
            await update.message.reply_document(
                document=open(f'history_{user_id}.txt', 'rb'),
                filename='conversation_history.txt'
            )
        else:
            await update.message.reply_text(export_text)

    async def handle_preferences(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle user preferences command"""
        keyboard = [
            [InlineKeyboardButton("Language", callback_data="pref_language"),
             InlineKeyboardButton("Response Format", callback_data="pref_format")],
            [InlineKeyboardButton("Notifications", callback_data="pref_notifications"),
             InlineKeyboardButton("AI Model", callback_data="pref_model")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "⚙️ *User Preferences*\n\n"
            "Select a setting to modify:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()

        data = query.data

        if data == 'help_command':
            await self.help_command(update, context)
        elif data == 'settings':
            await self.settings(update, context)
        elif data in ['toggle_markdown', 'toggle_code_suggestions']:
            # Implement toggle logic here
            user_id = update.effective_user.id
            if data == 'toggle_markdown':
                current_value = self.user_data_manager.get_user_settings(user_id).get('markdown_enabled', True)
                self.user_data_manager.set_user_setting(user_id, 'markdown_enabled', not current_value)
                status = 'enabled' if not current_value else 'disabled'
                await query.edit_message_text(f"Markdown Mode has been {status}.")
            elif data == 'toggle_code_suggestions':
                current_value = self.user_data_manager.get_user_settings(user_id).get('code_suggestions', True)
                self.user_data_manager.set_user_setting(user_id, 'code_suggestions', not current_value)
                status = 'enabled' if not current_value else 'disabled'
                await query.edit_message_text(f"Code Suggestions have been {status}.")
        elif data.startswith('img_'):
            await self.handle_image_settings(update, context, data)
        elif data.startswith('pref_'):
            await self.handle_user_preferences(update, context, data)
        else:
            await query.edit_message_text("Unknown action.")

    async def generate_image_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        
        if self.image_handler.is_rate_limited(user_id):
            remaining_time = self.image_handler.rate_limit_time
            await update.message.reply_text(
                f"Please wait {remaining_time} seconds before generating another image."
            )
            return

        prompt = ' '.join(context.args)
        if not prompt:
            await update.message.reply_text("Please provide a prompt. Usage: /generate_image <your prompt>")
            return

        if len(prompt) > 500:
            await update.message.reply_text("Prompt is too long. Please limit to 500 characters.")
            return

        # Store the prompt in user_data for later use
        context.user_data['image_prompt'] = prompt

        # Create a preview message with confirmation buttons
        preview_message = (
            f"📝 Your image generation prompt:\n\n"
            f"'{prompt}'\n\n"
            f"Do you want to proceed with this prompt?"
        )

        keyboard = [
            [
                InlineKeyboardButton("✅ Confirm", callback_data="confirm_image_prompt"),
                InlineKeyboardButton("❌ Cancel", callback_data="cancel_image_prompt")
            ],
            [
                InlineKeyboardButton("✏️ Edit Prompt", callback_data="edit_image_prompt")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            preview_message,
            reply_markup=reply_markup
        )

    async def handle_image_prompt_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()

        if query.data == "confirm_image_prompt":
            # Proceed with image generation
            await self.show_image_quality_options(update, context)
        elif query.data == "cancel_image_prompt":
            await query.edit_message_text("Image generation cancelled.")
        elif query.data == "edit_image_prompt":
            await query.edit_message_text(
                "Please send your updated prompt. You can cancel by sending /cancel."
            )
            context.user_data['awaiting_prompt_edit'] = True

    async def show_image_quality_options(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        keyboard = [
            [
                InlineKeyboardButton(
                    "📱 Standard - Quick (20 steps)", 
                    callback_data="img_256_steps_20"
                )
            ],
            [
                InlineKeyboardButton(
                    "📱 Standard - Detailed (50 steps)", 
                    callback_data="img_256_steps_50"
                )
            ],
            [
                InlineKeyboardButton(
                    "🖥️ HD - Quick (20 steps)", 
                    callback_data="img_512_steps_20"
                )
            ],
            [
                InlineKeyboardButton(
                    "🖥️ HD - Detailed (50 steps)", 
                    callback_data="img_512_steps_50"
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.callback_query.edit_message_text(
            "Choose image quality and generation settings:",
            reply_markup=reply_markup
        )

    async def handle_image_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        
        start_time = time.time()
        data = query.data
        
        match = re.match(r'img_(\d+)_steps_(\d+)', data)
        if not match:
            await query.edit_message_text(
                "Invalid selection. Please try again with /generate_image."
            )
            return
    
        width = height = int(match.group(1))
        steps = int(match.group(2))
        prompt = context.user_data.get('image_prompt', '')
    
        if not prompt:
            await query.edit_message_text(
                "No image prompt found. Please use /generate_image command first."
            )
            return
    
        cached_image = self.image_handler.get_cached_image(prompt, width, height, steps)
        if cached_image:
            await self._send_image(
                update, 
                context, 
                cached_image, 
                width, 
                height, 
                steps,
                "Retrieved from cache"
            )
            return
    
        progress_message = await query.edit_message_text(
            "🖌️ Generating image...\n\n"
            "⏳ Initializing..."
        )
    
        try:
            progress_task = asyncio.create_task(
                self._update_progress(progress_message, steps)
            )
    
            images = await self.flux_lora_image_generator.text_to_image(
                prompt=prompt,
                num_images=1,
                num_inference_steps=steps,
                width=width,
                height=height
            )
    
            progress_task.cancel()
    
            if not images:
                await progress_message.edit_text(
                    "Failed to generate image. Please try again."
                )
                return
    
            self.image_handler.cache_image(prompt, width, height, steps, images[0])
            
            generation_time = time.time() - start_time
            await self._send_image(
                update, 
                context, 
                images[0], 
                width, 
                height, 
                steps,
                f"Generated in {generation_time:.1f}s"
            )
    
            await progress_message.delete()
    
        except asyncio.CancelledError:
            await progress_message.edit_text(
                "Image generation was cancelled."
            )
        except Exception as e:
            self.logger.error(f"Error generating image: {e}")
            await progress_message.edit_text(
                "Sorry, I couldn't generate the image. Please try again later."
            )

    async def _update_progress(self, message, total_steps: int) -> None:
        try:
            for step in range(1, total_steps + 1):
                await asyncio.sleep(0.5)
                progress = step / total_steps * 100
                await message.edit_text(
                    f"🖌️ Generating image...\n\n"
                    f"Progress: {progress:.0f}%\n"
                    f"Step {step}/{total_steps}"
                )
        except asyncio.CancelledError:
            pass

    async def _send_image(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        image: Image.Image,
        width: int,
        height: int,
        steps: int,
        status: str
    ) -> None:
        with io.BytesIO() as output:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(
                output, 
                format='JPEG', 
                optimize=True
            )
            output.seek(0)
            
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=InputFile(output),
                caption=f"Generated image ({width}x{height}, {steps} steps)\n{status}"
            )

    async def handle_user_preferences(self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str) -> None:
        # Implement preference handlers
        # Placeholder for now
        await update.callback_query.edit_message_text("Preference settings not implemented yet.")

    def register_handlers(self, application: Application) -> None:
        try:
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("reset", self.reset_command))
            application.add_handler(CommandHandler("settings", self.settings))
            application.add_handler(CommandHandler("stats", self.handle_stats))
            application.add_handler(CommandHandler("export", self.handle_export))
            application.add_handler(CommandHandler("preferences", self.handle_preferences))
            application.add_handler(CommandHandler("generate_image", self.generate_image_command))
            application.add_handler(CallbackQueryHandler(self.handle_image_prompt_callback, pattern="^(confirm|cancel|edit)_image_prompt$"))
            application.add_handler(CallbackQueryHandler(self.handle_image_settings, pattern="^img_.+_steps_.+$"))
            application.add_handler(CallbackQueryHandler(self.handle_callback_query))
        
            
            self.logger.info("Command handlers registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register command handlers: {e}")
            raise