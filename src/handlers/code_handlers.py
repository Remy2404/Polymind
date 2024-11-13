import sys
import os
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.services.gemini_api import GeminiAPI
from src.utils.telegramlog import telegram_logger

class CodeHandler:
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api
        telegram_logger.log_message(0, "CodeHandler initialized")

    async def generate_code(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        telegram_logger.log_command(user_id, "/code")

        if not update.message or not context.args:
            await update.message.reply_text("Usage: /code [language] [prompt]\nExample: /code python 'create a hello world program'")
            return

        try:
            language = context.args[0]
            prompt = ' '.join(context.args[1:])
            
            # Send typing indicator
            await update.message.chat.send_action(action="typing")
            
            result = await self.gemini_api.generate_code(
                language=language,
                prompt=prompt,
                include_explanations=True
            )
            
            # Send code block
            if result['code']:
                await update.message.reply_text(
                    f"```{language}\n{result['code']}\n```",
                    parse_mode='Markdown'
                )
                
                # Send explanation separately if available
                if result['explanation']:
                    await update.message.reply_text(
                        f"Explanation:\n{result['explanation']}"
                    )
                    
            telegram_logger.log_message(user_id, f"Code generated successfully for language: {language}")
            
        except Exception as e:
            error_message = f"Error generating code: {str(e)}"
            telegram_logger.log_error(user_id, error_message)
            await update.message.reply_text(
                "I encountered an error while generating the code. Please try again."
            )

    def get_handlers(self):
        return [
            CommandHandler('code', self.generate_code)
        ]

# Create handler instance
def create_code_handler(gemini_api: GeminiAPI):
    return CodeHandler(gemini_api)
