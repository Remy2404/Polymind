from telegram import Update
from telegram.ext import ContextTypes, CommandHandler
from services.gemini_api import GeminiAPI

class CodeHandler:
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api

    async def generate_code(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not context.args:
            await update.message.reply_text("Please provide language and prompt. Example: /code python 'create a hello world program'")
            return

        try:
            language = context.args[0]
            prompt = ' '.join(context.args[1:])
            
            result = await self.gemini_api.generate_code(
                language=language,
                prompt=prompt,
                include_explanations=True
            )
            
            await update.message.reply_text(f"Generated Code:\n```{result['code']}```\n\nExplanation:\n{result['explanation']}")
            
        except Exception as e:
            await update.message.reply_text(f"Error generating code: {str(e)}")

    def get_handlers(self):
        return [
            CommandHandler('code', self.generate_code)
        ]
