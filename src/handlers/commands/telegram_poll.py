"""
AI-Powered Poll Creation Command Handler
Generates and sends Telegram polls based on user prompts using AI models.
"""
import sys
import os
import logging
from typing import Dict, Any, Optional
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.services.model_handlers.simple_api_manager import SuperSimpleAPIManager
from src.services.user_data_manager import UserDataManager
from src.utils.log.telegramlog import TelegramLogger
logger = logging.getLogger(__name__)
class PollCommands:
    """
    Handles AI-powered poll creation commands.
    Follows single responsibility principle by focusing solely on poll generation and sending.
    """
    def __init__(
        self,
        api_manager: SuperSimpleAPIManager,
        user_data_manager: UserDataManager,
        telegram_logger: TelegramLogger,
    ):
        """
        Initialize PollCommands with required dependencies.
        Args:
            api_manager: API manager for AI model interactions
            user_data_manager: User data management service
            telegram_logger: Logging service for user actions
        """
        self.api_manager = api_manager
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
    async def create_poll_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle the /createpoll command.
        Generates a poll using AI based on user prompt and sends it via Telegram.
        Args:
            update: Telegram update object
            context: Telegram context object
        """
        user_id = update.effective_user.id
        self.telegram_logger.log_message(
            f"Poll creation requested by user {user_id}", user_id
        )
        if not context.args:
            await self._send_help_message(update, context)
            return
        prompt = " ".join(context.args)
        try:
            await self.user_data_manager.initialize_user(user_id)
            thinking_message = await update.message.reply_text(
                "🤖 *Generating your poll...*\n\n"
                "I'm creating a poll based on your request. This may take a moment...",
                parse_mode=ParseMode.MARKDOWN,
            )
            poll_data = await self._generate_poll_with_ai(prompt, user_id)
            if not poll_data:
                await thinking_message.edit_text(
                    "❌ Sorry, I couldn't generate a poll from your request. Please try rephrasing it."
                )
                return
            await thinking_message.delete()
            await self._send_poll(update, context, poll_data)
            self.telegram_logger.log_message(
                f"Poll created successfully: {poll_data['question']}", user_id
            )
        except Exception as e:
            logger.error(f"Error creating poll: {str(e)}")
            await update.message.reply_text(
                f"❌ Sorry, there was an error creating your poll: {str(e)}"
            )
    async def _generate_poll_with_ai(
        self, prompt: str, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Generate poll content using AI model.
        Args:
            prompt: User's poll creation prompt
            user_id: User ID for context
        Returns:
            Dictionary containing poll question and options, or None if failed
        """
        preferred_model = await self._get_user_preferred_model(user_id)
        ai_prompt = (
            "You are a JSON-only API. Create a Telegram poll based on the user's request.\n\n"
            "IMPORTANT: Respond with ONLY a valid JSON object. No explanations, no markdown, no extra text.\n\n"
            "JSON format required:\n"
            "{\n"
            '  "question": "Your poll question here",\n'
            '  "options": ["Option 1", "Option 2", "Option 3"]\n'
            "}\n\n"
            "Rules:\n"
            "- question: string, engaging and clear\n"
            "- options: array of 2-10 strings\n"
            "- Make options relevant to the request\n"
            "- Keep options concise\n\n"
            f"User request: {prompt}"
        )
        try:
            response = await self.api_manager.chat(
                model_id=preferred_model,
                prompt=ai_prompt,
                temperature=0.7,
                max_tokens=1000,
            )

            # Clean the response - remove any markdown formatting or extra text
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            import json
            poll_data = json.loads(response)

            if (
                not isinstance(poll_data, dict)
                or "question" not in poll_data
                or "options" not in poll_data
            ):
                logger.warning(f"Invalid poll data structure from AI: {poll_data}")
                return None

            if (
                not isinstance(poll_data["options"], list)
                or len(poll_data["options"]) < 2
                or len(poll_data["options"]) > 10
            ):
                logger.warning(f"Invalid options in poll data: {poll_data}")
                return None

            # Validate that question and options are strings
            if not isinstance(poll_data["question"], str) or not poll_data["question"].strip():
                logger.warning(f"Invalid question in poll data: {poll_data}")
                return None

            poll_data["options"] = [str(opt).strip() for opt in poll_data["options"] if str(opt).strip()]
            if len(poll_data["options"]) < 2:
                logger.warning(f"Not enough valid options after cleanup: {poll_data}")
                return None

            return poll_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"AI response was: {response}")

            # Try to extract JSON from the response if it contains extra text
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    poll_data = json.loads(json_match.group())
                    if (
                        isinstance(poll_data, dict)
                        and "question" in poll_data
                        and "options" in poll_data
                        and isinstance(poll_data["options"], list)
                        and len(poll_data["options"]) >= 2
                    ):
                        logger.info("Successfully extracted JSON from AI response")
                        return poll_data
                except json.JSONDecodeError:
                    pass

            # Fallback: Create a simple poll based on the prompt
            logger.info("Using fallback poll generation")
            return self._create_fallback_poll(prompt)
        except Exception as e:
            logger.error(f"Error generating poll with AI: {e}")
            # Fallback: Create a simple poll based on the prompt
            logger.info("Using fallback poll generation due to error")
            return self._create_fallback_poll(prompt)
    async def _send_poll(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        poll_data: Dict[str, Any],
    ) -> None:
        """
        Send the generated poll via Telegram API.
        Args:
            update: Telegram update object
            context: Telegram context object
            poll_data: Poll data containing question and options
        """
        try:
            await context.bot.send_poll(
                chat_id=update.effective_chat.id,
                question=poll_data["question"],
                options=poll_data["options"],
                is_anonymous=True,
                allows_multiple_answers=False,
                reply_to_message_id=update.message.message_id,
            )
        except Exception as e:
            logger.error(f"Error sending poll: {e}")
            await update.message.reply_text(
                "❌ Sorry, there was an error sending the poll. Please try again."
            )
    async def _get_user_preferred_model(self, user_id: int) -> str:
        """
        Get user's preferred AI model for poll generation.
        Args:
            user_id: User ID
        Returns:
            Model ID string, defaults to 'gemini' if not set
        """
        try:
            from src.services.user_preferences_manager import UserPreferencesManager
            preferences_manager = UserPreferencesManager(self.user_data_manager)
            preferred_model = await preferences_manager.get_user_model_preference(
                user_id
            )
            return preferred_model or "gemini"
        except Exception:
            return "gemini"
    async def _send_help_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Send help message for poll creation command.
        Args:
            update: Telegram update object
            context: Telegram context object
        """
        help_text = (
            "🤖 *AI Poll Creator*\n\n"
            "Create polls automatically using AI! Just describe what you want.\n\n"
            "📝 *Usage:*\n"
            "`/createpoll <your poll description>`\n\n"
            "💡 *Examples:*\n"
            "• `/createpoll What new feature should we add next?`\n"
            "• `/createpoll Ask users their favorite programming language: Python, JavaScript, Java, or C++`\n"
            "• `/createpoll Should we implement dark mode?`\n"
            "• `/createpoll What's your favorite way to learn coding?`\n\n"
            "🎯 *Features:*\n"
            "• AI generates engaging questions and options automatically\n"
            "• Anonymous voting enabled\n"
            "• Single choice polls\n"
            "• Supports 2-10 options\n\n"
            "Try it now!"
        )
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    def _create_fallback_poll(self, prompt: str) -> dict:
        """Create a simple fallback poll when AI fails to generate valid JSON."""
        # Extract a question from the prompt
        question = prompt.strip()
        if len(question) > 100:
            question = question[:97] + "..."
        if not question.endswith("?"):
            question += "?"

        # Create simple yes/no options
        options = ["Yes", "No"]

        # Try to make it more relevant based on the prompt
        if any(word in prompt.lower() for word in ["what", "which", "choose", "prefer"]):
            options = ["Option A", "Option B", "Option C"]
        elif any(word in prompt.lower() for word in ["rate", "how", "scale"]):
            options = ["1", "2", "3", "4", "5"]

        return {
            "question": question,
            "options": options
        }
