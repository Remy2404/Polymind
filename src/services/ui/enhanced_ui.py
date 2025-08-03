"""
Enhanced UI/UX Components for Telegram Bot
Provides rich interactive elements and modern formatting
"""

import asyncio
from typing import Dict, List, Any
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EnhancedUI:
    """Enhanced UI components for better user experience"""

    def __init__(self):
        self.themes = {
            "dark": {"primary": "🔹", "secondary": "▫️", "accent": "🌟"},
            "light": {"primary": "🔸", "secondary": "▪️", "accent": "⭐"},
            "colorful": {"primary": "🟢", "secondary": "🔵", "accent": "🟡"},
        }

    async def create_smart_keyboard(
        self, options: List[Dict[str, Any]], layout: str = "grid", theme: str = "dark"
    ) -> InlineKeyboardMarkup:
        """Create intelligent keyboard layouts"""

        if layout == "carousel":
            return await self._create_carousel_keyboard(options, theme)
        elif layout == "smart_grid":
            return await self._create_smart_grid(options, theme)
        else:
            return await self._create_standard_grid(options, theme)

    async def _create_carousel_keyboard(
        self, options: List[Dict], theme: str
    ) -> InlineKeyboardMarkup:
        """Create a carousel-style keyboard"""
        keyboard = []

        # Add navigation if more than 3 options
        if len(options) > 3:
            nav_row = [
                InlineKeyboardButton("⬅️ Prev", callback_data="nav_prev"),
                InlineKeyboardButton(
                    f"1/{(len(options) - 1) // 3 + 1}", callback_data="nav_info"
                ),
                InlineKeyboardButton("Next ➡️", callback_data="nav_next"),
            ]
            keyboard.append(nav_row)

        # Add option buttons (3 per row for carousel)
        for i in range(0, min(3, len(options))):
            option = options[i]
            emoji = option.get("emoji", self.themes[theme]["primary"])
            text = f"{emoji} {option['text']}"
            keyboard.append(
                [InlineKeyboardButton(text, callback_data=option["callback"])]
            )

        return InlineKeyboardMarkup(keyboard)

    async def _create_smart_grid(
        self, options: List[Dict], theme: str
    ) -> InlineKeyboardMarkup:
        """Create an intelligent grid based on content"""
        keyboard = []

        # Analyze options to determine best layout
        avg_length = sum(len(opt["text"]) for opt in options) / len(options)

        if avg_length > 15:  # Long text = vertical layout
            cols = 1
        elif avg_length > 8:  # Medium text = 2 columns
            cols = 2
        else:  # Short text = 3 columns
            cols = 3

        for i in range(0, len(options), cols):
            row = []
            for j in range(cols):
                if i + j < len(options):
                    option = options[i + j]
                    emoji = option.get("emoji", self.themes[theme]["primary"])
                    text = f"{emoji} {option['text']}"
                    row.append(
                        InlineKeyboardButton(text, callback_data=option["callback"])
                    )
            keyboard.append(row)

        return InlineKeyboardMarkup(keyboard)

    async def _create_standard_grid(
        self, options: List[Dict], theme: str
    ) -> InlineKeyboardMarkup:
        """Create standard 2-column grid"""
        keyboard = []

        for i in range(0, len(options), 2):
            row = []
            for j in range(2):
                if i + j < len(options):
                    option = options[i + j]
                    emoji = option.get("emoji", self.themes[theme]["primary"])
                    text = f"{emoji} {option['text']}"
                    row.append(
                        InlineKeyboardButton(text, callback_data=option["callback"])
                    )
            keyboard.append(row)

        return InlineKeyboardMarkup(keyboard)


class ProgressIndicator:
    """Real-time progress indicators"""

    def __init__(self):
        self.active_indicators = {}

    async def create_progress_bar(
        self, percentage: int, width: int = 20, style: str = "modern"
    ) -> str:
        """Create animated progress bars"""

        if style == "modern":
            filled = int(percentage * width / 100)
            bar = "█" * filled + "░" * (width - filled)
            return f"🔄 {bar} {percentage}%"

        elif style == "dots":
            filled = int(percentage * width / 100)
            bar = "●" * filled + "○" * (width - filled)
            return f"⏳ {bar} {percentage}%"

        elif style == "arrows":
            filled = int(percentage * width / 100)
            bar = "▶" * filled + "▷" * (width - filled)
            return f"🚀 {bar} {percentage}%"

    async def start_spinner(self, message_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Start animated spinner"""
        self.active_indicators[message_id] = True
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        frame = 0

        while self.active_indicators.get(message_id, False):
            try:
                await context.bot.edit_message_text(
                    chat_id=context._chat_id,
                    message_id=message_id,
                    text=f"{spinner_frames[frame]} Processing...",
                )
                frame = (frame + 1) % len(spinner_frames)
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Spinner error: {e}")
                break

    async def stop_spinner(self, message_id: int):
        """Stop animated spinner"""
        self.active_indicators[message_id] = False


class ResponseFormatter:
    """Advanced response formatting"""

    def __init__(self):
        self.templates = {
            "success": "✅ **Success!**\n\n{content}",
            "error": "❌ **Error**\n\n{content}",
            "info": "ℹ️ **Information**\n\n{content}",
            "warning": "⚠️ **Warning**\n\n{content}",
            "loading": "⏳ **Processing**\n\n{content}",
            "complete": "🎉 **Complete!**\n\n{content}",
        }

    async def format_response(
        self, content: str, response_type: str = "info", style: str = "modern"
    ) -> str:
        """Format responses with advanced styling"""

        if style == "modern":
            return await self._format_modern(content, response_type)
        elif style == "card":
            return await self._format_card(content, response_type)
        elif style == "minimal":
            return await self._format_minimal(content, response_type)

        return content

    async def _format_modern(self, content: str, response_type: str) -> str:
        """Modern formatting with emojis and structure"""
        template = self.templates.get(response_type, self.templates["info"])
        formatted = template.format(content=content)

        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        formatted += f"\n\n🕐 {timestamp}"

        return formatted

    async def _format_card(self, content: str, response_type: str) -> str:
        """Card-style formatting"""
        border = "━" * 30

        icons = {
            "success": "✅",
            "error": "❌",
            "info": "ℹ️",
            "warning": "⚠️",
            "loading": "⏳",
            "complete": "🎉",
        }

        icon = icons.get(response_type, "ℹ️")

        return f"{border}\n{icon} **{response_type.upper()}**\n{border}\n\n{content}\n\n{border}"

    async def _format_minimal(self, content: str, response_type: str) -> str:
        """Minimal clean formatting"""
        icons = {
            "success": "✓",
            "error": "✗",
            "info": "i",
            "warning": "!",
            "loading": "⋯",
            "complete": "✓",
        }

        icon = icons.get(response_type, "i")
        return f"{icon} {content}"


class InteractiveElements:
    """Interactive UI elements"""

    def __init__(self):
        self.active_sessions = {}

    async def create_pagination(
        self, items: List[Any], page_size: int = 5, current_page: int = 0
    ) -> Dict[str, Any]:
        """Create paginated content"""

        total_pages = (len(items) - 1) // page_size + 1
        start_idx = current_page * page_size
        end_idx = start_idx + page_size

        page_items = items[start_idx:end_idx]

        # Create navigation buttons
        nav_buttons = []
        if current_page > 0:
            nav_buttons.append(
                {
                    "text": "⬅️ Previous",
                    "callback": f"page_{current_page - 1}",
                    "emoji": "⬅️",
                }
            )

        nav_buttons.append(
            {
                "text": f"{current_page + 1}/{total_pages}",
                "callback": "page_info",
                "emoji": "📄",
            }
        )

        if current_page < total_pages - 1:
            nav_buttons.append(
                {"text": "Next ➡️", "callback": f"page_{current_page + 1}", "emoji": "➡️"}
            )

        return {
            "items": page_items,
            "navigation": nav_buttons,
            "current_page": current_page,
            "total_pages": total_pages,
            "has_prev": current_page > 0,
            "has_next": current_page < total_pages - 1,
        }

    async def create_multi_step_form(
        self, steps: List[Dict[str, Any]], current_step: int = 0
    ) -> Dict[str, Any]:
        """Create multi-step form interface"""

        total_steps = len(steps)
        current = steps[current_step]

        # Progress indicator
        progress = await self._create_step_progress(current_step, total_steps)

        # Navigation buttons
        nav_buttons = []
        if current_step > 0:
            nav_buttons.append(
                {"text": "⬅️ Back", "callback": f"step_{current_step - 1}", "emoji": "⬅️"}
            )

        if current_step < total_steps - 1:
            nav_buttons.append(
                {"text": "Next ➡️", "callback": f"step_{current_step + 1}", "emoji": "➡️"}
            )
        else:
            nav_buttons.append(
                {"text": "✅ Complete", "callback": "form_complete", "emoji": "✅"}
            )

        return {
            "step_info": current,
            "progress": progress,
            "navigation": nav_buttons,
            "current_step": current_step,
            "total_steps": total_steps,
            "is_first": current_step == 0,
            "is_last": current_step == total_steps - 1,
        }

    async def _create_step_progress(self, current: int, total: int) -> str:
        """Create step progress indicator"""
        progress_chars = []

        for i in range(total):
            if i < current:
                progress_chars.append("✅")
            elif i == current:
                progress_chars.append("🔄")
            else:
                progress_chars.append("⭕")

        return " ".join(progress_chars)


class AdaptiveInterface:
    """AI-powered adaptive interface"""

    def __init__(self):
        self.user_preferences = {}
        self.interaction_patterns = {}

    async def learn_user_preferences(
        self, user_id: int, interaction_data: Dict[str, Any]
    ):
        """Learn from user interactions"""

        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "preferred_layout": "grid",
                "preferred_theme": "dark",
                "interaction_speed": "normal",
                "complexity_preference": "medium",
            }

        # Update preferences based on interactions
        prefs = self.user_preferences[user_id]

        # Analyze interaction patterns
        if "button_clicks" in interaction_data:
            clicks = interaction_data["button_clicks"]
            if len(clicks) > 5:  # Fast clicker
                prefs["interaction_speed"] = "fast"
            elif len(clicks) < 2:  # Slow/careful user
                prefs["interaction_speed"] = "slow"

        if "preferred_options" in interaction_data:
            options = interaction_data["preferred_options"]
            if "simple" in str(options).lower():
                prefs["complexity_preference"] = "simple"
            elif "advanced" in str(options).lower():
                prefs["complexity_preference"] = "advanced"

    async def get_adaptive_interface(
        self, user_id: int, context: str = "general"
    ) -> Dict[str, Any]:
        """Get interface adapted to user preferences"""

        prefs = self.user_preferences.get(
            user_id,
            {
                "preferred_layout": "grid",
                "preferred_theme": "dark",
                "interaction_speed": "normal",
                "complexity_preference": "medium",
            },
        )

        # Adapt interface based on preferences
        interface_config = {
            "layout": prefs["preferred_layout"],
            "theme": prefs["preferred_theme"],
            "animation_speed": self._get_animation_speed(prefs["interaction_speed"]),
            "button_complexity": self._get_button_complexity(
                prefs["complexity_preference"]
            ),
            "auto_advance": prefs["interaction_speed"] == "fast",
        }

        return interface_config

    def _get_animation_speed(self, speed: str) -> float:
        """Get animation speed based on user preference"""
        speeds = {"slow": 1.5, "normal": 1.0, "fast": 0.5}
        return speeds.get(speed, 1.0)

    def _get_button_complexity(self, complexity: str) -> str:
        """Get button complexity based on user preference"""
        if complexity == "simple":
            return "minimal"
        elif complexity == "advanced":
            return "detailed"
        else:
            return "standard"


# Global instances
enhanced_ui = EnhancedUI()
progress_indicator = ProgressIndicator()
response_formatter = ResponseFormatter()
interactive_elements = InteractiveElements()
adaptive_interface = AdaptiveInterface()
