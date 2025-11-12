"""
Rich UI Components for Enhanced User Experience
Provides interactive menus, progress indicators, and smart suggestions
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes
import time

logger = logging.getLogger(__name__)


class ProgressIndicator:
    """Visual progress indicators for long-running operations"""

    def __init__(self):
        self.active_operations = {}
        self.progress_symbols = ["⏳", "⏲️", "⌛", "🔄"]

    async def start_progress(
        self,
        operation_id: str,
        message,
        total_steps: int,
        description: str = "Processing...",
    ) -> str:
        """Start a progress indicator for an operation"""
        self.active_operations[operation_id] = {
            "message": message,
            "total_steps": total_steps,
            "current_step": 0,
            "description": description,
            "start_time": time.time(),
        }
        progress_text = f"{self.progress_symbols[0]} {description}\n▱▱▱▱▱▱▱▱▱▱ 0%"
        await message.edit_text(progress_text)
        return operation_id

    async def update_progress(
        self, operation_id: str, step: int, status_text: Optional[str] = None
    ) -> None:
        """Update progress for an operation"""
        if operation_id not in self.active_operations:
            return
        operation = self.active_operations[operation_id]
        operation["current_step"] = step
        progress_percent = min(100, int((step / operation["total_steps"]) * 100))
        filled_bars = int(progress_percent / 10)
        empty_bars = 10 - filled_bars
        progress_bar = "▰" * filled_bars + "▱" * empty_bars
        symbol_index = min(
            len(self.progress_symbols) - 1, step % len(self.progress_symbols)
        )
        symbol = self.progress_symbols[symbol_index]
        elapsed = time.time() - operation["start_time"]
        if step > 0:
            estimated_total = (elapsed / step) * operation["total_steps"]
            remaining = max(0, estimated_total - elapsed)
            time_text = f" (⏱️ {remaining:.0f}s remaining)" if remaining > 5 else ""
        else:
            time_text = ""
        description = status_text or operation["description"]
        progress_text = (
            f"{symbol} {description}\n{progress_bar} {progress_percent}%{time_text}"
        )
        try:
            await operation["message"].edit_text(progress_text)
        except Exception as e:
            logger.warning(f"Failed to update progress: {e}")

    async def complete_progress(
        self, operation_id: str, final_message: str, success: bool = True
    ) -> None:
        """Complete a progress operation"""
        if operation_id not in self.active_operations:
            return
        operation = self.active_operations[operation_id]
        symbol = "✅" if success else "❌"
        elapsed = time.time() - operation["start_time"]
        time_text = f" (⏱️ {elapsed:.1f}s)"
        final_text = f"{symbol} {final_message}{time_text}"
        try:
            await operation["message"].edit_text(final_text)
        except Exception as e:
            logger.warning(f"Failed to complete progress: {e}")
        del self.active_operations[operation_id]


class SmartSuggestions:
    """Intelligent action suggestions based on context"""

    def __init__(self):
        self.suggestion_patterns = {
            "code": {
                "keywords": ["code", "programming", "function", "class", "algorithm"],
                "suggestions": [
                    "🔍 Analyze code",
                    "🛠️ Suggest improvements",
                    "📚 Explain concepts",
                    "🧪 Generate tests",
                ],
            },
            "help": {
                "keywords": ["help", "how", "what", "explain", "unclear"],
                "suggestions": [
                    "💡 Get detailed explanation",
                    "📖 Show examples",
                    "🎯 Break down steps",
                    "🤝 Get expert assistance",
                ],
            },
            "creative": {
                "keywords": ["create", "design", "generate", "brainstorm"],
                "suggestions": [
                    "🎨 Generate variations",
                    "✨ Enhance creativity",
                    "🔄 Iterate design",
                    "🎭 Try different styles",
                ],
            },
            "analysis": {
                "keywords": ["analyze", "review", "check", "examine"],
                "suggestions": [
                    "📊 Deep analysis",
                    "🔬 Detailed review",
                    "📈 Show insights",
                    "🎯 Action items",
                ],
            },
        }

    def get_smart_suggestions(
        self, user_message: str, context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate smart suggestions based on user message and context"""
        message_lower = user_message.lower()
        suggestions = []
        for category, pattern_data in self.suggestion_patterns.items():
            if any(keyword in message_lower for keyword in pattern_data["keywords"]):
                for suggestion in pattern_data["suggestions"]:
                    suggestions.append(
                        {
                            "text": suggestion,
                            "callback_data": f"suggest_{category}_{len(suggestions)}",
                            "category": category,
                        }
                    )
                break
        if context.get("has_media"):
            suggestions.extend(
                [
                    {
                        "text": "🖼️ Analyze media",
                        "callback_data": "suggest_media_analyze",
                    },
                    {
                        "text": "💾 Save to memory",
                        "callback_data": "suggest_media_save",
                    },
                ]
            )
        if context.get("is_group"):
            suggestions.extend(
                [
                    {
                        "text": "👥 Share with group",
                        "callback_data": "suggest_group_share",
                    },
                    {"text": "📝 Add to notes", "callback_data": "suggest_group_note"},
                ]
            )
        if context.get("has_conversation_history"):
            suggestions.extend(
                [
                    {
                        "text": "📚 Review history",
                        "callback_data": "suggest_history_review",
                    },
                    {
                        "text": "📄 Generate summary",
                        "callback_data": "suggest_summary_create",
                    },
                ]
            )
        return suggestions[:6]


class InteractiveMenus:
    """Rich interactive menu system"""

    def __init__(self):
        self.menu_cache = {}
        self.logger = logging.getLogger(__name__)

    def create_quick_actions_menu(
        self, user_id: int, context: Dict[str, Any]
    ) -> InlineKeyboardMarkup:
        """Create quick actions menu based on user context"""
        buttons = []
        row1 = [
            InlineKeyboardButton("🤖 Switch Model", callback_data="quick_switch_model"),
            InlineKeyboardButton("⚙️ Settings", callback_data="quick_settings"),
        ]
        buttons.append(row1)
        row2 = []
        if context.get("can_generate_image"):
            row2.append(
                InlineKeyboardButton(
                    "🎨 Generate Image", callback_data="quick_gen_image"
                )
            )
        if context.get("has_conversation"):
            row2.append(
                InlineKeyboardButton("📊 Get Summary", callback_data="quick_summary")
            )
        if context.get("is_group"):
            row2.append(
                InlineKeyboardButton(
                    "👥 Group Tools", callback_data="quick_group_tools"
                )
            )
        if row2:
            buttons.append(row2)
        row3 = []
        if context.get("recent_errors"):
            row3.append(
                InlineKeyboardButton("🔧 Debug Help", callback_data="quick_debug")
            )
        if context.get("has_media"):
            row3.append(
                InlineKeyboardButton(
                    "🔍 Analyze Media", callback_data="quick_analyze_media"
                )
            )
        if row3:
            buttons.append(row3)
        return InlineKeyboardMarkup(buttons)

    def create_model_selection_carousel(
        self, models: List[Dict[str, Any]], current_page: int = 0, per_page: int = 4
    ) -> InlineKeyboardMarkup:
        """Create a carousel-style model selection menu"""
        buttons = []
        start_idx = current_page * per_page
        end_idx = min(start_idx + per_page, len(models))
        page_models = models[start_idx:end_idx]
        for model in page_models:
            button_text = f"{model.get('emoji', '🤖')} {model['display_name']}"
            callback_data = f"select_model_{model['model_id']}"
            buttons.append(
                [InlineKeyboardButton(button_text, callback_data=callback_data)]
            )
        nav_buttons = []
        if current_page > 0:
            nav_buttons.append(
                InlineKeyboardButton(
                    "⬅️ Previous", callback_data=f"model_page_{current_page - 1}"
                )
            )
        total_pages = (len(models) - 1) // per_page + 1
        nav_buttons.append(
            InlineKeyboardButton(
                f"📄 {current_page + 1}/{total_pages}", callback_data="page_info"
            )
        )
        if end_idx < len(models):
            nav_buttons.append(
                InlineKeyboardButton(
                    "➡️ Next", callback_data=f"model_page_{current_page + 1}"
                )
            )
        if nav_buttons:
            buttons.append(nav_buttons)
        buttons.append(
            [
                InlineKeyboardButton("🔙 Back to Menu", callback_data="back_to_main"),
                InlineKeyboardButton("ℹ️ Model Info", callback_data="model_info"),
            ]
        )
        return InlineKeyboardMarkup(buttons)

    def create_group_collaboration_menu(
        self, group_id: str, user_permissions: Dict[str, bool]
    ) -> InlineKeyboardMarkup:
        """Create collaboration menu for group chats"""
        buttons = []
        row1 = [
            InlineKeyboardButton(
                "📝 Add Note", callback_data=f"group_add_note_{group_id}"
            ),
            InlineKeyboardButton(
                "📊 Group Summary", callback_data=f"group_summary_{group_id}"
            ),
        ]
        buttons.append(row1)
        row2 = [
            InlineKeyboardButton(
                "🧠 Shared Memory", callback_data=f"group_memory_{group_id}"
            ),
            InlineKeyboardButton(
                "🔍 Search History", callback_data=f"group_search_{group_id}"
            ),
        ]
        buttons.append(row2)
        if user_permissions.get("can_manage"):
            row3 = [
                InlineKeyboardButton(
                    "⚙️ Group Settings", callback_data=f"group_settings_{group_id}"
                ),
                InlineKeyboardButton(
                    "📤 Export Chat", callback_data=f"group_export_{group_id}"
                ),
            ]
            buttons.append(row3)
        row4 = [
            InlineKeyboardButton(
                "🏃‍♂️ Quick Actions", callback_data=f"group_quick_{group_id}"
            ),
            InlineKeyboardButton("🔙 Back", callback_data="back_to_chat"),
        ]
        buttons.append(row4)
        return InlineKeyboardMarkup(buttons)

    def create_context_aware_suggestions(
        self, suggestions: List[Dict[str, str]]
    ) -> InlineKeyboardMarkup:
        """Create suggestion buttons based on smart analysis"""
        if not suggestions:
            return None
        buttons = []
        for i in range(0, len(suggestions), 2):
            row = []
            for j in range(2):
                if i + j < len(suggestions):
                    suggestion = suggestions[i + j]
                    row.append(
                        InlineKeyboardButton(
                            suggestion["text"],
                            callback_data=suggestion["callback_data"],
                        )
                    )
            if row:
                buttons.append(row)
        buttons.append(
            [InlineKeyboardButton("❌ Dismiss", callback_data="dismiss_suggestions")]
        )
        return InlineKeyboardMarkup(buttons)


class RichUIManager:
    """Main manager for rich UI components"""

    def __init__(self):
        self.progress = ProgressIndicator()
        self.suggestions = SmartSuggestions()
        self.menus = InteractiveMenus()
        self.logger = logging.getLogger(__name__)

    async def enhance_message_with_ui(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user_message: str,
        ai_response: str,
        message_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enhance a message exchange with rich UI elements"""
        user_id = update.effective_user.id
        smart_suggestions = self.suggestions.get_smart_suggestions(
            user_message, message_context
        )
        suggestion_keyboard = None
        if smart_suggestions:
            suggestion_keyboard = self.menus.create_context_aware_suggestions(
                smart_suggestions
            )
        quick_actions = self.menus.create_quick_actions_menu(user_id, message_context)
        enhanced_response = {
            "text": ai_response,
            "suggestion_keyboard": suggestion_keyboard,
            "quick_actions": quick_actions,
            "suggestions": smart_suggestions,
            "context": message_context,
        }
        return enhanced_response

    async def handle_long_operation(
        self,
        operation_name: str,
        operation_func,
        progress_message,
        steps: List[str],
        *args,
        **kwargs,
    ) -> Any:
        """Handle long-running operations with progress indication"""
        operation_id = f"{operation_name}_{int(time.time())}"
        try:
            await self.progress.start_progress(
                operation_id,
                progress_message,
                len(steps),
                f"Starting {operation_name}...",
            )
            for i, step_description in enumerate(steps):
                await self.progress.update_progress(
                    operation_id, i + 1, step_description
                )
                await asyncio.sleep(0.5)
            result = await operation_func(*args, **kwargs)
            await self.progress.complete_progress(
                operation_id, f"{operation_name} completed successfully!", success=True
            )
            return result
        except Exception as e:
            await self.progress.complete_progress(
                operation_id, f"{operation_name} failed: {str(e)}", success=False
            )
            raise

    async def create_collaborative_interface(
        self, group_id: str, user_id: int, group_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create collaborative interface for group interactions"""
        user_permissions = group_context.get("user_permissions", {})
        collab_menu = self.menus.create_group_collaboration_menu(
            group_id, user_permissions
        )
        participants = group_context.get("participants", [])
        recent_activity = group_context.get("recent_activity", [])
        status_text = "👥 **Group Collaboration Panel**\n\n"
        status_text += f"**Participants:** {len(participants)}\n"
        status_text += f"**Recent Activity:** {len(recent_activity)} messages\n"
        if group_context.get("active_topic"):
            status_text += f"**Current Topic:** {group_context['active_topic']}\n"
        interface = {
            "status_text": status_text,
            "collaboration_menu": collab_menu,
            "group_context": group_context,
            "suggestions": self.suggestions.get_smart_suggestions(
                "group collaboration", group_context
            ),
        }
        return interface


rich_ui = RichUIManager()
