"""
🎨 Enhanced UI Components for Telegram Bot
Provides modern, interactive UI elements with rich functionality
"""

import logging
from typing import List, Dict, Any, Optional
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedUIComponents:
    """Modern UI components with rich interactivity and visual appeal"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Emoji sets for different purposes
        self.emojis = {
            "status": {"active": "🟢", "inactive": "🔴", "pending": "🟡"},
            "actions": {"create": "➕", "edit": "✏️", "delete": "🗑️", "view": "👁️"},
            "navigation": {"back": "⬅️", "forward": "➡️", "home": "🏠", "up": "⬆️"},
            "features": {"ai": "🤖", "memory": "🧠", "group": "👥", "settings": "⚙️"},
            "media": {"image": "🖼️", "video": "🎥", "audio": "🎵", "document": "📄"},
            "collaboration": {
                "share": "🔗",
                "assign": "📌",
                "discuss": "💬",
                "vote": "🗳️",
            },
        }

    def create_quick_action_menu(
        self, user_context: Dict[str, Any]
    ) -> InlineKeyboardMarkup:
        """Create a dynamic quick action menu based on user context"""
        buttons = []

        # Row 1: Core AI Functions
        row1 = [
            InlineKeyboardButton(
                f"{self.emojis['features']['ai']} Chat", callback_data="quick_chat"
            ),
            InlineKeyboardButton(
                f"{self.emojis['media']['image']} Generate Image",
                callback_data="quick_image",
            ),
            InlineKeyboardButton(
                f"{self.emojis['features']['memory']} Smart Search",
                callback_data="quick_search",
            ),
        ]
        buttons.append(row1)

        # Row 2: Collaboration (if in group)
        if user_context.get("is_group", False):
            row2 = [
                InlineKeyboardButton(
                    f"{self.emojis['collaboration']['share']} Share Context",
                    callback_data="quick_share_context",
                ),
                InlineKeyboardButton(
                    f"{self.emojis['collaboration']['assign']} Create Note",
                    callback_data="quick_collab_note",
                ),
                InlineKeyboardButton(
                    f"{self.emojis['collaboration']['discuss']} Group Summary",
                    callback_data="quick_group_summary",
                ),
            ]
            buttons.append(row2)

        # Row 3: Recent Actions (personalized)
        recent_actions = user_context.get("recent_actions", [])
        if recent_actions:
            row3 = []
            for action in recent_actions[:3]:  # Max 3 recent actions
                emoji = self.emojis["actions"].get(action["type"], "🔄")
                row3.append(
                    InlineKeyboardButton(
                        f"{emoji} {action['label']}",
                        callback_data=f"quick_recent_{action['id']}",
                    )
                )
            buttons.append(row3)

        # Row 4: Settings and Help
        row4 = [
            InlineKeyboardButton(
                f"{self.emojis['features']['settings']} Settings",
                callback_data="quick_settings",
            ),
            InlineKeyboardButton(
                f"{self.emojis['navigation']['home']} Main Menu",
                callback_data="quick_main_menu",
            ),
        ]
        buttons.append(row4)

        return InlineKeyboardMarkup(buttons)

    def create_smart_model_selector(
        self, available_models: Dict[str, Any], current_model: str
    ) -> InlineKeyboardMarkup:
        """Create an intelligent model selector with recommendations"""
        buttons = []

        # Group models by capability and show recommendations
        model_categories = {
            "💬 Chat Experts": ["gemini-2.0-flash", "deepseek-r1"],
            "🎨 Creative AI": ["claude-3.5-sonnet", "gpt-4"],
            "⚡ Speed Champions": ["gemini-1.5-flash", "claude-3-haiku"],
            "🧠 Deep Thinkers": ["deepseek-v3", "qwen2.5-72b"],
        }

        for category, models in model_categories.items():
            category_buttons = []
            for model in models:
                if model in available_models:
                    model_info = available_models[model]

                    # Add indicators for current model and performance
                    indicators = ""
                    if model == current_model:
                        indicators += "✅ "

                    # Performance indicators
                    speed = model_info.get("speed_score", 0)
                    if speed > 8:
                        indicators += "⚡"

                    quality = model_info.get("quality_score", 0)
                    if quality > 8:
                        indicators += "🌟"

                    button_text = f"{indicators}{model_info.get('display_name', model)}"
                    category_buttons.append(
                        InlineKeyboardButton(
                            button_text, callback_data=f"select_model_{model}"
                        )
                    )

            if category_buttons:
                # Add category header
                buttons.append(
                    [
                        InlineKeyboardButton(
                            category, callback_data="model_category_info"
                        )
                    ]
                )

                # Add model buttons in rows of 2
                for i in range(0, len(category_buttons), 2):
                    row = category_buttons[i : i + 2]
                    buttons.append(row)

        # Add comparison and help options
        buttons.append(
            [
                InlineKeyboardButton(
                    "📊 Compare Models", callback_data="compare_models"
                ),
                InlineKeyboardButton(
                    "🎯 Auto-Select Best", callback_data="auto_select_model"
                ),
            ]
        )

        return InlineKeyboardMarkup(buttons)

    def create_conversation_insights_panel(self, insights: Dict[str, Any]) -> str:
        """Create a rich conversation insights panel"""
        panel = "🧠 **Conversation Intelligence Panel**\n\n"

        # Memory Statistics
        memory_stats = insights.get("memory_stats", {})
        panel += "📊 **Memory Analytics:**\n"
        panel += f"• Total messages: {memory_stats.get('total_messages', 0)}\n"
        panel += (
            f"• Important conversations: {memory_stats.get('important_count', 0)}\n"
        )
        panel += f"• Last activity: {memory_stats.get('last_activity', 'Unknown')}\n\n"

        # Active Topics
        topics = insights.get("active_topics", [])
        if topics:
            panel += "🎯 **Active Topics:**\n"
            for i, topic in enumerate(topics[:5], 1):
                panel += f"{i}. {topic}\n"
            panel += "\n"

        # Smart Suggestions
        suggestions = insights.get("smart_suggestions", [])
        if suggestions:
            panel += "💡 **Smart Suggestions:**\n"
            for suggestion in suggestions[:3]:
                panel += f"• {suggestion}\n"
            panel += "\n"

        # Group Insights (if applicable)
        if insights.get("is_group"):
            group_info = insights.get("group_info", {})
            panel += "👥 **Group Insights:**\n"
            panel += f"• Active members: {group_info.get('active_members', 0)}\n"
            panel += f"• Collaboration score: {group_info.get('collaboration_score', 'N/A')}\n"
            panel += (
                f"• Shared knowledge items: {group_info.get('shared_knowledge', 0)}\n"
            )

        return panel

    def create_progress_indicator(
        self, task_name: str, progress: float, eta: Optional[str] = None
    ) -> str:
        """Create an animated progress indicator"""
        # Progress bar with 10 segments
        filled_segments = int(progress * 10)
        progress_bar = "█" * filled_segments + "░" * (10 - filled_segments)

        # Add percentage and ETA
        percentage = int(progress * 100)
        eta_text = f" (ETA: {eta})" if eta else ""

        return f"🔄 **{task_name}**\n`[{progress_bar}]` {percentage}%{eta_text}"

    def create_context_aware_suggestions(
        self, context: Dict[str, Any]
    ) -> List[InlineKeyboardButton]:
        """Generate context-aware action suggestions"""
        suggestions = []

        # Analyze context and generate relevant suggestions
        last_message = context.get("last_message", "")

        # Code-related suggestions
        if any(
            keyword in last_message.lower()
            for keyword in ["code", "function", "python", "javascript"]
        ):
            suggestions.extend(
                [
                    InlineKeyboardButton(
                        "🔍 Explain Code", callback_data="suggest_explain_code"
                    ),
                    InlineKeyboardButton(
                        "🐛 Debug Help", callback_data="suggest_debug"
                    ),
                    InlineKeyboardButton(
                        "📚 Best Practices", callback_data="suggest_best_practices"
                    ),
                ]
            )

        # Creative suggestions
        elif any(
            keyword in last_message.lower()
            for keyword in ["image", "creative", "design", "art"]
        ):
            suggestions.extend(
                [
                    InlineKeyboardButton(
                        "🎨 Generate Variations", callback_data="suggest_variations"
                    ),
                    InlineKeyboardButton(
                        "🖼️ Style Transfer", callback_data="suggest_style_transfer"
                    ),
                    InlineKeyboardButton(
                        "📏 Resize/Edit", callback_data="suggest_edit_image"
                    ),
                ]
            )

        # Learning/Educational suggestions
        elif any(
            keyword in last_message.lower()
            for keyword in ["learn", "explain", "how", "what", "why"]
        ):
            suggestions.extend(
                [
                    InlineKeyboardButton(
                        "📖 More Details", callback_data="suggest_details"
                    ),
                    InlineKeyboardButton(
                        "🎯 Examples", callback_data="suggest_examples"
                    ),
                    InlineKeyboardButton(
                        "🔗 Related Topics", callback_data="suggest_related"
                    ),
                ]
            )

        # Default suggestions if no specific context
        if not suggestions:
            suggestions.extend(
                [
                    InlineKeyboardButton(
                        "🔄 Rephrase", callback_data="suggest_rephrase"
                    ),
                    InlineKeyboardButton("📊 Analyze", callback_data="suggest_analyze"),
                    InlineKeyboardButton(
                        "💾 Save Important", callback_data="suggest_save"
                    ),
                ]
            )

        return suggestions

    def create_smart_pagination(
        self, items: List[Any], current_page: int, items_per_page: int = 5
    ) -> Dict[str, Any]:
        """Create smart pagination with preview"""
        total_pages = max(1, (len(items) + items_per_page - 1) // items_per_page)
        start_idx = current_page * items_per_page
        end_idx = min(start_idx + items_per_page, len(items))

        # Create navigation buttons
        nav_buttons = []

        # Previous page
        if current_page > 0:
            nav_buttons.append(
                InlineKeyboardButton(
                    f"{self.emojis['navigation']['back']} Previous",
                    callback_data=f"page_{current_page - 1}",
                )
            )

        # Page indicator
        nav_buttons.append(
            InlineKeyboardButton(
                f"📄 {current_page + 1}/{total_pages}", callback_data="page_info"
            )
        )

        # Next page
        if current_page < total_pages - 1:
            nav_buttons.append(
                InlineKeyboardButton(
                    f"Next {self.emojis['navigation']['forward']}",
                    callback_data=f"page_{current_page + 1}",
                )
            )

        return {
            "items": items[start_idx:end_idx],
            "navigation": nav_buttons,
            "page_info": {
                "current": current_page + 1,
                "total": total_pages,
                "showing": f"{start_idx + 1}-{end_idx} of {len(items)}",
            },
        }

    def create_adaptive_keyboard(
        self, user_preferences: Dict[str, Any], context: Dict[str, Any]
    ) -> InlineKeyboardMarkup:
        """Create an adaptive keyboard based on user preferences and context"""
        buttons = []

        # Adapt based on user's most used features
        frequent_features = user_preferences.get("frequent_features", [])

        # First row: Most used features
        if frequent_features:
            row1 = []
            for feature in frequent_features[:3]:
                emoji = self.emojis["features"].get(feature, "🔧")
                row1.append(
                    InlineKeyboardButton(
                        f"{emoji} {feature.title()}",
                        callback_data=f"adaptive_{feature}",
                    )
                )
            buttons.append(row1)

        # Context-sensitive suggestions
        context_suggestions = self.create_context_aware_suggestions(context)
        if context_suggestions:
            # Add suggestions in rows of 2
            for i in range(0, len(context_suggestions), 2):
                buttons.append(context_suggestions[i : i + 2])

        # Add discovery section for new features
        if user_preferences.get("show_discovery", True):
            discovery_buttons = [
                InlineKeyboardButton(
                    "🔮 Discover New", callback_data="adaptive_discover"
                ),
                InlineKeyboardButton("📈 Usage Stats", callback_data="adaptive_stats"),
            ]
            buttons.append(discovery_buttons)

        return InlineKeyboardMarkup(buttons)

    def format_rich_message(self, content: str, message_type: str = "info") -> str:
        """Format messages with rich styling and structure"""

        # Message type indicators
        type_indicators = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "tip": "💡",
            "feature": "🚀",
        }

        indicator = type_indicators.get(message_type, "📝")

        # Add rich formatting
        formatted = f"{indicator} **{message_type.title()}**\n\n{content}"

        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        formatted += f"\n\n🕐 *{timestamp}*"

        return formatted

    async def handle_ui_interaction(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, interaction_data: str
    ) -> None:
        """Handle UI component interactions"""
        try:
            # Parse interaction data
            parts = interaction_data.split("_", 2)
            component_type = parts[0]
            action = parts[1] if len(parts) > 1 else ""
            data = parts[2] if len(parts) > 2 else ""

            if component_type == "quick":
                await self._handle_quick_action(update, context, action, data)
            elif component_type == "adaptive":
                await self._handle_adaptive_action(update, context, action, data)
            elif component_type == "suggest":
                await self._handle_suggestion_action(update, context, action, data)
            else:
                self.logger.warning(f"Unknown UI component type: {component_type}")

        except Exception as e:
            self.logger.error(f"Error handling UI interaction: {e}")
            if update.callback_query:
                await update.callback_query.answer(
                    "❌ Action failed. Please try again."
                )

    async def _handle_quick_action(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, data: str
    ) -> None:
        """Handle quick action interactions"""
        if action == "chat":
            await update.callback_query.message.reply_text(
                "🤖 Quick Chat activated! What would you like to discuss?"
            )
        elif action == "image":
            await update.callback_query.message.reply_text(
                "🎨 Image Generator ready! Describe the image you want to create."
            )
        elif action == "search":
            await update.callback_query.message.reply_text(
                "🔍 Smart Search activated! What would you like to find in our conversation history?"
            )
        # Add more quick actions as needed

    async def _handle_adaptive_action(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, data: str
    ) -> None:
        """Handle adaptive keyboard interactions"""
        if action == "discover":
            await update.callback_query.message.reply_text(
                "🔮 **Discover New Features**\n\nHere are some features you might not know about:\n"
                "• 🧠 Semantic memory search\n"
                "• 👥 Group collaboration tools\n"
                "• 📊 Conversation analytics\n"
                "• 🎯 Smart model selection"
            )
        elif action == "stats":
            await update.callback_query.message.reply_text(
                "📈 **Your Usage Statistics**\n\n"
                "This feature is being prepared for you!\n"
                "Soon you'll see detailed analytics about your bot usage."
            )

    async def _handle_suggestion_action(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, data: str
    ) -> None:
        """Handle context-aware suggestion interactions"""
        suggestion_responses = {
            "explain_code": "🔍 **Code Explanation Service**\n\nI'll analyze and explain any code you share!",
            "debug": "🐛 **Debug Assistant**\n\nShare your code and error message, I'll help you fix it!",
            "examples": "🎯 **Example Generator**\n\nI'll provide practical examples for better understanding!",
            "analyze": "📊 **Content Analyzer**\n\nI'll provide detailed analysis of your content!",
        }

        response = suggestion_responses.get(
            action,
            f"🔧 **{action.title()}**\n\nThis feature is being activated for you!",
        )
        await update.callback_query.message.reply_text(response)
