"""
MCP Commands for Telegram Bot
This module provides MCP-related commands for the Telegram bot.
"""

import logging
from telegram import Update
from telegram.ext import ContextTypes
from src.services.mcp_bot_integration import mcp_integration


class MCPCommands:
    """MCP-related commands for the Telegram bot."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def mcp_status_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show MCP integration status."""
        message = update.message
        try:
            status = await mcp_integration.get_mcp_status()
            if not status["initialized"]:
                await message.reply_text(
                    "❌ MCP integration is not initialized.\n"
                    "Please check the bot configuration and try again."
                )
                return
            status_msg = "🔧 **MCP Integration Status**\n\n"
            if status["openrouter_available"]:
                status_msg += "✅ OpenRouter with MCP: Available\n"
            else:
                status_msg += "❌ OpenRouter with MCP: Not available\n"
                status_msg += "   (OPENROUTER_API_KEY not set)\n"
            status_msg += f"🔗 Connected Servers: {len(status['servers'])}\n"
            status_msg += f"🛠️ Available Tools: {status['tools_count']}\n\n"
            if status["servers"]:
                status_msg += "**Connected Servers:**\n"
                for server_name, info in status["servers"].items():
                    status_msg += f"• {server_name}: {info['tool_count']} tools\n"
                    tools_preview = info["tools"][:3]
                    if tools_preview:
                        status_msg += f"  └ {', '.join(tools_preview)}\n"
                        if len(info["tools"]) > 3:
                            status_msg += f"  └ ... and {len(info['tools']) - 3} more\n"
            else:
                status_msg += "No servers connected."
            await message.reply_text(status_msg, parse_mode="Markdown")
        except Exception as e:
            self.logger.error(f"Error in mcp_status command: {str(e)}")
            await message.reply_text(
                "❌ Error retrieving MCP status. Please try again later."
            )

    async def mcp_toggle_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Toggle MCP functionality for the user."""
        user_id = update.effective_user.id
        message = update.message
        try:
            currently_enabled = await mcp_integration.is_mcp_available_for_user(user_id)
            new_status = not currently_enabled
            await mcp_integration.set_mcp_enabled_for_user(user_id, new_status)
            status_emoji = "✅" if new_status else "❌"
            status_text = "enabled" if new_status else "disabled"
            await message.reply_text(
                f"{status_emoji} MCP functionality has been **{status_text}** for you.\n\n"
                f"When enabled, the bot will use MCP tools to enhance responses with real-time data from connected servers."
            )
        except Exception as e:
            self.logger.error(f"Error in mcp_toggle command: {str(e)}")
            await message.reply_text(
                "❌ Error toggling MCP functionality. Please try again later."
            )

    async def mcp_tools_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show available MCP tools."""
        user_id = update.effective_user.id
        message = update.message
        try:
            if not await mcp_integration.is_mcp_available_for_user(user_id):
                await message.reply_text(
                    "❌ MCP is not available for you.\n"
                    "Use /mcptoggle to enable MCP functionality."
                )
                return
            tools = await mcp_integration.get_available_tools()
            if not tools:
                await message.reply_text(
                    "📭 No MCP tools are currently available.\n"
                    "This might be due to server connection issues."
                )
                return
            tools_msg = f"🛠️ **Available MCP Tools** ({len(tools)})\n\n"
            for i, tool in enumerate(tools[:10], 1):
                func = tool.get("function", {})
                name = func.get("name", "Unknown")
                description = func.get("description", "No description")
                tools_msg += f"**{i}. {name}**\n"
                tools_msg += f"└ {description[:100]}{'...' if len(description) > 100 else ''}\n\n"
            if len(tools) > 10:
                tools_msg += f"... and {len(tools) - 10} more tools"
            if len(tools_msg) > 4000:
                chunks = [
                    tools_msg[i : i + 4000] for i in range(0, len(tools_msg), 4000)
                ]
                for chunk in chunks:
                    await message.reply_text(chunk, parse_mode="Markdown")
            else:
                await message.reply_text(tools_msg, parse_mode="Markdown")
        except Exception as e:
            self.logger.error(f"Error in mcp_tools command: {str(e)}")
            await message.reply_text(
                "❌ Error retrieving MCP tools. Please try again later."
            )

    async def mcp_help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show MCP help information."""
        message = update.message
        help_msg = """
🤖 **MCP (Model Context Protocol) Integration**
MCP allows the bot to use external tools and services to enhance responses with real-time data.
**Available Commands:**
• `/mcpstatus` - Show MCP integration status
• `/mcptoggle` - Enable/disable MCP for your account
• `/mcptools` - List available MCP tools
• `/mcphelp` - Show this help message
**How it works:**
1. **Automatic Tool Discovery**: The bot connects to MCP servers and discovers available tools
2. **Smart Tool Selection**: When you ask a question, the bot automatically selects relevant tools
3. **Real-time Data**: Tools provide up-to-date information from various sources
4. **Enhanced Responses**: Your responses include data from multiple sources
**Example Usage:**
• "What's the weather in Tokyo?" → Uses weather API tools
• "Show me the latest React documentation" → Uses Context7 documentation tools
• "Search for Python tutorials" → Uses search tools
**Supported Servers:**
• **Context7**: Documentation and code examples
• **Exa Search**: Web search capabilities
• **Fetch MCP**: Web content fetching
• **Sequential Thinking**: Reasoning tools
**Note**: MCP functionality requires proper server configuration and API keys.
        """
        await message.reply_text(help_msg, parse_mode="Markdown")


mcp_commands = MCPCommands()
