"""
MCP Command Handlers for Telegram Bot
Provides user-facing commands that interact with MCP tools via Pydantic AI agents.
"""

import sys
import os
import logging
from typing import Optional

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from src.services.mcp import get_mcp_registry
from src.utils.log.telegramlog import TelegramLogger


class MCPCommands:
    """Handles MCP-related commands for the Telegram bot."""
    
    def __init__(self, user_data_manager, telegram_logger: TelegramLogger, openrouter_api=None):
        """
        Initialize MCP Commands handler.
        
        Args:
            user_data_manager: User data management service
            telegram_logger: Telegram logging service
            openrouter_api: OpenRouter API with MCP integration (optional, will be set later)
        """
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.openrouter_api = openrouter_api
        self.logger = logging.getLogger(__name__)
        
    def set_openrouter_api(self, openrouter_api):
        """Set the OpenRouter API instance after initialization."""
        self.openrouter_api = openrouter_api
        
    async def _ensure_mcp_integration(self) -> bool:
        """Ensure MCP integration is available."""
        if not self.openrouter_api:
            self.logger.error("OpenRouter API not available for MCP commands")
            return False
            
        # Initialize MCP if not already done
        if not self.openrouter_api._mcp_initialized:
            await self.openrouter_api.initialize_mcp()
        
        # Check if MCP registry is available after initialization
        if not hasattr(self.openrouter_api, 'mcp_registry') or not self.openrouter_api.mcp_registry:
            self.logger.error("MCP registry not available in OpenRouter API")
            return False
            
        return True
        
    async def _execute_mcp_tool(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                               tool_name: str, query: str, server_name: Optional[str] = None) -> None:
        """
        Execute an MCP tool via the Pydantic AI agent.
        
        Args:
            update: Telegram update object
            context: Telegram context
            tool_name: Name of the MCP tool to execute
            query: User query to process
            server_name: Optional server name for logging
        """
        user_id = update.effective_user.id
        
        try:
            # Ensure MCP integration is available
            if not await self._ensure_mcp_integration():
                await update.message.reply_text(
                    "❌ MCP integration is not available. Please try again later."
                )
                return
                
            # Show typing indicator
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            
            # Log the request
            self.telegram_logger.log_message(f"MCP {tool_name} request: {query[:100]}...", user_id)
            
            # Get user's preferred model
            from services.user_preferences_manager import UserPreferencesManager
            preferences_manager = UserPreferencesManager(self.user_data_manager)
            user_id = update.effective_user.id
            preferred_model = await preferences_manager.get_user_model_preference(user_id)
            
            # Create a specialized prompt for the MCP tool
            mcp_prompt = f"""Please use the {tool_name} tool to help with this request: {query}

Use the appropriate MCP tool to provide a comprehensive and helpful response. Format the response clearly for the user."""
            
            # Call the enhanced OpenRouter API with MCP tools (using user's preferred model)
            response = await self.openrouter_api.generate_response(
                prompt=mcp_prompt,
                model=preferred_model,  # Use user's preferred model
                temperature=0.7,
                max_tokens=2000
            )
            
            if response and response.strip():
                # Format the response for Telegram
                formatted_response = self._format_mcp_response(response, tool_name, server_name)
                
                # Send response
                await update.message.reply_text(
                    formatted_response,
                    parse_mode='Markdown',
                    disable_web_page_preview=True
                )
                
                # Log successful execution
                self.telegram_logger.log_message(f"MCP {tool_name} response sent", user_id)
                
            else:
                await update.message.reply_text(
                    f"❌ No response received from {tool_name}. Please try again."
                )
                
        except Exception as e:
            self.logger.error(f"Error executing MCP tool {tool_name}: {e}")
            await update.message.reply_text(
                f"❌ Error executing {tool_name}: {str(e)[:200]}..."
            )
            
    def _format_mcp_response(self, response: str, tool_name: str, server_name: Optional[str] = None) -> str:
        """Format MCP tool response for Telegram."""
        # Add header with tool information
        header = f"🔧 **{tool_name}**"
        if server_name:
            header += f" *({server_name})*"
        header += "\n" + "─" * 40 + "\n\n"
        
        # Clean up response
        cleaned_response = response.strip()
        
        # Ensure response isn't too long for Telegram
        if len(cleaned_response) > 3500:
            cleaned_response = cleaned_response[:3500] + "...\n\n*Response truncated*"
            
        return header + cleaned_response
        
    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /search command - Web search via Exa Search MCP tool."""
        user_id = update.effective_user.id
        
        if not context.args:
            await update.message.reply_text(
                "📝 **Usage:** `/search <query>`\n\n"
                "**Example:** `/search latest developments in AI`\n\n"
                "Performs web search using Exa AI search engine.",
                parse_mode='Markdown'
            )
            return
            
        query = " ".join(context.args)
        self.telegram_logger.log_message(f"Search command: {query}", user_id)
        
        await self._execute_mcp_tool(update, context, "exa_search_web_search_exa", query, "Exa Search")
        
    async def context7_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /Context7 command - Documentation search via Context7 MCP tool."""
        user_id = update.effective_user.id
        
        if not context.args:
            await update.message.reply_text(
                "📚 **Usage:** `/Context7 <query>`\n\n"
                "**Example:** `/Context7 FastAPI middleware`\n\n"
                "Searches library documentation and code examples.",
                parse_mode='Markdown'
            )
            return
            
        query = " ".join(context.args)
        self.telegram_logger.log_message(f"Context7 command: {query}", user_id)
        
        await self._execute_mcp_tool(update, context, "resolve-library-id", query, "Context7")
        
    async def sequentialthinking_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /sequentialthinking command - Problem-solving via Sequential Thinking MCP tool."""
        user_id = update.effective_user.id
        
        if not context.args:
            await update.message.reply_text(
                "🧠 **Usage:** `/sequentialthinking <problem>`\n\n"
                "**Example:** `/sequentialthinking How to optimize database queries`\n\n"
                "Uses sequential thinking to break down and solve complex problems.",
                parse_mode='Markdown'
            )
            return
            
        query = " ".join(context.args)
        self.telegram_logger.log_message(f"Sequential thinking command: {query}", user_id)
        
        await self._execute_mcp_tool(update, context, "sequentialthinking", query, "Sequential Thinking")
        
    async def docfork_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /Docfork command - Document analysis via Docfork MCP tool."""
        user_id = update.effective_user.id
        
        if not context.args:
            await update.message.reply_text(
                "📄 **Usage:** `/Docfork <document_url_or_query>`\n\n"
                "**Example:** `/Docfork https://example.com/document.pdf`\n\n"
                "Analyzes documents and provides insights.",
                parse_mode='Markdown'
            )
            return
            
        query = " ".join(context.args)
        self.telegram_logger.log_message(f"Docfork command: {query}", user_id)
        
        await self._execute_mcp_tool(update, context, "get-library-docs", query, "Docfork")
        
    async def mcp_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mcp_status command - Show MCP server status."""
        user_id = update.effective_user.id
        self.telegram_logger.log_message("MCP status command", user_id)
        
        try:
            # Get MCP registry
            registry = await get_mcp_registry()
            
            # Get server information
            servers_info = registry.get_available_tools_info()
            
            if not servers_info:
                await update.message.reply_text("❌ No MCP servers configured.")
                return
                
            # Format status message
            status_message = "🔧 **MCP Server Status**\n" + "=" * 30 + "\n\n"
            
            for server_name, info in servers_info.items():
                status_icon = "✅" if info["status"] == "initialized" else "❌"
                status_message += f"{status_icon} **{server_name}**\n"
                status_message += f"   Status: `{info['status']}`\n"
                
                if info["status"] == "initialized":
                    status_message += f"   Command: `{info['command']}`\n"
                    if info.get("tool_prefix"):
                        status_message += f"   Prefix: `{info['tool_prefix']}`\n"
                        
                status_message += "\n"
                
            # Add integration status
            if await self._ensure_mcp_integration():
                status_message += "✅ **Integration Status:** Ready\n"
            else:
                status_message += "❌ **Integration Status:** Not Available\n"
                
            await update.message.reply_text(status_message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error getting MCP status: {e}")
            await update.message.reply_text(f"❌ Error getting MCP status: {str(e)}")
            
    async def mcp_help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mcp_help command - Show MCP commands help."""
        help_text = """🔧 **MCP Commands Help**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Available Commands:**

🔍 `/search <query>`
   Web search using Exa AI search engine
   *Example:* `/search latest AI developments`

📚 `/Context7 <query>`  
   Search library documentation and examples
   *Example:* `/Context7 FastAPI middleware`

🧠 `/sequentialthinking <problem>`
   Break down complex problems step-by-step
   *Example:* `/sequentialthinking optimize database`

📄 `/Docfork <document_url_or_query>`
   Analyze documents and provide insights
   *Example:* `/Docfork https://example.com/doc.pdf`

⚙️ `/mcp_status`
   Show status of all MCP servers

❓ `/mcp_help`
   Show this help message

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**About MCP:**
Model Context Protocol provides AI agents with access to external tools and data sources, enabling more powerful and context-aware responses."""

        await update.message.reply_text(help_text, parse_mode='Markdown')
        self.telegram_logger.log_message("MCP help command", update.effective_user.id)
