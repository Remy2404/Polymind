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
        Execute MCP tool directly using OpenRouter API and existing response formatter.
        This follows the workflow: openrouter_api.py -> existing response formatting
        
        Args:
            update: Telegram update object
            context: Telegram context
            tool_name: Name of the MCP tool to execute
            query: User query to process
            server_name: Optional server name for logging
        """
        user_id = update.effective_user.id
        
        try:
            # Log the MCP command request
            self.telegram_logger.log_message(f"MCP {tool_name} command: {query[:100]}...", user_id)
            
            # Send processing message
            processing_message = await update.message.reply_text(
                f"🔄 **Processing {tool_name}**\n"
                f"Query: `{query[:100]}{'...' if len(query) > 100 else ''}`\n\n"
                "⏳ Executing tool...",
                parse_mode='Markdown'
            )
            
            # Create enhanced prompt for the AI that lets it choose the appropriate MCP tool
            enhanced_prompt = f"""I need help with: {query}

Please use the most appropriate available MCP tool to help answer this query. You have access to web search, company research, documentation search, and other specialized tools through the Model Context Protocol.

Provide a comprehensive and helpful response using the available tools."""
            
            # Use OpenRouter API directly with MCP tools
            if self.openrouter_api:
                response, tool_logger = await self.openrouter_api.generate_response_with_tool_logging(
                    prompt=enhanced_prompt,
                    model="qwen3-235b",  # Use a reliable default model
                    use_mcp=True,
                    timeout=120.0
                )
                
                # Format response with tool information
                if response and tool_logger.tool_calls:
                    formatted_response = f"🔧 **{tool_name} Results**\n"
                    formatted_response += "═" * 30 + "\n\n"
                    
                    # Add tool execution summary
                    for tool_call in tool_logger.tool_calls:
                        if tool_call.duration_ms:
                            formatted_response += f"✅ Tool executed in {tool_call.duration_ms:.0f}ms\n\n"
                    
                    formatted_response += response
                    
                    # Edit the processing message with results
                    await processing_message.edit_text(
                        formatted_response,
                        parse_mode='Markdown',
                        disable_web_page_preview=True
                    )
                elif response:
                    # AI responded but no tools were called - still show the response
                    formatted_response = f"🤖 **AI Response** (no tool used)\n"
                    formatted_response += "═" * 30 + "\n\n"
                    formatted_response += response
                    
                    await processing_message.edit_text(
                        formatted_response,
                        parse_mode='Markdown',
                        disable_web_page_preview=True
                    )
                else:
                    # No tool was called and no response
                    await processing_message.edit_text(
                        f"❌ **{tool_name} Failed**\n\n"
                        f"No response generated. Please try rephrasing your query. #{server_name if server_name else 'MCP'}",
                        parse_mode='Markdown'
                    )
            else:
                await processing_message.edit_text(
                    "❌ OpenRouter API not available. Please try again later.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            self.logger.error(f"Error executing MCP tool {tool_name}: {e}")
            # Edit processing message with error
            try:
                await processing_message.edit_text(
                    f"❌ **{tool_name} Error**\n\n"
                    f"Error: `{str(e)[:200]}{'...' if len(str(e)) > 200 else ''}`\n\n"
                    "Please try again or contact support.",
                    parse_mode='Markdown'
                )
            except:
                # If editing fails, send new message
                await update.message.reply_text(
                    f"❌ **{tool_name} Error**\n\n"
                    f"Error: `{str(e)[:200]}{'...' if len(str(e)) > 200 else ''}`\n\n"
                    "Please try again or contact support.",
                    parse_mode='Markdown'
                )
            
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
        
        await self._execute_mcp_tool(update, context, "web_search_exa", query, "Exa Search")
        
    async def company_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /company command - Company research via Exa Search MCP tool."""
        user_id = update.effective_user.id
        
        if not context.args:
            await update.message.reply_text(
                "🏢 **Usage:** `/company <company_name>`\n\n"
                "**Example:** `/company Tesla`\n\n"
                "Performs company research using Exa AI search engine.",
                parse_mode='Markdown'
            )
            return
            
        query = " ".join(context.args)
        self.telegram_logger.log_message(f"Company research command: {query}", user_id)
        
        await self._execute_mcp_tool(update, context, "company_research", query, "Exa Search")
        
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
        
        await self._execute_mcp_tool(update, context, "mcp_context7_resolve-library-id", query, "Context7")
        
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
        
        await self._execute_mcp_tool(update, context, "mcp_sequentialthi_sequentialthinking", query, "Sequential Thinking")
        
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
        
        await self._execute_mcp_tool(update, context, "mcp_docfork_get-library-docs", query, "Docfork")
        
    async def mcp_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mcp_status command - Show MCP server status."""
        user_id = update.effective_user.id
        self.telegram_logger.log_message("MCP status command", user_id)
        
        try:
            # Get MCP registry
            registry = await get_mcp_registry()
            
            # Get server information
            servers_info = registry.get_available_tools_info()
            
            # Check integration status
            integration_ready = await self._ensure_mcp_integration()
            
            # Format status message
            status_message = "🔧 **MCP Status**\n"
            status_message += "═" * 30 + "\n\n"
            
            if integration_ready:
                status_message += "✅ **MCP Integration**: Ready\n\n"
            else:
                status_message += "❌ **MCP Integration**: Not Available\n\n"
                
            status_message += "**Available Servers**:\n"
            for server_name, info in servers_info.items():
                status_message += f"• {server_name}: {len(info.get('tools', []))} tools\n"
            
            await update.message.reply_text(status_message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error getting MCP status: {e}")
            error_message = f"❌ **MCP Status Error**\n\n" \
                          f"Error: `{str(e)[:200]}{'...' if len(str(e)) > 200 else ''}`"
            await update.message.reply_text(error_message, parse_mode='Markdown')
            
    async def mcp_help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mcp_help command - Show MCP commands help."""
        help_text = """🔧 **MCP Commands Help**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Available Commands:**

🔍 `/search <query>`
   Web search using Exa AI search engine
   *Example:* `/search latest AI developments`

🏢 `/company <company_name>`
   Company research using Exa AI search engine
   *Example:* `/company Tesla`

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
