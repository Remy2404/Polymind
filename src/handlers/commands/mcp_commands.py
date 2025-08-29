"""
MCP Command Handlers for Telegram Bot
Provides user-facing commands that interact with MCP tools via Pydantic AI agents.
"""

import sys
import os
import logging
from typing import Optional, Dict, List, Any

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

    def __init__(
        self, user_data_manager, telegram_logger: TelegramLogger, openrouter_api=None
    ):
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
        self.discovered_tools = {}  # Cache for discovered tools

    def set_openrouter_api(self, openrouter_api):
        """Set the OpenRouter API instance after initialization."""
        self.openrouter_api = openrouter_api

    async def _discover_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover available tools from MCP servers dynamically."""
        if self.discovered_tools:
            return self.discovered_tools

        try:
            registry = await get_mcp_registry()
            self.discovered_tools = await registry.discover_available_tools()
            return self.discovered_tools
        except Exception as e:
            self.logger.error(f"Failed to discover MCP tools: {e}")
            return {}

    def _truncate_message(self, message: str, max_length: int = 4000) -> str:
        """Truncate message to fit Telegram limits with proper formatting."""
        if len(message) <= max_length:
            return message

        # Find a good break point (end of a paragraph or sentence)
        truncated = message[: max_length - 100]  # Leave room for truncation notice

        # Try to break at paragraph
        last_paragraph = truncated.rfind("\n\n")
        if last_paragraph > max_length * 0.8:  # If paragraph break is not too early
            truncated = truncated[:last_paragraph]
        else:
            # Try to break at sentence
            last_sentence = max(
                truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? ")
            )
            if last_sentence > max_length * 0.7:  # If sentence break is reasonable
                truncated = truncated[: last_sentence + 1]

        return truncated + f"\n\n... *[Message truncated due to length limit]*"

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
        if (
            not hasattr(self.openrouter_api, "mcp_registry")
            or not self.openrouter_api.mcp_registry
        ):
            self.logger.error("MCP registry not available in OpenRouter API")
            return False

        return True

    async def _execute_mcp_tool(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        tool_name: str,
        query: str,
        server_name: Optional[str] = None,
    ) -> None:
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
            self.telegram_logger.log_message(
                f"MCP {tool_name} command: {query[:100]}...", user_id
            )

            # Send processing message
            processing_message = await update.message.reply_text(
                f"🔄 **Processing {tool_name}**\n"
                f"Query: `{query[:100]}{'...' if len(query) > 100 else ''}`\n\n"
                "⏳ Executing tool...",
                parse_mode="Markdown",
            )

            # Create enhanced prompt for the AI that lets it choose the appropriate MCP tool
            enhanced_prompt = f"""I need help with: {query}

Please use the most appropriate available MCP tool to help answer this query. You have access to web search, company research, documentation search, and other specialized tools through the Model Context Protocol.

Provide a comprehensive and helpful response using the available tools."""

            # Use OpenRouter API directly with MCP tools
            if self.openrouter_api:
                response, tool_logger = (
                    await self.openrouter_api.generate_response_with_tool_logging(
                        prompt=enhanced_prompt,
                        model="qwen3-235b",  # Use a reliable default model
                        use_mcp=True,
                        timeout=120.0,
                    )
                )

                # Format response with tool information
                if response and tool_logger.tool_calls:
                    formatted_response = f"🔧 **{tool_name} Results**\n"
                    formatted_response += "═" * 30 + "\n\n"

                    # Add tool execution summary
                    for tool_call in tool_logger.tool_calls:
                        if tool_call.duration_ms:
                            formatted_response += (
                                f"✅ Tool executed in {tool_call.duration_ms:.0f}ms\n\n"
                            )

                    formatted_response += response

                    # Truncate if too long for Telegram
                    formatted_response = self._truncate_message(formatted_response)

                    # Edit the processing message with results
                    await processing_message.edit_text(
                        formatted_response,
                        parse_mode="Markdown",
                        disable_web_page_preview=True,
                    )
                elif response:
                    # AI responded but no tools were called - still show the response
                    formatted_response = f"🤖 **AI Response** (no tool used)\n"
                    formatted_response += "═" * 30 + "\n\n"
                    formatted_response += response

                    # Truncate if too long
                    formatted_response = self._truncate_message(formatted_response)

                    await processing_message.edit_text(
                        formatted_response,
                        parse_mode="Markdown",
                        disable_web_page_preview=True,
                    )
                else:
                    # No tool was called and no response
                    await processing_message.edit_text(
                        f"❌ **{tool_name} Failed**\n\n"
                        f"No response generated. Please try rephrasing your query. #{server_name if server_name else 'MCP'}",
                        parse_mode="Markdown",
                    )
            else:
                await processing_message.edit_text(
                    "❌ OpenRouter API not available. Please try again later.",
                    parse_mode="Markdown",
                )

        except Exception as e:
            self.logger.error(f"Error executing MCP tool {tool_name}: {e}")
            # Edit processing message with error
            try:
                await processing_message.edit_text(
                    f"❌ **{tool_name} Error**\n\n"
                    f"Error: `{str(e)[:200]}{'...' if len(str(e)) > 200 else ''}`\n\n"
                    "Please try again or contact support.",
                    parse_mode="Markdown",
                )
            except:
                # If editing fails, send new message
                await update.message.reply_text(
                    f"❌ **{tool_name} Error**\n\n"
                    f"Error: `{str(e)[:200]}{'...' if len(str(e)) > 200 else ''}`\n\n"
                    "Please try again or contact support.",
                    parse_mode="Markdown",
                )

    async def search_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /search command - Web search using available MCP tools."""
        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text(
                "📝 **Usage:** `/search <query>`\n\n"
                "**Example:** `/search latest developments in AI`\n\n"
                "Performs web search using available MCP tools.",
                parse_mode="Markdown",
            )
            return

        query = " ".join(context.args)
        self.telegram_logger.log_message(f"Search command: {query}", user_id)

        # Discover available tools and find search-related tools
        discovered_tools = await self._discover_tools()
        search_tool = None

        # Look for search tools in Exa Search server
        if "Exa Search" in discovered_tools:
            for tool in discovered_tools["Exa Search"]:
                if "search" in tool["name"].lower() and "web" in tool["name"].lower():
                    search_tool = tool["name"]
                    break

        if search_tool:
            await self._execute_mcp_tool(
                update, context, search_tool, query, "Exa Search"
            )
        else:
            await update.message.reply_text(
                "❌ **Search Unavailable**\n\n"
                "No web search tools are currently available from MCP servers.",
                parse_mode="Markdown",
            )

    async def company_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /company command - Company research using available MCP tools."""
        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text(
                "🏢 **Usage:** `/company <company_name>`\n\n"
                "**Example:** `/company Tesla`\n\n"
                "Performs company research using available MCP tools.",
                parse_mode="Markdown",
            )
            return

        query = " ".join(context.args)
        self.telegram_logger.log_message(f"Company research command: {query}", user_id)

        # Discover available tools and find company research tools
        discovered_tools = await self._discover_tools()
        company_tool = None

        # Look for company research tools
        for server_name, tools in discovered_tools.items():
            for tool in tools:
                if (
                    "company" in tool["name"].lower()
                    or "research" in tool["name"].lower()
                ):
                    company_tool = tool["name"]
                    server_name_for_tool = server_name
                    break
            if company_tool:
                break

        if company_tool:
            await self._execute_mcp_tool(
                update, context, company_tool, query, server_name_for_tool
            )
        else:
            await update.message.reply_text(
                "❌ **Company Research Unavailable**\n\n"
                "No company research tools are currently available from MCP servers.",
                parse_mode="Markdown",
            )

    async def context7_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /Context7 command - Documentation search using available MCP tools."""
        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text(
                "📚 **Usage:** `/Context7 <query>`\n\n"
                "**Example:** `/Context7 FastAPI middleware`\n\n"
                "Searches library documentation using available MCP tools.",
                parse_mode="Markdown",
            )
            return

        query = " ".join(context.args)
        self.telegram_logger.log_message(f"Context7 command: {query}", user_id)

        # Discover available tools and find documentation tools
        discovered_tools = await self._discover_tools()
        doc_tool = None

        # Look for documentation/library tools in Context7 server
        if "Context7" in discovered_tools:
            for tool in discovered_tools["Context7"]:
                if "library" in tool["name"].lower() or "docs" in tool["name"].lower():
                    doc_tool = tool["name"]
                    break

        if doc_tool:
            await self._execute_mcp_tool(update, context, doc_tool, query, "Context7")
        else:
            await update.message.reply_text(
                "❌ **Documentation Search Unavailable**\n\n"
                "No documentation search tools are currently available from MCP servers.",
                parse_mode="Markdown",
            )

    async def sequentialthinking_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /sequentialthinking command - Problem-solving using available MCP tools."""
        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text(
                "🧠 **Usage:** `/sequentialthinking <problem>`\n\n"
                "**Example:** `/sequentialthinking How to optimize database queries`\n\n"
                "Uses sequential thinking to break down complex problems.",
                parse_mode="Markdown",
            )
            return

        query = " ".join(context.args)
        self.telegram_logger.log_message(
            f"Sequential thinking command: {query}", user_id
        )

        # Discover available tools and find sequential thinking tools
        discovered_tools = await self._discover_tools()
        thinking_tool = None

        # Look for sequential thinking tools
        if "sequentialthinking" in discovered_tools:
            for tool in discovered_tools["sequentialthinking"]:
                if (
                    "sequential" in tool["name"].lower()
                    or "thinking" in tool["name"].lower()
                ):
                    thinking_tool = tool["name"]
                    break

        if thinking_tool:
            await self._execute_mcp_tool(
                update, context, thinking_tool, query, "Sequential Thinking"
            )
        else:
            await update.message.reply_text(
                "❌ **Sequential Thinking Unavailable**\n\n"
                "No sequential thinking tools are currently available from MCP servers.",
                parse_mode="Markdown",
            )

    async def docfork_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /Docfork command - Document analysis using available MCP tools."""
        user_id = update.effective_user.id

        if not context.args:
            await update.message.reply_text(
                "📄 **Usage:** `/Docfork <document_url_or_query>`\n\n"
                "**Example:** `/Docfork https://example.com/document.pdf`\n\n"
                "Analyzes documents using available MCP tools.",
                parse_mode="Markdown",
            )
            return

        query = " ".join(context.args)
        self.telegram_logger.log_message(f"Docfork command: {query}", user_id)

        # Discover available tools and find document analysis tools
        discovered_tools = await self._discover_tools()
        doc_tool = None

        # Look for document analysis tools in Docfork server
        if "Docfork" in discovered_tools:
            for tool in discovered_tools["Docfork"]:
                if "library" in tool["name"].lower() or "docs" in tool["name"].lower():
                    doc_tool = tool["name"]
                    break

        if doc_tool:
            await self._execute_mcp_tool(update, context, doc_tool, query, "Docfork")
        else:
            await update.message.reply_text(
                "❌ **Document Analysis Unavailable**\n\n"
                "No document analysis tools are currently available from MCP servers.",
                parse_mode="Markdown",
            )

    async def mcp_status_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
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
                status_message += (
                    f"• {server_name}: {len(info.get('tools', []))} tools\n"
                )

            await update.message.reply_text(status_message, parse_mode="Markdown")

        except Exception as e:
            self.logger.error(f"Error getting MCP status: {e}")
            error_message = (
                f"❌ **MCP Status Error**\n\n"
                f"Error: `{str(e)[:200]}{'...' if len(str(e)) > 200 else ''}`"
            )

    async def mcp_tools_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /mcp_tools command - List all available MCP tools."""
        user_id = update.effective_user.id
        self.telegram_logger.log_message("MCP tools command", user_id)

        try:
            # Discover all available tools
            discovered_tools = await self._discover_tools()

            # Format tools message
            tools_message = "🔧 **Available MCP Tools**\n"
            tools_message += "═" * 30 + "\n\n"

            if not discovered_tools:
                tools_message += "❌ No tools discovered from MCP servers.\n"
            else:
                for server_name, tools in discovered_tools.items():
                    tools_message += f"**{server_name}:**\n"
                    if tools:
                        for tool in tools:
                            tools_message += (
                                f"• `{tool['name']}` - {tool['description']}\n"
                            )
                    else:
                        tools_message += "• No tools available\n"
                    tools_message += "\n"

            # Truncate if too long
            tools_message = self._truncate_message(tools_message)

            await update.message.reply_text(tools_message, parse_mode="Markdown")

        except Exception as e:
            self.logger.error(f"Error listing MCP tools: {e}")
            error_message = (
                f"❌ **MCP Tools Error**\n\n"
                f"Error: `{str(e)[:200]}{'...' if len(str(e)) > 200 else ''}`"
            )
            await update.message.reply_text(error_message, parse_mode="Markdown")

    async def mcp_help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /mcp_help command - Show MCP commands help."""
        help_text = """🔧 **MCP Commands Help**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Available Commands:**

🔍 `/search <query>`
   Web search using available MCP tools
   *Example:* `/search latest AI developments`

🏢 `/company <company_name>`
   Company research using available MCP tools
   *Example:* `/company Tesla`

📚 `/Context7 <query>`  
   Search library documentation using available MCP tools
   *Example:* `/Context7 FastAPI middleware`

🧠 `/sequentialthinking <problem>`
   Break down complex problems using available MCP tools
   *Example:* `/sequentialthinking optimize database`

📄 `/Docfork <document_url_or_query>`
   Analyze documents using available MCP tools
   *Example:* `/Docfork https://example.com/doc.pdf`

⚙️ `/mcp_status`
   Show status of all MCP servers

🔧 `/mcp_tools`
   List all available MCP tools from all servers

❓ `/mcp_help`
   Show this help message

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**About MCP:**
Model Context Protocol provides AI agents with access to external tools and data sources, enabling more powerful and context-aware responses.

**Note:** Commands automatically discover and use available tools from configured MCP servers. No hardcoded tool names are used."""

        await update.message.reply_text(help_text, parse_mode="Markdown")
        self.telegram_logger.log_message("MCP help command", update.effective_user.id)
