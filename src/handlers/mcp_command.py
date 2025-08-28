import logging
from typing import Optional
from telegram import Update
from telegram.ext import ContextTypes
from src.services.agent import MCPAgent
from src.utils.log.telegramlog import telegram_logger
from src.services.model_handlers.model_configs import get_default_mcp_model

class MCPCommands:
    """
    Command handlers for MCP (Model Context Protocol) servers.

    This class provides command handlers for interacting with various MCP servers
    including Context7, Exa Search, Sequential Thinking, and Docfork.
    """

    def __init__(self, mcp_agent: MCPAgent):
        self.mcp_agent = mcp_agent
        self.logger = logging.getLogger(__name__)

    async def context7_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /context7 command - Use Context7 MCP for documentation and code examples."""
        await self._handle_mcp_command(update, context, "context7")

    async def exa_search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /exa command - Use Exa Search MCP for web search."""
        await self._handle_mcp_command(update, context, "exa search")

    async def sequentialthinking_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /sequentialthinking command - Use Sequential Thinking MCP for structured reasoning."""
        await self._handle_mcp_command(update, context, "sequentialthinking")

    async def docfork_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /docfork command - Use Docfork MCP for documentation search."""
        await self._handle_mcp_command(update, context, "docfork")

    async def _handle_mcp_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, server_name: str) -> None:
        """Generic handler for MCP server commands."""
        try:
            if not update or not update.message:
                return

            user_id = update.effective_user.id
            chat_id = update.effective_chat.id

            # Get the query from command arguments
            args = context.args
            if not args:
                await update.message.reply_text(
                    f"❌ Please provide a query for {server_name}.\n\n"
                    f"Usage: /{server_name.replace(' ', '')} <your query>"
                )
                return

            query = " ".join(args)

            # Send processing message
            processing_msg = await update.message.reply_text(
                f"🔄 Processing your query with {server_name}..."
            )

            try:
                # Execute MCP query with centralized default model
                response = await self.mcp_agent.execute_mcp_query(
                    server_name=server_name,
                    query=query,
                    model=get_default_mcp_model(),  # Use centralized default
                    temperature=0.7,
                    max_tokens=4000,
                    timeout=300.0
                )

                if response:
                    # Send the response
                    await processing_msg.edit_text(
                        f"🤖 **{server_name.title()} Response:**\n\n{response}"
                    )

                    # Log successful query
                    telegram_logger.log_message(
                        f"MCP query successful: {server_name} - User: {user_id}",
                        chat_id
                    )
                else:
                    await processing_msg.edit_text(
                        f"❌ Failed to get response from {server_name}. Please try again."
                    )

            except Exception as e:
                self.logger.error(f"MCP query failed for {server_name}: {e}")
                await processing_msg.edit_text(
                    f"❌ Error processing {server_name} query. Please try again."
                )

        except Exception as e:
            self.logger.error(f"Error in {server_name} command: {e}")
            if update and update.message:
                await update.message.reply_text(
                    "❌ An error occurred. Please try again."
                )

    async def mcp_list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mcplist command - List all available MCP servers."""
        try:
            if not update or not update.message:
                return

            response = await self.mcp_agent.list_servers_command()
            await update.message.reply_text(response)

        except Exception as e:
            self.logger.error(f"Error in mcplist command: {e}")
            if update and update.message:
                await update.message.reply_text(
                    "❌ Failed to retrieve MCP server list."
                )

    async def mcp_info_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mcpinfo command - Get information about a specific MCP server."""
        try:
            if not update or not update.message:
                return

            args = context.args
            if not args:
                await update.message.reply_text(
                    "❌ Please specify an MCP server name.\n\n"
                    "Usage: /mcpinfo <server_name>\n\n"
                    "Available servers: context7, exa search, sequentialthinking, docfork"
                )
                return

            server_name = " ".join(args)
            response = await self.mcp_agent.server_info_command(server_name)
            await update.message.reply_text(response)

        except Exception as e:
            self.logger.error(f"Error in mcpinfo command: {e}")
            if update and update.message:
                await update.message.reply_text(
                    "❌ Failed to retrieve MCP server information."
                )

    async def mcp_reload_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mcpreload command - Reload MCP server configuration."""
        try:
            if not update or not update.message:
                return

            # Check if user is admin (you might want to add admin check here)
            user_id = update.effective_user.id

            success = self.mcp_agent.reload_config()

            if success:
                await update.message.reply_text(
                    "✅ MCP server configuration reloaded successfully!"
                )
                telegram_logger.log_message(
                    f"MCP config reloaded by user: {user_id}",
                    update.effective_chat.id
                )
            else:
                await update.message.reply_text(
                    "❌ Failed to reload MCP server configuration."
                )

        except Exception as e:
            self.logger.error(f"Error in mcpreload command: {e}")
            if update and update.message:
                await update.message.reply_text(
                    "❌ Failed to reload MCP configuration."
                )

    # Alias commands for convenience
    async def context_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Alias for /context7 command."""
        await self.context7_command(update, context)

    async def exa_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Alias for /exa_search command."""
        await self.exa_search_command(update, context)

    async def thinking_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Alias for /sequentialthinking command."""
        await self.sequentialthinking_command(update, context)

    async def docs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Alias for /docfork command."""
        await self.docfork_command(update, context)
