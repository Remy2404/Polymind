"""
MCP Commands Module - Handles MCP-related Telegram bot commands.
Provides search, company research, URL crawling, and MCP server management commands.
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

# Handle optional imports gracefully
try:
    from ...services.agent import EnhancedAgent, AgentConfig
except ImportError:
    EnhancedAgent = None
    AgentConfig = None
    
try:
    from ...services.user_data_manager import UserDataManager
except ImportError:
    UserDataManager = None
    
try:
    from ...services.model_handlers.api_manager import UnifiedAPIManager
except ImportError:
    UnifiedAPIManager = None
    
try:
    from ...utils.log.telegramlog import TelegramLogger
except ImportError:
    TelegramLogger = None

logger = logging.getLogger(__name__)


class MCPCommands:
    """Handles MCP-related commands for the Telegram bot."""
    
    def __init__(self, 
                 user_data_manager: UserDataManager,
                 telegram_logger: TelegramLogger,
                 api_manager: Optional[UnifiedAPIManager] = None):
        """Initialize MCP commands handler."""
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.api_manager = api_manager
        self.logger = logger
        
        # Initialize the Enhanced Agent
        self.agent_config = AgentConfig(enable_mcp=True)
        self.agent = EnhancedAgent(self.agent_config, api_manager)
        
        # Track initialization
        self._initialized = False
        
    async def _ensure_initialized(self):
        """Ensure the agent is initialized with proper error handling."""
        if not self._initialized:
            try:
                success = await self.agent.initialize()
                self._initialized = success
                if not success:
                    self.logger.warning("Enhanced Agent initialization failed - using fallback mode")
            except Exception as e:
                self.logger.error(f"Agent initialization error: {e}")
                self._initialized = False
                
    async def _get_fallback_response(self, query: str, search_type: str) -> str:
        """Generate fallback response when MCP is not available."""
        try:
            if search_type == "search":
                return (
                    f"🔍 **Search Query:** {query}\n\n"
                    "⚠️ **MCP Search Currently Unavailable**\n\n"
                    "**Alternative Options:**\n"
                    "• Try rephrasing your query\n"
                    "• Use a web browser for real-time results\n"
                    "• Contact admin to configure MCP servers\n\n"
                    "**What I can help with instead:**\n"
                    "• General knowledge questions\n"
                    "• Document analysis\n"
                    "• Code assistance\n"
                    "• Image generation\n\n"
                    "💡 *MCP integration provides enhanced search capabilities when properly configured.*"
                )
            elif search_type == "company":
                return (
                    f"🏢 **Company Research:** {query}\n\n"
                    "⚠️ **MCP Company Research Currently Unavailable**\n\n"
                    "**I can still help with:**\n"
                    "• General business knowledge\n"
                    "• Industry analysis concepts\n"
                    "• Business strategy discussions\n"
                    "• Financial concepts\n\n"
                    "**For real-time data:**\n"
                    "• Check company websites directly\n"
                    "• Use financial news sources\n"
                    "• Visit investor relations pages\n\n"
                    "💡 *When MCP is configured, I can provide real-time company intelligence.*"
                )
            elif search_type == "crawl":
                return (
                    f"🕸️ **URL Content Request:** {query}\n\n"
                    "⚠️ **MCP URL Crawling Currently Unavailable**\n\n"
                    "**Alternative Approaches:**\n"
                    "• Copy and paste the content manually\n"
                    "• Upload content as a document for analysis\n"
                    "• Share specific text excerpts for discussion\n\n"
                    "**I can analyze:**\n"
                    "• Uploaded documents (PDF, DOCX, TXT)\n"
                    "• Pasted text content\n"
                    "• Images with text\n\n"
                    "💡 *MCP integration enables automated content extraction when configured.*"
                )
            else:
                return (
                    f"🔧 **MCP Request:** {query}\n\n"
                    "⚠️ **MCP Services Currently Unavailable**\n\n"
                    "This may be due to:\n"
                    "• Missing API configuration (SMITHERY_API_KEY)\n"
                    "• Network connectivity issues\n"
                    "• Server maintenance\n\n"
                    "**Check:**\n"
                    "• `/mcp status` for system information\n"
                    "• Environment variables are set\n"
                    "• Internet connection is stable\n\n"
                    "💡 *Contact admin if this issue persists.*"
                )
        except Exception as e:
            self.logger.error(f"Error generating fallback response: {e}")
            return f"❌ Unable to process request: {query}\n\nPlease try again or contact support."
                
    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /search command for web search."""
        try:
            await self._ensure_initialized()
            
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            
            # Extract search query from command arguments
            query = " ".join(context.args) if context.args else ""
            
            if not query:
                await update.message.reply_text(
                    "🔍 **Web Search**\n\n"
                    "Please provide a search query.\n\n"
                    "**Usage:** `/search <query>`\n"
                    "**Example:** `/search latest AI developments 2024`",
                    parse_mode='Markdown'
                )
                return
                
            # Send "searching" message
            status_msg = await update.message.reply_text(
                f"🔍 Searching for: *{query}*\n\n"
                "⏳ Please wait...",
                parse_mode='Markdown'
            )
            
            try:
                # Perform search using Enhanced Agent
                search_result = await self.agent.search_web(query)
                
                if search_result and not search_result.get("mock", False):
                    # Format real search results
                    response = self._format_search_results(query, search_result)
                else:
                    # Fallback: Use AI model to simulate search assistance
                    if not self._initialized:
                        # MCP not available, provide fallback response
                        response = await self._get_fallback_response(query, "search")
                    else:
                        # MCP available but returned mock results
                        ai_prompt = f"User is searching for: {query}. Provide helpful information and suggest where they might find current information about this topic."
                        
                        if self.api_manager:
                            current_model = self.user_data_manager.get_user_model(user_id)
                            ai_response = await self.api_manager.generate_response(
                                model_id=current_model,
                                prompt=ai_prompt,
                                max_tokens=1000
                            )
                        else:
                            ai_response = "Search functionality is currently in development. Please try again later."
                            
                        response = f"🔍 **Search Results for:** {query}\n\n{ai_response}\n\n"
                        response += "💡 *Note: This is an AI-generated response. For real-time results, ensure MCP servers are properly configured.*"
                
                # Update status message with results
                await status_msg.edit_text(response, parse_mode='Markdown')
                
                # Log the search
                await self.telegram_logger.log_user_interaction(
                    user_id, f"Search query: {query}", "search_command"
                )
                
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                await status_msg.edit_text(
                    f"❌ Search failed: {str(e)}\n\n"
                    "Please try again or contact support if the issue persists.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            self.logger.error(f"Error in search command: {e}")
            await update.message.reply_text(
                "❌ An error occurred while processing your search request."
            )
            
    async def company_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /company command for company research."""
        try:
            await self._ensure_initialized()
            
            user_id = update.effective_user.id
            
            # Extract company name from command arguments
            company_name = " ".join(context.args) if context.args else ""
            
            if not company_name:
                await update.message.reply_text(
                    "🏢 **Company Research**\n\n"
                    "Please provide a company name to research.\n\n"
                    "**Usage:** `/company <company name>`\n"
                    "**Example:** `/company Tesla`",
                    parse_mode='Markdown'
                )
                return
                
            # Send "researching" message
            status_msg = await update.message.reply_text(
                f"🏢 Researching company: *{company_name}*\n\n"
                "⏳ Gathering business intelligence...",
                parse_mode='Markdown'
            )
            
            try:
                # Perform company research using Enhanced Agent
                research_result = await self.agent.research_company(company_name)
                
                if research_result and not research_result.get("mock", False):
                    # Format real research results
                    response = self._format_company_results(company_name, research_result)
                else:
                    # Fallback: Use AI model for company analysis
                    if not self._initialized:
                        # MCP not available, provide fallback response
                        response = await self._get_fallback_response(company_name, "company")
                    else:
                        # MCP available but returned mock results
                        ai_prompt = f"Provide a comprehensive analysis of the company '{company_name}'. Include information about their business model, recent developments, market position, and key metrics if known."
                        
                        if self.api_manager:
                            current_model = self.user_data_manager.get_user_model(user_id)
                            ai_response = await self.api_manager.generate_response(
                                model_id=current_model,
                                prompt=ai_prompt,
                                max_tokens=1500
                            )
                        else:
                            ai_response = "Company research functionality is currently in development."
                            
                        response = f"🏢 **Company Research:** {company_name}\n\n{ai_response}\n\n"
                        response += "💡 *Note: For real-time data and financial metrics, ensure MCP servers are properly configured.*"
                
                # Update status message with results
                await status_msg.edit_text(response, parse_mode='Markdown')
                
                # Log the research
                await self.telegram_logger.log_user_interaction(
                    user_id, f"Company research: {company_name}", "company_command"
                )
                
            except Exception as e:
                self.logger.error(f"Company research failed: {e}")
                await status_msg.edit_text(
                    f"❌ Company research failed: {str(e)}\n\n"
                    "Please check the company name and try again.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            self.logger.error(f"Error in company command: {e}")
            await update.message.reply_text(
                "❌ An error occurred while processing your company research request."
            )
            
    async def crawl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /crawl command for URL content extraction."""
        try:
            await self._ensure_initialized()
            
            user_id = update.effective_user.id
            
            # Extract URL from command arguments
            url = context.args[0] if context.args else ""
            
            if not url:
                await update.message.reply_text(
                    "🕸️ **URL Crawler**\n\n"
                    "Please provide a URL to crawl and extract content.\n\n"
                    "**Usage:** `/crawl <url>`\n"
                    "**Example:** `/crawl https://example.com/article`",
                    parse_mode='Markdown'
                )
                return
                
            # Validate URL format
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Send "crawling" message
            status_msg = await update.message.reply_text(
                f"🕸️ Crawling URL: `{url}`\n\n"
                "⏳ Extracting content...",
                parse_mode='Markdown'
            )
            
            try:
                # Crawl URL using Enhanced Agent
                crawl_result = await self.agent.crawl_website(url)
                
                if crawl_result and not crawl_result.get("mock", False):
                    # Format real crawl results
                    response = self._format_crawl_results(url, crawl_result)
                else:
                    # Fallback: Inform user about limitations
                    if not self._initialized:
                        # MCP not available, provide fallback response
                        response = await self._get_fallback_response(url, "crawl")
                    else:
                        # MCP available but returned mock results
                        response = f"🕸️ **URL Crawl Request:** {url}\n\n"
                        response += "⚠️ URL crawling functionality requires MCP server configuration.\n\n"
                        response += "**What you can do:**\n"
                        response += "• Copy and paste the content manually\n"
                        response += "• Use document upload if the content is in a file\n"
                        response += "• Contact admin to configure MCP servers\n\n"
                        response += "💡 *Note: Mock crawl result - MCP servers not fully configured.*"
                
                # Update status message with results
                await status_msg.edit_text(response, parse_mode='Markdown')
                
                # Log the crawl attempt
                await self.telegram_logger.log_user_interaction(
                    user_id, f"URL crawl: {url}", "crawl_command"
                )
                
            except Exception as e:
                self.logger.error(f"URL crawl failed: {e}")
                await status_msg.edit_text(
                    f"❌ URL crawl failed: {str(e)}\n\n"
                    "Please check the URL and try again.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            self.logger.error(f"Error in crawl command: {e}")
            await update.message.reply_text(
                "❌ An error occurred while processing your crawl request."
            )
            
    async def mcp_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /mcp command for MCP server management and queries."""
        try:
            await self._ensure_initialized()
            
            user_id = update.effective_user.id
            
            if not context.args:
                # Show MCP status and available commands
                await self._show_mcp_status(update)
                return
                
            subcommand = context.args[0].lower()
            
            if subcommand == "status":
                await self._show_mcp_status(update)
            elif subcommand == "servers":
                await self._show_mcp_servers(update)
            elif subcommand == "tools":
                await self._show_mcp_tools(update)
            elif subcommand == "query" and len(context.args) >= 3:
                server_name = context.args[1]
                query = " ".join(context.args[2:])
                await self._query_mcp_server(update, server_name, query)
            else:
                await update.message.reply_text(
                    "🔧 **MCP Commands**\n\n"
                    "**Available subcommands:**\n"
                    "• `/mcp status` - Show MCP system status\n"
                    "• `/mcp servers` - List available servers\n"
                    "• `/mcp tools` - List available tools\n"
                    "• `/mcp query <server> <query>` - Query specific server\n\n"
                    "**Examples:**\n"
                    "• `/mcp status`\n"
                    "• `/mcp query exa-search artificial intelligence`",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            self.logger.error(f"Error in mcp command: {e}")
            await update.message.reply_text(
                "❌ An error occurred while processing your MCP request."
            )
            
    async def _show_mcp_status(self, update: Update):
        """Show MCP system status."""
        try:
            server_status = self.agent.get_server_status()
            
            response = "🔧 **MCP System Status**\n\n"
            
            if server_status.get("mcp_enabled"):
                response += "✅ MCP System: **Enabled**\n"
                response += f"📊 Total Tools: **{server_status.get('total_tools', 0)}**\n\n"
                
                servers = server_status.get("servers", {})
                if servers:
                    response += "**Server Status:**\n"
                    for name, info in servers.items():
                        status_icon = "🟢" if info.get("connected") else "🔴"
                        enabled_icon = "✅" if info.get("enabled") else "❌"
                        response += f"{status_icon} {enabled_icon} **{name}**\n"
                        response += f"   └ {info.get('description', 'No description')}\n"
                        response += f"   └ Tools: {info.get('tools', 0)}\n\n"
                else:
                    response += "⚠️ No servers configured\n\n"
                    
            else:
                response += "❌ MCP System: **Disabled**\n\n"
                response += "**To enable MCP:**\n"
                response += "• Configure SMITHERY_API_KEY\n"
                response += "• Configure EXA_PROFILE (optional)\n"
                response += "• Check mcp.json configuration\n\n"
                
            response += "💡 Use `/mcp servers` for detailed server info"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error showing MCP status: {e}")
            await update.message.reply_text(
                "❌ Could not retrieve MCP status"
            )
            
    async def _show_mcp_servers(self, update: Update):
        """Show detailed MCP server information."""
        try:
            server_status = self.agent.get_server_status()
            
            response = "🖥️ **MCP Servers**\n\n"
            
            servers = server_status.get("servers", {})
            if servers:
                for name, info in servers.items():
                    response += f"**{name}**\n"
                    response += f"• Status: {'🟢 Connected' if info.get('connected') else '🔴 Disconnected'}\n"
                    response += f"• Enabled: {'✅ Yes' if info.get('enabled') else '❌ No'}\n"
                    response += f"• Type: `{info.get('type', 'Unknown')}`\n"
                    response += f"• Tools: {info.get('tools', 0)}\n"
                    response += f"• Description: {info.get('description', 'No description')}\n\n"
            else:
                response += "⚠️ No servers configured\n\n"
                response += "Check your mcp.json configuration file."
                
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error showing MCP servers: {e}")
            await update.message.reply_text(
                "❌ Could not retrieve MCP server information"
            )
            
    async def _show_mcp_tools(self, update: Update):
        """Show available MCP tools."""
        try:
            tools = self.agent.get_available_tools()
            
            response = "🛠️ **Available MCP Tools**\n\n"
            
            if tools:
                for tool in tools:
                    response += f"• `{tool}`\n"
                response += f"\n**Total:** {len(tools)} tools available\n\n"
                response += "💡 Use `/mcp query <server> <query>` to use tools"
            else:
                response += "⚠️ No tools available\n\n"
                response += "This may be because:\n"
                response += "• MCP servers are not connected\n"
                response += "• No servers are enabled\n"
                response += "• Configuration issues\n\n"
                response += "Check `/mcp status` for more information"
                
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error showing MCP tools: {e}")
            await update.message.reply_text(
                "❌ Could not retrieve MCP tools information"
            )
            
    async def _query_mcp_server(self, update: Update, server_name: str, query: str):
        """Query a specific MCP server."""
        try:
            user_id = update.effective_user.id
            
            # Send "processing" message
            status_msg = await update.message.reply_text(
                f"🔧 Querying server: *{server_name}*\n"
                f"📝 Query: *{query}*\n\n"
                "⏳ Processing...",
                parse_mode='Markdown'
            )
            
            # For now, use the enhanced agent to process the request
            result = await self.agent.process_request(
                f"Query {server_name} server: {query}",
                use_tools=True
            )
            
            if result and result.get("response"):
                response = f"🔧 **MCP Query Result**\n\n"
                response += f"**Server:** {server_name}\n"
                response += f"**Query:** {query}\n\n"
                response += f"**Result:**\n{result['response']}\n\n"
                
                if result.get("tool_results"):
                    response += "**Tool Results:**\n"
                    for tool_result in result["tool_results"]:
                        response += f"• {tool_result.get('tool', 'Unknown')}: {tool_result.get('result', 'No result')}\n"
                        
            else:
                response = f"❌ No response from server: {server_name}\n\n"
                response += "The server may be unavailable or the query format may be incorrect."
                
            await status_msg.edit_text(response, parse_mode='Markdown')
            
            # Log the query
            await self.telegram_logger.log_user_interaction(
                user_id, f"MCP query to {server_name}: {query}", "mcp_query"
            )
            
        except Exception as e:
            self.logger.error(f"Error querying MCP server: {e}")
            await update.message.reply_text(
                f"❌ Error querying server {server_name}: {str(e)}"
            )
            
    def _format_search_results(self, query: str, search_result: Dict) -> str:
        """Format search results for display."""
        response = f"🔍 **Search Results for:** {query}\n\n"
        
        if search_result.get("result"):
            response += f"{search_result['result']}\n\n"
            
        response += f"**Source:** {search_result.get('server', 'Unknown')}\n"
        response += f"**Tool:** {search_result.get('tool', 'Unknown')}"
        
        return response
        
    def _format_company_results(self, company_name: str, research_result: Dict) -> str:
        """Format company research results for display."""
        response = f"🏢 **Company Research:** {company_name}\n\n"
        
        if research_result.get("result"):
            response += f"{research_result['result']}\n\n"
            
        response += f"**Source:** {research_result.get('server', 'Unknown')}\n"
        response += f"**Tool:** {research_result.get('tool', 'Unknown')}"
        
        return response
        
    def _format_crawl_results(self, url: str, crawl_result: Dict) -> str:
        """Format URL crawl results for display."""
        response = f"🕸️ **Content from:** {url}\n\n"
        
        if crawl_result.get("result"):
            content = crawl_result["result"]
            # Truncate if too long
            if len(content) > 2000:
                content = content[:2000] + "...\n\n[Content truncated]"
            response += f"{content}\n\n"
            
        response += f"**Source:** {crawl_result.get('server', 'Unknown')}\n"
        response += f"**Tool:** {crawl_result.get('tool', 'Unknown')}"
        
        return response