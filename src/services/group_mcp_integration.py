"""
Enhanced Group MCP Integration for Polymind AI.
Integrates MCP search and research capabilities with group chat features.
"""

import logging
from typing import Dict, Any, Optional, List
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


class GroupMCPIntegration:
    """
    Integrates MCP functionality with group chat features.
    Provides collaborative search and research capabilities.
    """
    
    def __init__(self, mcp_commands=None):
        """Initialize Group MCP Integration."""
        self.mcp_commands = mcp_commands
        self.logger = logger
        
        # Cache for group-specific MCP preferences
        self.group_preferences = {}
        
    async def process_group_search_request(self, 
                                          update: Update, 
                                          context: ContextTypes.DEFAULT_TYPE,
                                          query: str,
                                          search_type: str = "web") -> Optional[Dict[str, Any]]:
        """
        Process search requests in group context with collaborative features.
        
        Args:
            update: Telegram update object
            context: Telegram context
            query: Search query
            search_type: Type of search ("web", "company", "url")
            
        Returns:
            Search result with group context metadata
        """
        try:
            chat_id = update.effective_chat.id
            user_id = update.effective_user.id
            
            # Add group context to the search
            group_context = {
                "chat_id": chat_id,
                "user_id": user_id,
                "search_type": search_type,
                "collaborative": True
            }
            
            # Perform the search using MCP commands
            if self.mcp_commands:
                if search_type == "web":
                    result = await self.mcp_commands.agent.search_web(query)
                elif search_type == "company":
                    result = await self.mcp_commands.agent.research_company(query)
                elif search_type == "url":
                    result = await self.mcp_commands.agent.crawl_website(query)
                else:
                    result = None
                    
                if result:
                    # Enhance result with group metadata
                    result["group_context"] = group_context
                    result["collaborative_search"] = True
                    
                    # Store search in group history (optional)
                    await self._store_group_search(chat_id, user_id, query, search_type, result)
                    
                return result
                
        except Exception as e:
            self.logger.error(f"Error processing group search: {e}")
            
        return None
        
    async def handle_collaborative_research(self, 
                                          update: Update, 
                                          context: ContextTypes.DEFAULT_TYPE,
                                          topic: str) -> Optional[str]:
        """
        Handle collaborative research requests in groups.
        
        Args:
            update: Telegram update object
            context: Telegram context
            topic: Research topic
            
        Returns:
            Formatted research response for group sharing
        """
        try:
            chat_id = update.effective_chat.id
            
            # Perform multiple searches for comprehensive research
            search_results = []
            
            if self.mcp_commands:
                # Web search
                web_result = await self.mcp_commands.agent.search_web(topic)
                if web_result:
                    search_results.append(("Web Search", web_result))
                    
                # Check if it's a company name and do company research
                if any(keyword in topic.lower() for keyword in ["company", "corp", "inc", "ltd"]):
                    company_result = await self.mcp_commands.agent.research_company(topic)
                    if company_result:
                        search_results.append(("Company Research", company_result))
                        
            # Format comprehensive response for group
            if search_results:
                response = f"🔍 **Collaborative Research: {topic}**\n\n"
                
                for search_type, result in search_results:
                    response += f"**{search_type}:**\n"
                    response += f"{result.get('result', 'No results available')}\n\n"
                    
                response += "💡 *This research was collaboratively generated for the group.*\n"
                response += f"🏷️ **Topic:** {topic}"
                
                return response
                
        except Exception as e:
            self.logger.error(f"Error in collaborative research: {e}")
            
        return None
        
    async def get_group_search_history(self, chat_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent search history for a group.
        
        Args:
            chat_id: Group chat ID
            limit: Maximum number of results
            
        Returns:
            List of recent search results
        """
        try:
            # For now, return empty list (would integrate with database in full implementation)
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting group search history: {e}")
            return []
            
    async def set_group_mcp_preferences(self, 
                                       chat_id: int, 
                                       preferences: Dict[str, Any]) -> bool:
        """
        Set MCP preferences for a group.
        
        Args:
            chat_id: Group chat ID
            preferences: MCP preferences dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.group_preferences[chat_id] = preferences
            self.logger.info(f"Set MCP preferences for group {chat_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting group MCP preferences: {e}")
            return False
            
    async def get_group_mcp_preferences(self, chat_id: int) -> Dict[str, Any]:
        """
        Get MCP preferences for a group.
        
        Args:
            chat_id: Group chat ID
            
        Returns:
            Group MCP preferences
        """
        return self.group_preferences.get(chat_id, {
            "auto_search": False,
            "collaborative_mode": True,
            "search_history": True,
            "allowed_search_types": ["web", "company", "url"]
        })
        
    async def _store_group_search(self, 
                                 chat_id: int, 
                                 user_id: int, 
                                 query: str, 
                                 search_type: str, 
                                 result: Dict[str, Any]) -> bool:
        """
        Store search result in group history.
        
        Args:
            chat_id: Group chat ID
            user_id: User ID who performed the search
            query: Search query
            search_type: Type of search
            result: Search result
            
        Returns:
            True if stored successfully
        """
        try:
            # In a full implementation, this would store in the database
            # For now, just log the search
            self.logger.info(f"Group search stored: {chat_id}, user {user_id}, query: {query}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing group search: {e}")
            return False
            
    def is_search_command(self, message_text: str) -> Optional[Dict[str, str]]:
        """
        Check if a message contains a search command and extract details.
        
        Args:
            message_text: Message text to check
            
        Returns:
            Dictionary with search details or None
        """
        try:
            message_lower = message_text.lower().strip()
            
            # Check for explicit search commands
            if message_lower.startswith(("search for", "search", "find", "look up")):
                # Extract query
                for prefix in ["search for", "search", "find", "look up"]:
                    if message_lower.startswith(prefix):
                        query = message_text[len(prefix):].strip()
                        return {
                            "type": "web",
                            "query": query,
                            "command": prefix
                        }
                        
            # Check for company research
            if any(keyword in message_lower for keyword in ["research company", "company info", "about company"]):
                # Extract company name
                for prefix in ["research company", "company info about", "about company"]:
                    if prefix in message_lower:
                        start_idx = message_lower.find(prefix) + len(prefix)
                        query = message_text[start_idx:].strip()
                        return {
                            "type": "company",
                            "query": query,
                            "command": prefix
                        }
                        
            # Check for URL crawling
            if "http" in message_text and any(keyword in message_lower for keyword in ["crawl", "extract", "analyze"]):
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message_text)
                if urls:
                    return {
                        "type": "url",
                        "query": urls[0],
                        "command": "crawl"
                    }
                    
        except Exception as e:
            self.logger.error(f"Error checking search command: {e}")
            
        return None
        
    async def enhance_group_message_with_mcp(self, 
                                           message_text: str, 
                                           chat_id: int) -> Optional[str]:
        """
        Enhance group messages with MCP capabilities if relevant.
        
        Args:
            message_text: Original message text
            chat_id: Group chat ID
            
        Returns:
            Enhanced message with MCP suggestions or None
        """
        try:
            # Check group preferences
            preferences = await self.get_group_mcp_preferences(chat_id)
            
            if not preferences.get("auto_search", False):
                return None
                
            # Detect potential search opportunities
            search_info = self.is_search_command(message_text)
            
            if search_info:
                suggestion = f"💡 *MCP Suggestion:* Use `/{search_info['type']} {search_info['query']}` for enhanced results."
                return suggestion
                
        except Exception as e:
            self.logger.error(f"Error enhancing group message with MCP: {e}")
            
        return None