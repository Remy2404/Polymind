"""
Pydantic AI Agent Integration for Polymind AI.
Provides an enhanced agent that can orchestrate multiple MCP servers and AI models.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from .mcp_registry import MCPRegistry, MCPTool
from .model_handlers.api_manager import UnifiedAPIManager

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the Enhanced Agent."""
    
    default_model: str = "gemini"
    max_tokens: int = 32000
    temperature: float = 0.7
    enable_mcp: bool = True
    mcp_config_path: str = "mcp.json"
    system_message: Optional[str] = None


class EnhancedAgent:
    """
    Enhanced AI Agent that combines Polymind's model system with MCP tools.
    Orchestrates requests across multiple AI models and MCP servers.
    """
    
    def __init__(self, config: AgentConfig = None, model_manager: UnifiedAPIManager = None):
        """Initialize the Enhanced Agent."""
        self.config = config or AgentConfig()
        self.model_manager = model_manager
        self.mcp_registry = MCPRegistry(self.config.mcp_config_path) if self.config.enable_mcp else None
        self.available_tools: List[MCPTool] = []
        self.logger = logger
        
    async def initialize(self) -> bool:
        """Initialize the agent and its components."""
        try:
            # Initialize MCP registry if enabled
            if self.mcp_registry:
                if not await self.mcp_registry.initialize():
                    self.logger.warning("MCP registry initialization failed - continuing without MCP tools")
                    self.mcp_registry = None
                else:
                    self.available_tools = self.mcp_registry.get_all_tools()
                    self.logger.info(f"Agent initialized with {len(self.available_tools)} MCP tools")
            
            self.logger.info("Enhanced Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced Agent: {e}")
            return False
            
    async def process_request(self, 
                            user_message: str, 
                            context: Optional[List[Dict[str, Any]]] = None,
                            model_id: Optional[str] = None,
                            use_tools: bool = True) -> Dict[str, Any]:
        """
        Process a user request with AI model and optional MCP tools.
        
        Args:
            user_message: The user's message/request
            context: Previous conversation context
            model_id: Specific model to use (defaults to configured default)
            use_tools: Whether to use MCP tools
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Determine which model to use
            target_model = model_id or self.config.default_model
            
            # Analyze if the request needs tool assistance
            needs_tools = use_tools and self._requires_tool_assistance(user_message)
            
            response_data = {
                "user_message": user_message,
                "model_used": target_model,
                "tools_used": [],
                "response": "",
                "tool_results": [],
                "metadata": {}
            }
            
            # Execute tools if needed
            if needs_tools and self.mcp_registry:
                tool_results = await self._execute_relevant_tools(user_message)
                response_data["tool_results"] = tool_results
                
                # Add tool context to the prompt
                enhanced_prompt = self._enhance_prompt_with_tools(user_message, tool_results)
            else:
                enhanced_prompt = user_message
                
            # Generate AI response
            if self.model_manager:
                ai_response = await self.model_manager.generate_response(
                    model_id=target_model,
                    prompt=enhanced_prompt,
                    context=context,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                response_data["response"] = ai_response
            else:
                response_data["response"] = "AI model manager not available"
                
            response_data["metadata"] = {
                "enhanced_prompt_used": needs_tools,
                "tools_available": len(self.available_tools),
                "mcp_enabled": self.mcp_registry is not None
            }
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Failed to process request: {e}")
            return {
                "user_message": user_message,
                "response": f"Error processing request: {str(e)}",
                "error": True,
                "metadata": {}
            }
            
    def _requires_tool_assistance(self, message: str) -> bool:
        """Determine if a message requires tool assistance."""
        # Simple keyword-based detection
        search_keywords = ["search", "find", "look up", "research", "company", "website", "url", "crawl"]
        message_lower = message.lower()
        
        return any(keyword in message_lower for keyword in search_keywords)
        
    async def _execute_relevant_tools(self, message: str) -> List[Dict[str, Any]]:
        """Execute relevant MCP tools based on the message content."""
        results = []
        
        if not self.mcp_registry:
            return results
            
        try:
            message_lower = message.lower()
            
            # Detect search queries
            if any(keyword in message_lower for keyword in ["search", "find", "look up"]):
                # Extract search query (simple approach)
                search_query = self._extract_search_query(message)
                if search_query:
                    search_result = await self.mcp_registry.search_with_exa(search_query, "web")
                    if search_result:
                        results.append(search_result)
                        
            # Detect company research requests
            if any(keyword in message_lower for keyword in ["company", "business", "corporation"]):
                company_query = self._extract_company_name(message)
                if company_query:
                    company_result = await self.mcp_registry.search_with_exa(company_query, "company")
                    if company_result:
                        results.append(company_result)
                        
            # Detect URL crawling requests
            if any(keyword in message_lower for keyword in ["url", "website", "crawl", "extract"]):
                urls = self._extract_urls(message)
                for url in urls:
                    crawl_result = await self.mcp_registry.crawl_url(url)
                    if crawl_result:
                        results.append(crawl_result)
                        
        except Exception as e:
            self.logger.error(f"Error executing tools: {e}")
            
        return results
        
    def _extract_search_query(self, message: str) -> Optional[str]:
        """Extract search query from message (simple implementation)."""
        # Remove common command words and return the rest
        words = message.lower().split()
        query_words = []
        skip_words = {"search", "for", "find", "look", "up", "please", "can", "you"}
        
        for word in words:
            if word not in skip_words and len(word) > 2:
                query_words.append(word)
                
        return " ".join(query_words) if query_words else None
        
    def _extract_company_name(self, message: str) -> Optional[str]:
        """Extract company name from message (simple implementation)."""
        # Look for company-related patterns
        words = message.split()
        for i, word in enumerate(words):
            if word.lower() in ["company", "corp", "corporation", "inc", "ltd"]:
                # Take words before the company identifier
                if i > 0:
                    return " ".join(words[max(0, i-2):i])
        
        # Fallback: return the whole message minus command words
        return self._extract_search_query(message)
        
    def _extract_urls(self, message: str) -> List[str]:
        """Extract URLs from message (simple implementation)."""
        import re
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, message)
        
    def _enhance_prompt_with_tools(self, original_prompt: str, tool_results: List[Dict[str, Any]]) -> str:
        """Enhance the prompt with tool results."""
        if not tool_results:
            return original_prompt
            
        enhanced_prompt = f"Original request: {original_prompt}\n\n"
        enhanced_prompt += "Additional context from tools:\n"
        
        for i, result in enumerate(tool_results, 1):
            enhanced_prompt += f"{i}. Tool: {result.get('tool', 'Unknown')}\n"
            enhanced_prompt += f"   Result: {result.get('result', 'No result')}\n\n"
            
        enhanced_prompt += "Please provide a comprehensive response using both the original request and the additional context above."
        
        return enhanced_prompt
        
    async def search_web(self, query: str) -> Optional[Dict[str, Any]]:
        """Perform web search using MCP tools."""
        if not self.mcp_registry:
            return None
            
        return await self.mcp_registry.search_with_exa(query, "web")
        
    async def research_company(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Research a company using MCP tools."""
        if not self.mcp_registry:
            return None
            
        return await self.mcp_registry.search_with_exa(company_name, "company")
        
    async def crawl_website(self, url: str) -> Optional[Dict[str, Any]]:
        """Crawl a website using MCP tools."""
        if not self.mcp_registry:
            return None
            
        return await self.mcp_registry.crawl_url(url)
        
    def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools."""
        return [tool.name for tool in self.available_tools]
        
    def get_server_status(self) -> Dict[str, Any]:
        """Get status of MCP servers."""
        if not self.mcp_registry:
            return {"mcp_enabled": False}
            
        return {
            "mcp_enabled": True,
            "servers": self.mcp_registry.get_server_status(),
            "total_tools": len(self.available_tools)
        }
        
    async def query_specific_server(self, server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Query a specific MCP server with specific parameters."""
        if not self.mcp_registry:
            return None
            
        return await self.mcp_registry.execute_tool(tool_name, parameters)