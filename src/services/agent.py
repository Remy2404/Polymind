from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from .model_handlers.model_configs import (
    ModelConfigurations,
)
from .mcp_registry import MCPRegistry
from .simple_mcp_executor import SimpleMCPExecutor


class SearchResult(BaseModel):
    summary: str = Field(description="Concise answer text with proper Telegram formatting")
    sources: List[str] = Field(default_factory=list, description="Cited source URLs")


@dataclass
class AgentDeps:
    user_id: Optional[int] = None
    username: Optional[str] = None
    preferred_model: Optional[str] = None


class EnhancedAgent:
    """Wraps a Pydantic AI Agent with dynamic MCP tools registration."""

    def __init__(self, registry: Optional[MCPRegistry] = None, preferred_model: Optional[str] = None) -> None:
        self.registry = registry or MCPRegistry()
        self.mcp_executor = SimpleMCPExecutor(self.registry)
        self.logger = logging.getLogger(__name__)

        # Always use user's preferred model from model_configs.py
        all_models = ModelConfigurations.get_all_models()

        # If preferred_model is not set or invalid, default to "gemini-flash"
        model_key = preferred_model if preferred_model in all_models else "gemini-flash"
        cfg = all_models[model_key]

        # Resolve API model identifier using centralized function
        model_id = ModelConfigurations.resolve_api_model_id(cfg.model_id)

        # System prompt
        base_prompt = (getattr(cfg, "system_message", None) or "You are a concise research assistant.")
        system_prompt = (base_prompt.rstrip() + 
                        " Always cite 2-5 sources. Format responses using Telegram markdown: " +
                        "use *bold*, _italic_, `code`, [links](url), and ```code blocks```. " +
                        "Keep responses concise but informative.")

        self.agent = Agent(
            model_id,
            deps_type=AgentDeps,
            output_type=SearchResult,
            system_prompt=system_prompt,
        )

        self.logger.info(f"Research agent initialized with model: {model_id}")
        self._register_dynamic_tools()

    def _register_dynamic_tools(self) -> None:
        # Register MCP tools that use the simple executor
        async def exa_search_web_search_exa(ctx: RunContext[AgentDeps], query: str) -> str:
            """Search the web using Exa AI."""
            result = await self.mcp_executor.execute_exa_search(query)
            return result

        async def exa_search_company_research_exa(ctx: RunContext[AgentDeps], company_name: str) -> str:
            """Research a company using Exa AI."""
            result = await self.mcp_executor.execute_exa_company_research(company_name)
            return result

        async def exa_search_crawling_exa(ctx: RunContext[AgentDeps], url: str) -> str:
            """Crawl and extract content from a URL using Exa AI."""
            result = await self.mcp_executor.execute_exa_crawl(url)
            return result

        async def context7_search(ctx: RunContext[AgentDeps], query: str) -> str:
            """Search library documentation using Context7."""
            result = await self.mcp_executor.execute_context7_search(query)
            return result

        async def duckduckgo_search(ctx: RunContext[AgentDeps], query: str) -> str:
            """Search using DuckDuckGo."""
            result = await self.mcp_executor.execute_duckduckgo_search(query)
            return result

        # Register the tools with the agent
        self.agent.tool(exa_search_web_search_exa)
        self.agent.tool(exa_search_company_research_exa)
        self.agent.tool(exa_search_crawling_exa)
        self.agent.tool(context7_search)
        self.agent.tool(duckduckgo_search)

    async def run(self, query: str, deps: Optional[AgentDeps] = None) -> SearchResult:
        try:
            # Log the attempt
            self.logger.info(f"Running agent query: {query[:100]}...")
            
            result = await self.agent.run(query, deps=deps)
            output = result.output
            
            # Ensure minimal citations expectation
            if isinstance(output, SearchResult) and not output.sources:
                # Request retry via validator-style mechanism
                raise ModelRetry('Please include at least 2 source URLs in sources[].')
            
            self.logger.info(f"Agent query completed successfully with {len(output.sources)} sources")
            return output
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Agent run failed: {error_msg}")
            
            # Handle specific error types
            if "429" in error_msg or "Rate limit" in error_msg:
                return SearchResult(
                    summary="🔄 *Rate limit reached*. The search service is temporarily busy. Please try again in a few minutes.\n\n*Alternative:* Use `/search` command for web searches or try a different query.",
                    sources=[]
                )
            elif "NetworkTimeout" in error_msg or "connection" in error_msg.lower():
                return SearchResult(
                    summary="🔌 *Connection issue*. Database or API service is temporarily unavailable. Please try again later.\n\n*Status:* Service maintenance in progress.",
                    sources=[]
                )
            else:
                # Return a basic error response instead of letting it bubble up
                return SearchResult(
                    summary=f"❌ *Search failed*: {error_msg[:200]}...\n\n*Suggestion:* Try rephrasing your query or use a different search method.",
                    sources=[]
                )


__all__ = ["EnhancedAgent", "AgentDeps", "SearchResult"]
