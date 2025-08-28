from __future__ import annotations

import logging
import re
import json
from dataclasses import dataclass
from typing import Optional, List

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, RunContext, ModelRetry

from .model_handlers.model_configs import ModelConfigurations
from .mcp_registry import MCPRegistry
from .simple_mcp_executor import SimpleMCPExecutor


# ----------------------------
# Schema
# ----------------------------
class SearchResult(BaseModel):
    summary: str = Field(
        description="Concise answer text with proper Telegram formatting"
    )
    sources: List[str] = Field(default_factory=list, description="Cited source URLs")

    # Pydantic v2 configuration
    model_config = {
        "extra": "ignore",  # ignore hallucinated fields
        "str_strip_whitespace": True,  # trim string fields
    }


# ----------------------------
# Deps
# ----------------------------
@dataclass
class AgentDeps:
    user_id: Optional[int] = None
    username: Optional[str] = None
    preferred_model: Optional[str] = None


# ----------------------------
# Utilities
# ----------------------------
def sanitize_text(text: str, max_len: int = 10000) -> str:
    """
    Sanitize arbitrary text coming from tools or models to reduce JSON-parse hazards.
    - Remove BOM and nulls
    - Strip code fences
    - Trim length
    """
    if text is None:
        return ""

    # Remove problematic control characters
    cleaned = text.replace("\x00", "").replace("\ufeff", "")

    # Strip common code-fence wrappers
    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        # Remove starting ```json / ``` and trailing ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Trim overlong outputs
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "... (truncated)"

    return cleaned


def attempt_extract_json_block(text: str) -> str:
    """
    Try to carve out the first well-formed JSON object-ish region from text.
    This is a lightweight repair step for cases where the model adds commentary.
    """
    text = sanitize_text(text)

    # If it already looks like a JSON object, return as-is
    if text.startswith("{") and text.endswith("}"):
        return text

    # Attempt to extract the largest {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text  # fallback: return original sanitized text


# ----------------------------
# Agent Wrapper
# ----------------------------
class EnhancedAgent:
    """Wraps a Pydantic AI Agent with dynamic MCP tools registration and robust JSON handling."""

    def __init__(
        self,
        registry: Optional[MCPRegistry] = None,
        preferred_model: Optional[str] = None,
    ) -> None:
        self.registry = registry or MCPRegistry()
        self.mcp_executor = SimpleMCPExecutor(self.registry)
        self.logger = logging.getLogger(__name__)

        # Resolve model
        all_models = ModelConfigurations.get_all_models()
        model_key = preferred_model if preferred_model in all_models else "gemini-flash"
        cfg = all_models[model_key]
        model_id = ModelConfigurations.resolve_api_model_id(cfg.model_id)

        # System prompt
        base_prompt = (
            getattr(cfg, "system_message", None)
            or "You are a concise research assistant."
        ).rstrip()

        # We keep Telegram markdown in the summary field, but require the model to output JSON only.
        system_prompt = (
            f"{base_prompt} "
            "Always cite 2-5 sources. "
            "Format responses for Telegram in the 'summary' field using *bold*, _italic_, `code`, [links](url), and ```code blocks```. "
            "Respond ONLY with a single valid JSON object matching this schema: "
            '{ "summary": string, "sources": array of URL strings }. '
            "No commentary, no code fences, no trailing text."
        )

        self.agent = Agent(
            model_id,
            deps_type=AgentDeps,
            output_type=SearchResult,
            system_prompt=system_prompt,
        )

        self.logger.info(f"Research agent initialized with model: {model_id}")
        self._register_dynamic_tools()

    # ------------------------
    # Tool registration
    # ------------------------
    def _register_dynamic_tools(self) -> None:
        async def exa_search_web_search_exa(
            ctx: RunContext[AgentDeps], query: str
        ) -> str:
            """Search the web using Exa AI."""
            try:
                result = await self.mcp_executor.execute_exa_search(query)
                return sanitize_text(result)
            except Exception as e:
                self.logger.error(f"Exa search failed: {e}")
                return f"Tool execution encountered an issue: {sanitize_text(str(e))}"

        async def exa_search_company_research_exa(
            ctx: RunContext[AgentDeps], company_name: str
        ) -> str:
            """Research a company using Exa AI."""
            try:
                result = await self.mcp_executor.execute_exa_company_research(
                    company_name
                )
                return sanitize_text(result)
            except Exception as e:
                self.logger.error(f"Exa company research failed: {e}")
                return f"Tool execution encountered an issue: {sanitize_text(str(e))}"

        async def exa_search_crawling_exa(ctx: RunContext[AgentDeps], url: str) -> str:
            """Crawl and extract content from a URL using Exa AI."""
            try:
                result = await self.mcp_executor.execute_exa_crawl(url)
                return sanitize_text(result)
            except Exception as e:
                self.logger.error(f"Exa URL crawling failed: {e}")
                return f"Tool execution encountered an issue: {sanitize_text(str(e))}"

        async def context7_search(ctx: RunContext[AgentDeps], query: str) -> str:
            """Search library documentation using Context7."""
            try:
                result = await self.mcp_executor.execute_context7_search(query)
                return sanitize_text(result)
            except Exception as e:
                self.logger.error(f"Context7 search failed: {e}")
                return f"Tool execution encountered an issue: {sanitize_text(str(e))}"

        async def duckduckgo_search(ctx: RunContext[AgentDeps], query: str) -> str:
            """Search using DuckDuckGo."""
            try:
                result = await self.mcp_executor.execute_duckduckgo_search(query)
                return sanitize_text(result)
            except Exception as e:
                self.logger.error(f"DuckDuckGo search failed: {e}")
                return f"Tool execution encountered an issue: {sanitize_text(str(e))}"

        # Register tools
        self.agent.tool(exa_search_web_search_exa)
        self.agent.tool(exa_search_company_research_exa)
        self.agent.tool(exa_search_crawling_exa)
        self.agent.tool(context7_search)
        self.agent.tool(duckduckgo_search)

    # ------------------------
    # Core run with resilience
    # ------------------------
    async def run(self, query: str, deps: Optional[AgentDeps] = None) -> SearchResult:
        # Input validation
        if not query or not query.strip():
            return SearchResult(
                summary="❌ *Empty Query* - Please provide a search query.",
                sources=[],
            )

        # First attempt
        try:
            self.logger.info(f"Running agent query: {query[:100]}...")
            result = await self.agent.run(query, deps=deps)
            output = result.output

            # Enforce minimal citations expectation
            if isinstance(output, SearchResult) and len(output.sources) < 2:
                # Soft retry: ask for proper sourcing
                self.logger.warning(
                    "Output had insufficient sources; requesting reformat with sources."
                )
                reformatted = await self._retry_with_reformat(
                    query,
                    deps,
                    reason="Please include at least 2 source URLs in sources[].",
                )
                if reformatted:
                    return reformatted

            self.logger.info(
                f"Agent query completed successfully with {len(output.sources)} sources"
            )
            return output

        # JSON / schema errors: try a repair pass
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in agent run: {e}")
            repaired = await self._retry_with_repair(
                query, deps, f"JSON parse error: {e}"
            )
            if repaired:
                return repaired
            return SearchResult(
                summary=(
                    "❌ *Response Processing Error* - The AI service returned an invalid response format.\n\n"
                    f"*Details:* `{sanitize_text(str(e))[:200]}`\n\n"
                    "*Suggestion:* Try rephrasing your query or try again later."
                ),
                sources=[],
            )

        except ValidationError as e:
            self.logger.error(f"Pydantic validation error in agent run: {e}")
            repaired = await self._retry_with_repair(
                query, deps, f"Pydantic validation error: {e}"
            )
            if repaired:
                return repaired
            return SearchResult(
                summary=(
                    "⚠️ *Parsing Issue* - The response didn't match the expected schema.\n\n"
                    f"*Details:* `{sanitize_text(str(e))[:200]}`\n\n"
                    "*Suggestion:* Try a more specific query."
                ),
                sources=[],
            )

        except ModelRetry as e:
            # If a tool or validator explicitly requested a retry, do a guided re-run.
            self.logger.warning(f"Model retry triggered: {e}")
            retried = await self._retry_with_reformat(query, deps, str(e))
            if retried:
                return retried
            return SearchResult(
                summary=(
                    "⚠️ *Processing Issue* - The AI needs more information to complete your request.\n\n"
                    "*Suggestion:* Please provide more specific details in your query."
                ),
                sources=[],
            )

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Agent run failed: {error_msg}")

            # Friendly, user-facing fallbacks
            lower = error_msg.lower()
            if "429" in error_msg or "rate limit" in lower:
                return SearchResult(
                    summary=(
                        "🔄 *Rate limit reached*. The service is temporarily busy. Please try again in a few minutes.\n\n"
                        "*Alternative:* Use `/search` for a lightweight web search or try a different query."
                    ),
                    sources=[],
                )
            if "networktimeout" in error_msg or "connection" in lower:
                return SearchResult(
                    summary=(
                        "🔌 *Connection issue*. An upstream service is temporarily unavailable. Please try again later.\n\n"
                        "*Status:* Service may be under maintenance."
                    ),
                    sources=[],
                )
            if "timeout" in lower:
                return SearchResult(
                    summary=(
                        "⏰ *Request timeout*. The search took too long to complete. Please try a more specific query.\n\n"
                        "*Suggestion:* Use shorter or more focused terms."
                    ),
                    sources=[],
                )

            # Default fallback
            return SearchResult(
                summary=f"❌ *Search failed*: `{sanitize_text(error_msg)[:200]}`\n\n*Suggestion:* Try rephrasing your query or use a different search method.",
                sources=[],
            )

    # ------------------------
    # Retry helpers
    # ------------------------
    async def _retry_with_repair(
        self, query: str, deps: Optional[AgentDeps], reason: str
    ) -> Optional[SearchResult]:
        """
        Second-chance call that explicitly asks the model for clean JSON.
        """
        self.logger.info(f"Retrying with repair hint due to: {reason}")
        repair_hint = (
            "\n\nIMPORTANT: Output ONLY a valid JSON object matching "
            '{ "summary": string, "sources": array of URL strings }. '
            "No code fences, no commentary, no extra text."
        )
        try:
            result = await self.agent.run(query + repair_hint, deps=deps)
            output = result.output
            if isinstance(output, SearchResult):
                # Ensure sources threshold if possible
                if len(output.sources) >= 2:
                    self.logger.info("Repair retry succeeded with sufficient sources.")
                    return output
                self.logger.info(
                    "Repair retry returned but sources still insufficient; accepting result."
                )
                return output
        except Exception as e:
            self.logger.error(f"Repair retry failed: {e}")
        return None

    async def _retry_with_reformat(
        self, query: str, deps: Optional[AgentDeps], message: str
    ) -> Optional[SearchResult]:
        """
        Ask the model to reformat its reply to match the schema and include sources.
        """
        self.logger.info(f"Retrying with reformat request: {message}")
        reformat_hint = (
            "\n\nIMPORTANT: Reformat your reply as a single valid JSON object with keys "
            '"summary" (string) and "sources" (array of URL strings). Include at least 2 sources. '
            "No code fences, no commentary."
        )
        try:
            result = await self.agent.run(query + reformat_hint, deps=deps)
            output = result.output
            if isinstance(output, SearchResult):
                self.logger.info(
                    f"Reformat retry succeeded with {len(output.sources)} sources."
                )
                return output
        except Exception as e:
            self.logger.error(f"Reformat retry failed: {e}")
        return None


__all__ = ["EnhancedAgent", "AgentDeps", "SearchResult"]
