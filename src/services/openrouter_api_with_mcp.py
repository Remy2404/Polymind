"""
Enhanced OpenRouter API with MCP Tool Integration
This module extends the base OpenRouter API to include automatic MCP server
integration, enabling LLMs to use MCP tools seamlessly.
"""
import logging
from typing import Dict, List, Optional, Any
from src.services.openrouter_api import OpenRouterAPI
from src.services.mcp import MCPManager
from src.services.model_handlers.model_configs import ModelConfigurations, Provider
from src.utils.log.telegramlog import telegram_logger
from src.services.gemini_api import GeminiAPI
class OpenRouterAPIWithMCP(OpenRouterAPI):
    """
    Enhanced OpenRouter API with MCP (Model Context Protocol) integration.
    This class extends the base OpenRouterAPI to automatically load and use
    MCP server tools, converting them to OpenAI-compatible format.
    """
    def __init__(self, rate_limiter, mcp_config_path: str = "mcp.json"):
        """
        Initialize the enhanced OpenRouter API with MCP support and Gemini integration.
        Args:
            rate_limiter: Rate limiter instance
            mcp_config_path: Path to MCP configuration file
        """
        super().__init__(rate_limiter)
        self.mcp_manager = MCPManager(mcp_config_path)
        self.mcp_tools_loaded = False
        self.logger = logging.getLogger(__name__)
        self._tool_unsupported_models = set()
        try:
            self.gemini_api = GeminiAPI(rate_limiter, mcp_config_path)
            self.logger.info("Gemini API initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini API: {e}")
            self.gemini_api = None
    def _get_model_provider(self, model: str) -> Provider:
        """
        Determine which provider to use for a given model.
        Args:
            model: Model identifier
        Returns:
            Provider enum indicating which API to use
        """
        model_config = ModelConfigurations.get_all_models().get(model)
        if model_config:
            return model_config.provider
        return Provider.OPENROUTER
    def _should_use_gemini(self, model: str) -> bool:
        """
        Check if we should use Gemini for the given model.
        Args:
            model: Model identifier
        Returns:
            True if Gemini should be used, False otherwise
        """
        if not self.gemini_api:
            return False
        provider = self._get_model_provider(model)
        return provider == Provider.GEMINI
    async def initialize_mcp_tools(self) -> bool:
        """
        Initialize and load MCP tools from configured servers.
        Returns:
            True if MCP tools were loaded successfully
        """
        try:
            self.logger.info("Initializing MCP tools...")
            telegram_logger.log_message("Initializing MCP tools...", 0)
            success = await self.mcp_manager.load_servers()
            if success:
                self.mcp_tools_loaded = True
                server_info = self.mcp_manager.get_server_info()
                self.logger.info(f"MCP tools initialized successfully: {server_info}")
                telegram_logger.log_message(
                    f"MCP tools initialized: {len(server_info)} servers", 0
                )
                return True
            else:
                self.logger.warning("Failed to initialize MCP tools")
                telegram_logger.log_message("Failed to initialize MCP tools", 0)
                return False
        except Exception as e:
            self.logger.error(f"Error initializing MCP tools: {str(e)}")
            telegram_logger.log_error(f"Error initializing MCP tools: {str(e)}", 0)
            return False
    async def generate_response_with_mcp_tools(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate a response using OpenRouter with MCP tools available.
        Args:
            prompt: The user prompt
            context: Conversation context
            model: Optional model override (if None, uses default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
        Returns:
            Generated response or None if failed
        """
        if not self.mcp_tools_loaded:
            await self.initialize_mcp_tools()
        if self._should_use_gemini(model):
            self.logger.info(f"Using Gemini API for model {model}")
            return await self._generate_gemini_with_mcp_tools(
                prompt=prompt,
                context=context,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            self.logger.info(f"Using OpenRouter API for model {model}")
            actual_model = model if model is not None else "gemini"
        if model:
            model_config = ModelConfigurations.get_all_models().get(model)
            if model_config:
                if model_config.provider != Provider.OPENROUTER:
                    if model_config.openrouter_model_key:
                        actual_model = model_config.openrouter_model_key
                        self.logger.info(
                            f"Using OpenRouter equivalent {actual_model} for model {model}"
                        )
                    else:
                        self.logger.warning(
                            f"Model {model} has no OpenRouter equivalent, cannot use MCP tools"
                        )
                        return None
                else:
                    if model_config.openrouter_model_key:
                        actual_model = model_config.openrouter_model_key
        mcp_tools = (
            await self.mcp_manager.get_all_tools() if self.mcp_tools_loaded else []
        )
        if mcp_tools:
            if actual_model in self._tool_unsupported_models:
                self.logger.info(
                    f"Model {actual_model} is known to not support tools, using standard generation"
                )
                return await self.generate_response(
                    prompt=prompt,
                    context=context,
                    model=actual_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            self.logger.info(
                f"Using {len(mcp_tools)} MCP tools for generation with model {actual_model}"
            )
            return await self.generate_response_with_tools(
                prompt=prompt,
                tools=mcp_tools,
                context=context,
                model=actual_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            self.logger.info(
                f"No MCP tools available, using standard generation with model {actual_model}"
            )
            return await self.generate_response(
                prompt=prompt,
                context=context,
                model=actual_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
    def _build_system_message(
        self,
        model_id: str,
        context: Optional[List[Dict]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Return a system message based on model and context, with dynamic tool usage instructions."""
        model_config = ModelConfigurations.get_all_models().get(model_id)
        if model_config and model_config.system_message:
            base_message = model_config.system_message
        else:
            base_message = (
                "You are an advanced AI assistant that helps users with various tasks."
            )
        context_hint = (
            " Use conversation history/context when relevant." if context else ""
        )
        tool_instructions = ""
        if tools:
            tool_categories = self._categorize_tools(tools)
            tool_names = [tool["function"]["name"] for tool in tools]
            tool_instructions = f"""
You have access to the following tools: {', '.join(tool_names)}
- **Documentation & Code Examples**: Use documentation tools when users ask about libraries, frameworks, APIs, or need code examples
- **Search & Research**: Use search tools for finding information, current data, or web content
- **Analysis & Processing**: Use specialized tools for data analysis, file processing, or complex computations
- **External Services**: Use tools that connect to external services or APIs
1. **Identify the Right Tool**: Choose the most appropriate tool based on the user's request
2. **Provide Complete Arguments**: Ensure all required parameters are included in your tool calls
3. **Handle Results**: Use the tool results to provide comprehensive, accurate responses
4. **Combine Tools**: Use multiple tools in parallel when possible to provide comprehensive answers. Call all relevant tools in one response to gather complete information.
{chr(10).join([f"- **{category}**: {', '.join(category_tools)}" for category, category_tools in tool_categories.items()])}
- Always use tools when they can provide more accurate or current information
- When multiple tools are relevant, use them together in one response for thorough analysis
- Provide detailed, helpful responses based on tool results
- If a tool fails, try alternative approaches or inform the user
- Do not mention tool internal details or <think> tags in your final response
Focus on providing the most helpful and accurate response possible using the available tools."""
        return base_message + context_hint + tool_instructions
    def _categorize_tools(self, tools: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Categorize tools by their functionality for better organization.
        Args:
            tools: List of tool definitions
        Returns:
            Dictionary mapping categories to tool names
        """
        categories = {
            "Documentation": [],
            "Search & Research": [],
            "Development": [],
            "Analysis": [],
            "Communication": [],
            "Other": [],
        }
        for tool in tools:
            tool_name = tool["function"]["name"].lower()
            description = tool["function"].get("description", "").lower()
            if any(
                keyword in tool_name or keyword in description
                for keyword in [
                    "doc",
                    "docs",
                    "documentation",
                    "library",
                    "api",
                    "guide",
                    "tutorial",
                    "reference",
                ]
            ):
                categories["Documentation"].append(tool["function"]["name"])
            elif any(
                keyword in tool_name or keyword in description
                for keyword in [
                    "search",
                    "find",
                    "query",
                    "lookup",
                    "research",
                    "web",
                    "browse",
                ]
            ):
                categories["Search & Research"].append(tool["function"]["name"])
            elif any(
                keyword in tool_name or keyword in description
                for keyword in [
                    "code",
                    "dev",
                    "build",
                    "compile",
                    "test",
                    "debug",
                    "git",
                ]
            ):
                categories["Development"].append(tool["function"]["name"])
            elif any(
                keyword in tool_name or keyword in description
                for keyword in [
                    "analyze",
                    "process",
                    "calculate",
                    "data",
                    "metrics",
                    "stats",
                ]
            ):
                categories["Analysis"].append(tool["function"]["name"])
            elif any(
                keyword in tool_name or keyword in description
                for keyword in ["chat", "message", "email", "notify", "communication"]
            ):
                categories["Communication"].append(tool["function"]["name"])
            else:
                categories["Other"].append(tool["function"]["name"])
        return {k: v for k, v in categories.items() if v}
    async def generate_response_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate a response with tool calling support.
        Args:
            prompt: The user prompt
            tools: List of tools in OpenAI format
            context: Conversation context
            model: Optional model to use (if None, uses default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
        Returns:
            Generated response or None if failed
        """
        try:
            openrouter_model = self.available_models.get(model, model)
            system_message = self._build_system_message(model, context, tools)
            messages = []
            # Use the new capability detection system
            safe_config = ModelConfigurations.get_safe_model_config(openrouter_model)
            if safe_config["use_system_message"]:
                messages.append({"role": "system", "content": system_message})
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})

            # Adapt the request parameters based on model capabilities
            request_params = {
                "model": openrouter_model,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            adapted_params = ModelConfigurations.adapt_request_for_model(openrouter_model, request_params)

            response = await self.client.chat.completions.create(**adapted_params)
            if (
                hasattr(response.choices[0].message, "tool_calls")
                and response.choices[0].message.tool_calls
            ):
                tool_calls = response.choices[0].message.tool_calls
                messages.append(response.choices[0].message.model_dump())
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    try:
                        if isinstance(tool_args, str):
                            import json
                            tool_args = json.loads(tool_args)
                            self.logger.info(
                                f"Parsed tool arguments for {tool_name}: {tool_args}"
                            )
                        self.logger.info(
                            f"Executing MCP tool: {tool_name} with args: {tool_args}"
                        )
                        tool_result = await self.mcp_manager.execute_tool(
                            tool_name, tool_args
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(tool_result),
                            }
                        )
                        self.logger.info(f"Successfully executed MCP tool: {tool_name}")
                    except json.JSONDecodeError as json_error:
                        self.logger.error(
                            f"Failed to parse tool arguments for {tool_name}: {str(json_error)}"
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error parsing tool arguments: {str(json_error)}",
                            }
                        )
                    except Exception as e:
                        self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
                        self.logger.error(f"Tool arguments were: {tool_args}")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error executing tool: {str(e)}",
                            }
                        )
                # Adapt the request parameters for the final response call
                final_request_params = {
                    "model": openrouter_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                }
                final_adapted_params = ModelConfigurations.adapt_request_for_model(openrouter_model, final_request_params)

                final_response = await self.client.chat.completions.create(**final_adapted_params)
                cleaned_content = self._clean_response_content(
                    final_response.choices[0].message.content
                )
                return cleaned_content
            else:
                cleaned_content = self._clean_response_content(
                    response.choices[0].message.content
                )
                return cleaned_content
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Error in generate_response_with_tools: {error_str}")
            should_fallback = any(
                phrase in error_str.lower()
                for phrase in [
                    "no endpoints found that support tool use",
                    "does not support tool calling",
                    "tool use not supported",
                    "tool calling not supported",
                    "404",
                ]
            )
            if should_fallback:
                self.logger.info(
                    f"Tool calling failed for model {openrouter_model}, attempting fallback without tools..."
                )
                telegram_logger.log_message(
                    f"Falling back to non-tool mode for {model}", 0
                )
                self._tool_unsupported_models.add(openrouter_model)
                self.logger.info(f"Added {openrouter_model} to tool-unsupported cache")
                try:
                    fallback_response = await self.generate_response(
                        prompt=prompt,
                        context=context,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                    if fallback_response:
                        self.logger.info(
                            f"Successfully generated fallback response for model {openrouter_model}"
                        )
                        return fallback_response
                    else:
                        self.logger.error(
                            f"Fallback generation also failed for model {openrouter_model}"
                        )
                        return "I apologize, but I'm having trouble processing your request right now. Please try again or use a different model."
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback generation failed: {str(fallback_error)}"
                    )
                    return "I apologize, but I'm having trouble processing your request right now. Please try again or use a different model."
            if "not a valid model ID" in error_str:
                self.logger.warning(
                    f"Model '{openrouter_model}' is not a valid OpenRouter model ID"
                )
            elif "400" in error_str:
                self.logger.warning(
                    f"Bad request error for model '{openrouter_model}' - likely tool calling not supported"
                )
            return "I encountered an error while processing your request. Please try again or use a different model."
    def _clean_response_content(self, content: str) -> str:
        """
        Clean response content by removing thinking tags and tool calls.
        Args:
            content: Raw response content from the model
        Returns:
            Cleaned content suitable for user display
        """
        if not content:
            return content
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
        content = re.sub(r"<[^>]+>.*?</[^>]+>", "", content, flags=re.DOTALL)
        content = content.strip()
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
        return content
    async def get_available_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available MCP tools in OpenAI format.
        Returns:
            List of available MCP tools
        """
        if not self.mcp_tools_loaded:
            await self.initialize_mcp_tools()
        if self.mcp_tools_loaded:
            return await self.mcp_manager.get_all_tools()
        else:
            return []
    def get_mcp_server_info(self) -> Dict[str, Any]:
        """
        Get information about connected MCP servers.
        Returns:
            Dictionary with server information
        """
        if self.mcp_tools_loaded:
            return self.mcp_manager.get_server_info()
        else:
            return {}
    async def close(self):
        """Close the API client and MCP connections."""
        await super().close()
        if self.mcp_tools_loaded:
            await self.mcp_manager.disconnect_all()
        if self.gemini_api:
            await self.gemini_api.close()
    async def _generate_gemini_with_mcp_tools(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate a response using Gemini with MCP tools available.
        Args:
            prompt: The user prompt
            context: Conversation context
            model: Optional model override
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
        Returns:
            Generated response or None if failed
        """
        if not self.gemini_api:
            self.logger.error("Gemini API not available")
            return None
        actual_model = model
        if model == "gemini":
            actual_model = "gemini-2.5-flash"
            self.logger.debug(f"Translated model alias '{model}' to '{actual_model}'")
        return await self.gemini_api.generate_response_with_mcp_tools(
            prompt=prompt,
            context=context,
            model=actual_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
