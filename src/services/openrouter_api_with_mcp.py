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


class OpenRouterAPIWithMCP(OpenRouterAPI):
    """
    Enhanced OpenRouter API with MCP (Model Context Protocol) integration.

    This class extends the base OpenRouterAPI to automatically load and use
    MCP server tools, converting them to OpenAI-compatible format.
    """

    def __init__(self, rate_limiter, mcp_config_path: str = "mcp.json"):
        """
        Initialize the enhanced OpenRouter API with MCP support.

        Args:
            rate_limiter: Rate limiter instance
            mcp_config_path: Path to MCP configuration file
        """
        super().__init__(rate_limiter)
        self.mcp_manager = MCPManager(mcp_config_path)
        self.mcp_tools_loaded = False
        self.logger = logging.getLogger(__name__)

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
                telegram_logger.log_message(f"MCP tools initialized: {len(server_info)} servers", 0)
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
        # Ensure MCP tools are loaded
        if not self.mcp_tools_loaded:
            await self.initialize_mcp_tools()

        # Use provided model or default
        actual_model = model if model is not None else "gemini"
        
        # Validate model compatibility with OpenRouter
        if model:
            model_config = ModelConfigurations.get_all_models().get(model)
            if model_config:
                # If it's a non-OpenRouter model, check for OpenRouter equivalent
                if model_config.provider != Provider.OPENROUTER:
                    if model_config.openrouter_model_key:
                        actual_model = model_config.openrouter_model_key
                        self.logger.info(f"Using OpenRouter equivalent {actual_model} for model {model}")
                    else:
                        self.logger.warning(f"Model {model} has no OpenRouter equivalent, cannot use MCP tools")
                        return None
                else:
                    # It's already an OpenRouter model, use the openrouter_model_key if available
                    if model_config.openrouter_model_key:
                        actual_model = model_config.openrouter_model_key

        # Get available MCP tools in OpenAI format
        mcp_tools = await self.mcp_manager.get_all_tools() if self.mcp_tools_loaded else []

        if mcp_tools:
            self.logger.info(f"Using {len(mcp_tools)} MCP tools for generation with model {actual_model}")
            return await self.generate_response_with_tools(
                prompt=prompt,
                tools=mcp_tools,
                context=context,
                model=actual_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
        else:
            self.logger.info(f"No MCP tools available, using standard generation with model {actual_model}")
            return await self.generate_response(
                prompt=prompt,
                context=context,
                model=actual_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )

    def _build_system_message(
        self, model_id: str, context: Optional[List[Dict]] = None, tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Return a system message based on model and context, with dynamic tool usage instructions."""
        model_config = ModelConfigurations.get_all_models().get(model_id)
        if model_config and model_config.system_message:
            base_message = model_config.system_message
        else:
            base_message = "You are an advanced AI assistant that helps users with various tasks."

        context_hint = " Use conversation history/context when relevant." if context else ""

        # Add dynamic tool usage instructions if tools are available
        tool_instructions = ""
        if tools:
            # Group tools by their functionality for better organization
            tool_categories = self._categorize_tools(tools)
            tool_names = [tool["function"]["name"] for tool in tools]

            tool_instructions = f"""

You have access to the following tools: {', '.join(tool_names)}

## Tool Usage Guidelines:

### When to Use Tools:
- **Documentation & Code Examples**: Use documentation tools when users ask about libraries, frameworks, APIs, or need code examples
- **Search & Research**: Use search tools for finding information, current data, or web content
- **Analysis & Processing**: Use specialized tools for data analysis, file processing, or complex computations
- **External Services**: Use tools that connect to external services or APIs

### How to Use Tools Effectively:
1. **Identify the Right Tool**: Choose the most appropriate tool based on the user's request
2. **Provide Complete Arguments**: Ensure all required parameters are included in your tool calls
3. **Handle Results**: Use the tool results to provide comprehensive, accurate responses
4. **Combine Tools**: Use multiple tools when needed to provide complete answers

### Available Tool Categories:
{chr(10).join([f"- **{category}**: {', '.join(category_tools)}" for category, category_tools in tool_categories.items()])}

### Important Notes:
- Always use tools when they can provide more accurate or current information
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
            "Other": []
        }

        for tool in tools:
            tool_name = tool["function"]["name"].lower()
            description = tool["function"].get("description", "").lower()

            # Categorize based on tool name and description
            if any(keyword in tool_name or keyword in description for keyword in
                   ["doc", "docs", "documentation", "library", "api", "guide", "tutorial", "reference"]):
                categories["Documentation"].append(tool["function"]["name"])
            elif any(keyword in tool_name or keyword in description for keyword in
                    ["search", "find", "query", "lookup", "research", "web", "browse"]):
                categories["Search & Research"].append(tool["function"]["name"])
            elif any(keyword in tool_name or keyword in description for keyword in
                    ["code", "dev", "build", "compile", "test", "debug", "git"]):
                categories["Development"].append(tool["function"]["name"])
            elif any(keyword in tool_name or keyword in description for keyword in
                    ["analyze", "process", "calculate", "data", "metrics", "stats"]):
                categories["Analysis"].append(tool["function"]["name"])
            elif any(keyword in tool_name or keyword in description for keyword in
                    ["chat", "message", "email", "notify", "communication"]):
                categories["Communication"].append(tool["function"]["name"])
            else:
                categories["Other"].append(tool["function"]["name"])

        # Remove empty categories
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
            # Get model with fallback support
            openrouter_model = self.available_models.get(model, model)

            system_message = self._build_system_message(model, context, tools)
            messages = [{"role": "system", "content": system_message}]

            if context:
                messages.extend(context)

            messages.append({"role": "user", "content": prompt})

            # Make initial request with tools
            response = await self.client.chat.completions.create(
                model=openrouter_model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            # Handle tool calls if present
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls

                # Add assistant's message with tool calls to context
                messages.append(response.choices[0].message.model_dump())

                # Execute tools and add results
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    try:
                        # Parse arguments if they're a string
                        if isinstance(tool_args, str):
                            import json
                            tool_args = json.loads(tool_args)
                            self.logger.info(f"Parsed tool arguments for {tool_name}: {tool_args}")

                        # Execute the tool
                        self.logger.info(f"Executing MCP tool: {tool_name} with args: {tool_args}")
                        tool_result = await self.mcp_manager.execute_tool(tool_name, tool_args)

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_result)
                        })

                        self.logger.info(f"Successfully executed MCP tool: {tool_name}")

                    except json.JSONDecodeError as json_error:
                        self.logger.error(f"Failed to parse tool arguments for {tool_name}: {str(json_error)}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error parsing tool arguments: {str(json_error)}"
                        })
                    except Exception as e:
                        self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
                        self.logger.error(f"Tool arguments were: {tool_args}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error executing tool: {str(e)}"
                        })

                # Make final request with tool results
                final_response = await self.client.chat.completions.create(
                    model=openrouter_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )

                # Clean the response content to remove thinking tags and tool calls
                cleaned_content = self._clean_response_content(final_response.choices[0].message.content)
                return cleaned_content

            else:
                # No tool calls, return the direct response
                cleaned_content = self._clean_response_content(response.choices[0].message.content)
                return cleaned_content

        except Exception as e:
            self.logger.error(f"Error in generate_response_with_tools: {str(e)}")
            
            # Check if it's a model compatibility error and provide more specific logging
            if "not a valid model ID" in str(e):
                self.logger.warning(f"Model '{openrouter_model}' is not a valid OpenRouter model ID")
            elif "400" in str(e):
                self.logger.warning(f"Bad request error for model '{openrouter_model}' - likely tool calling not supported")
            elif "does not support tool calling" in str(e):
                self.logger.warning(f"Model '{openrouter_model}' does not support tool calling")
            
            return f"Error generating response with tools: {str(e)}"

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

        # Remove thinking tags (common in reasoning models like DeepSeek)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # Remove tool call tags
        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)

        # Remove any remaining XML-like tags that might be in the response
        content = re.sub(r'<[^>]+>.*?</[^>]+>', '', content, flags=re.DOTALL)

        # Clean up extra whitespace
        content = content.strip()

        # Remove multiple consecutive newlines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

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
