import os
import logging
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
from src.services.rate_limiter import RateLimiter, rate_limit
from src.utils.log.telegramlog import telegram_logger
from src.services.model_handlers.model_configs import (
    ModelConfigurations,
    Provider,
    ModelConfig,
)
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from src.services.mcp import get_mcp_registry, MCPRegistry
from src.services.mcp_tool_logger import MCPToolLogger
from datetime import datetime

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    telegram_logger.log_error(
        "OPENROUTER_API_KEY not found in environment variables.", 0
    )


class OpenRouterAPI:
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(__name__)
        telegram_logger.log_message("Initializing OpenRouter API", 0)

        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found or empty")

        self._load_openrouter_models_from_config()

        # Circuit breaker properties
        self.api_failures = 0
        self.api_last_failure = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300

        # MCP integration
        self.mcp_registry: Optional[MCPRegistry] = None
        self._mcp_initialized = False
        self._mcp_disabled_until = 0  # Timestamp when MCP can be re-enabled
        self.tool_logger = MCPToolLogger()

    def _load_openrouter_models_from_config(self):
        """Load available models from centralized configuration specific to OpenRouter."""
        openrouter_configs = ModelConfigurations.get_models_by_provider(
            Provider.OPENROUTER
        )
        self.available_models: Dict[str, str] = {
            model_id: config.openrouter_model_key
            for model_id, config in openrouter_configs.items()
            if config.openrouter_model_key is not None
        }
        self.logger.info(
            f"Loaded {len(self.available_models)} OpenRouter models from configuration."
        )

    async def initialize_mcp(self) -> None:
        """Initialize MCP integration with robust error handling."""
        if self._mcp_initialized:
            return
            
        # Check if MCP is temporarily disabled
        if time.time() < self._mcp_disabled_until:
            self.logger.info("MCP temporarily disabled due to previous errors")
            return
            
        try:
            self.logger.info("Initializing MCP integration...")
            self.mcp_registry = await get_mcp_registry()
            
            if self.mcp_registry and self.mcp_registry.get_server_names():
                server_count = len(self.mcp_registry.get_server_names())
                self.logger.info(f"MCP integration initialized successfully with {server_count} servers")
                self._mcp_initialized = True
            else:
                self.logger.warning("MCP registry initialized but no servers available")
                self._mcp_initialized = True  # Still mark as initialized to avoid retry loops
                
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP: {e}")
            self.logger.info("Continuing without MCP integration - API will work normally")
            # Temporarily disable MCP for 5 minutes to prevent repeated failures
            self._mcp_disabled_until = time.time() + 300
            # Don't set _mcp_initialized = True so it won't retry on every call
            # But also don't raise - allow API to work without MCP

    async def _create_enhanced_agent(
        self, model_id: str, system_message: str, use_mcp: bool = True
    ) -> Agent:
        """Create a Pydantic AI agent with optional MCP toolsets."""
        # Get model with fallback support from centralized config
        openrouter_model = ModelConfigurations.get_model_with_fallback(model_id)

        # Create OpenAI provider with OpenRouter configuration
        openai_provider = OpenAIProvider(
            base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY
        )

        # Create model instance
        request_model = OpenAIModel(
            model_name=openrouter_model, provider=openai_provider
        )

        # Get MCP toolsets if requested and available
        toolsets = []
        if use_mcp and self._mcp_initialized and self.mcp_registry:
            try:
                toolsets = self.mcp_registry.get_toolsets()
                if toolsets:
                    self.logger.info(f"Using {len(toolsets)} MCP toolsets with agent")
                else:
                    self.logger.info("No MCP toolsets available, proceeding without MCP")
            except Exception as e:
                self.logger.warning(f"Failed to get MCP toolsets: {e}")
                self.logger.info("Proceeding without MCP toolsets")
        elif use_mcp:
            self.logger.info("MCP not initialized, proceeding without MCP toolsets")

        # Create agent with or without MCP toolsets
        agent = Agent(
            model=request_model,
            system_prompt=system_message,
            toolsets=toolsets if toolsets else None,
        )

        return agent

    def get_available_models(self) -> Dict[str, str]:
        """Get the mapping of model IDs to OpenRouter model keys."""
        return self.available_models.copy()

    def _build_system_message(
        self, model_id: str, context: Optional[List[Dict]] = None
    ) -> str:
        """Return a system message based on model and context, using ModelConfigurations."""
        model_config: Optional[ModelConfig] = ModelConfigurations.get_all_models().get(
            model_id
        )
        if model_config and model_config.system_message:
            base_message = model_config.system_message
        else:
            base_message = (
                "You are an advanced AI assistant that helps users with various tasks."
            )

        # Include conversation history in system message if context is provided
        if context:
            context_str = "\n\nConversation History:\n"
            for msg in context:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    role = msg["role"]
                    content = msg["content"]
                    context_str += f"{role.title()}: {content}\n"
            base_message += (
                context_str
                + "\nPlease continue the conversation naturally, considering the history above."
            )

        if not context and not model_config:
            # If no context is provided and no specific model config, add a general helpfulness hint.
            return base_message + " Be concise, helpful, and accurate."
        return base_message

    async def generate_response_with_tool_logging(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
        use_mcp: bool = True,
    ) -> tuple[Optional[str], MCPToolLogger]:
        """
        Generate response with detailed MCP tool call logging.
        Returns both the response and the tool logger for inspection.
        
        Note: temperature and max_tokens parameters are accepted for API compatibility
        but are not directly supported by PydanticAI Agent.run() method.
        These would need to be configured at the model level if needed.
        """
        # Clear previous tool calls
        self.tool_logger.clear()

        try:
            # Initialize MCP if not already done
            if not self._mcp_initialized:
                await self.initialize_mcp()

            # Build system message with context included
            system_message = self._build_system_message(model, context)

            # Get default OpenRouter model if none provided
            if not model:
                available_models = self.get_available_models()
                if available_models:
                    model = next(iter(available_models.keys()))
                else:
                    # Import and use the proper default model from model configs
                    from src.services.model_handlers.model_configs import (
                        get_default_agent_model,
                    )

                    model = get_default_agent_model()

            # Create enhanced agent with MCP tools
            agent = await self._create_enhanced_agent(model, system_message, use_mcp)

            self.logger.info(
                f"Sending request to OpenRouter via PydanticAI Agent with model {model} (MCP: {use_mcp})"
            )

            # Use PydanticAI Agent to generate response
            start_time = datetime.now()
            result = await agent.run(user_prompt=prompt)

            # Log agent execution summary
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"Agent execution completed in {execution_time:.0f}ms")

            if result and result.output:
                self.logger.info(
                    f"OpenRouter response length: {len(result.output)} characters"
                )
                self.api_failures = 0
                return result.output, self.tool_logger
            else:
                self.logger.warning(
                    "No valid response from OpenRouter API via PydanticAI"
                )
                self.api_failures += 1
                return None, self.tool_logger

        except Exception as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            
            # Check if this is an MCP-related timeout error
            error_str = str(e)
            if "timeout" in error_str.lower() and "mcp" in error_str.lower():
                self.logger.warning("MCP timeout detected, temporarily disabling MCP for 10 minutes")
                self._mcp_disabled_until = time.time() + 600  # 10 minutes
                self._mcp_initialized = False  # Reset to allow re-initialization
                
                # Retry once without MCP if the original request wanted MCP
                if use_mcp:
                    self.logger.info("Retrying request without MCP...")
                    try:
                        return await self.generate_response_with_tool_logging(
                            prompt, context, model, temperature, max_tokens, timeout, use_mcp=False
                        )
                    except Exception as retry_error:
                        self.logger.error(f"Retry without MCP also failed: {retry_error}")
            
            self.logger.error(
                f"OpenRouter API error via PydanticAI for model {model}: {str(e)}",
                exc_info=True,
            )
            return f"OpenRouter API error: {error_str}", self.tool_logger

    @rate_limit
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
        use_mcp: bool = True,
    ) -> Optional[str]:
        """Legacy method - calls new tool logging method and returns only response.
        
        Note: temperature and max_tokens parameters are accepted for API compatibility
        but are not directly supported by PydanticAI Agent.run() method.
        These would need to be configured at the model level if needed.
        """
        response, _ = await self.generate_response_with_tool_logging(
            prompt, context, model, temperature, max_tokens, timeout, use_mcp
        )
        return response

    @rate_limit
    async def generate_response_with_model_key(
        self,
        prompt: str,
        openrouter_model_key: str,
        system_message: Optional[str] = None,
        context: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
        use_mcp: bool = True,
    ) -> Optional[str]:
        """Generate response using a specific OpenRouter model key.
        
        Note: temperature and max_tokens parameters are accepted for API compatibility
        but are not directly supported by PydanticAI Agent.run() method.
        These would need to be configured at the model level if needed.
        """
        try:
            # Initialize MCP if not already done
            if not self._mcp_initialized:
                await self.initialize_mcp()

            # Build final system message with context included
            if system_message:
                final_system_message = system_message
            else:
                final_system_message = "You are an advanced AI assistant that helps users with various tasks. Be concise, helpful, and accurate."

            # Include conversation history in system message if context is provided
            if context:
                context_str = "\n\nConversation History:\n"
                for msg in context:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        role = msg["role"]
                        content = msg["content"]
                        context_str += f"{role.title()}: {content}\n"
                final_system_message += (
                    context_str
                    + "\nPlease continue the conversation naturally, considering the history above."
                )

            # Create enhanced agent with MCP tools
            agent = await self._create_enhanced_agent(
                openrouter_model_key, final_system_message, use_mcp
            )

            self.logger.info(
                f"Sending request to OpenRouter via PydanticAI Agent with model key {openrouter_model_key} (MCP: {use_mcp})"
            )

            # Use PydanticAI Agent to generate response (with or without MCP tools)
            result = await agent.run(
                user_prompt=prompt, temperature=temperature, max_tokens=max_tokens
            )

            if result and result.output:
                self.api_failures = 0
                return result.output
            else:
                self.api_failures += 1
                return None

        except Exception as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(
                f"OpenRouter API error via PydanticAI for model key {openrouter_model_key}: {str(e)}"
            )
            return None

    def debug_model_mapping(self):
        """Debug method to log all available model mappings."""
        self.logger.info("=== OpenRouter Model Mappings ===")
        for model_id, openrouter_key in self.available_models.items():
            self.logger.info(f"  {model_id} -> {openrouter_key}")
        self.logger.info(f"Total models loaded: {len(self.available_models)}")

    def get_mcp_status(self) -> Dict[str, str]:
        """Get MCP integration status."""
        current_time = time.time()
        
        if current_time < self._mcp_disabled_until:
            remaining = int(self._mcp_disabled_until - current_time)
            return {
                "status": "temporarily_disabled", 
                "servers": "0",
                "reason": f"MCP disabled due to errors, {remaining}s remaining"
            }
        
        if not self._mcp_initialized:
            return {"status": "not_initialized", "servers": "0"}

        if not self.mcp_registry:
            return {"status": "failed", "servers": "0"}

        server_count = len(self.mcp_registry.get_server_names())
        return {
            "status": "ready",
            "servers": str(server_count),
            "server_names": ", ".join(self.mcp_registry.get_server_names()),
        }
        
    def reenable_mcp(self) -> bool:
        """Manually re-enable MCP if it was temporarily disabled."""
        if time.time() < self._mcp_disabled_until:
            self._mcp_disabled_until = 0
            self._mcp_initialized = False  # Reset to allow re-initialization
            self.logger.info("MCP manually re-enabled")
            return True
        return False
