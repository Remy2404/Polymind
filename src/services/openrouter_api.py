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
        """Initialize MCP integration."""
        try:
            if self._mcp_initialized:
                return

            self.mcp_registry = await get_mcp_registry()
            self._mcp_initialized = True
            self.logger.info("MCP integration initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP: {e}")
            # Don't raise - allow API to work without MCP

    async def _create_enhanced_agent(self, model_id: str, system_message: str, 
                                   use_mcp: bool = True) -> Agent:
        """Create a Pydantic AI agent with optional MCP toolsets."""
        # Get model with fallback support from centralized config
        openrouter_model = ModelConfigurations.get_model_with_fallback(model_id)

        # Create OpenAI provider with OpenRouter configuration
        openai_provider = OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )

        # Create model instance
        request_model = OpenAIModel(
            model_name=openrouter_model,
            provider=openai_provider
        )

        # Get MCP toolsets if requested and available
        toolsets = []
        if use_mcp and self._mcp_initialized and self.mcp_registry:
            try:
                toolsets = self.mcp_registry.get_toolsets()
                if toolsets:
                    self.logger.info(f"Using {len(toolsets)} MCP toolsets with agent")
            except Exception as e:
                self.logger.warning(f"Failed to get MCP toolsets: {e}")

        # Create agent with or without MCP toolsets
        agent = Agent(
            model=request_model,
            system_prompt=system_message,
            toolsets=toolsets if toolsets else None
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
            base_message += context_str + "\nPlease continue the conversation naturally, considering the history above."

        if not context and not model_config:
            # If no context is provided and no specific model config, add a general helpfulness hint.
            return base_message + " Be concise, helpful, and accurate."
        return base_message

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
        try:
            # Initialize MCP if not already done
            if not self._mcp_initialized:
                await self.initialize_mcp()

            # Build system message with context included
            system_message = self._build_system_message(model, context)

            # Get default OpenRouter model if none provided
            if not model:
                # Get the first available OpenRouter model as fallback
                available_models = self.get_available_models()
                if available_models:
                    model = next(iter(available_models.keys()))
                else:
                    # Last resort fallback
                    model = "qwen3-235b"

            # Create enhanced agent with MCP tools
            agent = await self._create_enhanced_agent(model, system_message, use_mcp)

            self.logger.info(
                f"Sending request to OpenRouter via PydanticAI Agent with model {model} (MCP: {use_mcp})"
            )

            # Use PydanticAI Agent to generate response (with or without MCP tools)
            result = await agent.run(
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if result and result.output:
                self.logger.info(
                    f"OpenRouter response length: {len(result.output)} characters"
                )
                self.api_failures = 0  # Reset failures on successful response
                return result.output
            else:
                self.logger.warning("No valid response from OpenRouter API via PydanticAI")
                self.api_failures += 1
                return None

        except Exception as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(
                f"OpenRouter API error via PydanticAI for model {model}: {str(e)}", exc_info=True
            )
            return f"OpenRouter API error: {str(e)}"

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
                final_system_message += context_str + "\nPlease continue the conversation naturally, considering the history above."

            # Create enhanced agent with MCP tools
            agent = await self._create_enhanced_agent(openrouter_model_key, final_system_message, use_mcp)

            self.logger.info(
                f"Sending request to OpenRouter via PydanticAI Agent with model key {openrouter_model_key} (MCP: {use_mcp})"
            )

            # Use PydanticAI Agent to generate response (with or without MCP tools)
            result = await agent.run(
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
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
        if not self._mcp_initialized:
            return {"status": "not_initialized", "servers": "0"}
        
        if not self.mcp_registry:
            return {"status": "failed", "servers": "0"}
            
        server_count = len(self.mcp_registry.get_server_names())
        return {
            "status": "ready", 
            "servers": str(server_count),
            "server_names": ", ".join(self.mcp_registry.get_server_names())
        }
