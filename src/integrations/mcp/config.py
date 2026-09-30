import logging
import os
from typing import Any, Dict
from src.utils.log.telegramlog import telegram_logger

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Substitute environment variables in MCP configuration.
    Handles $VAR_NAME syntax in configuration values.
    Args:
        config: MCP configuration dictionary
    Returns:
        Configuration with environment variables substituted
    """

    def substitute_value(value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith("$"):
                env_var = value[1:]
                env_value = os.getenv(env_var)
                if env_value is None:
                    logging.warning(
                        f"Environment variable '{env_var}' not found, using original value"
                    )
                    return value
                return env_value
            return value
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value

    return substitute_value(config)


def validate_mcp_environment() -> bool:
    """
    Validate that required MCP environment variables are available.
    Returns:
        True if all required variables are present
    """
    required_vars = ["MCP_API_KEY"]
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    if missing_vars:
        logging.error(
            f"Missing required MCP environment variables: {', '.join(missing_vars)}"
        )
        telegram_logger.log_error(
            f"Missing required MCP environment variables: {', '.join(missing_vars)}", 0
        )
        return False
    logging.info("All required MCP environment variables are present")
    return True
