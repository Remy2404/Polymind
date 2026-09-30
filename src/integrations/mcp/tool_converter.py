from typing import Any, Dict, List, Optional
from mcp.types import Tool as MCPTool


class MCPToolConverter:
    """Converts MCP tool definitions to OpenAI-compatible format with structured schemas."""

    @staticmethod
    def convert_mcp_tool_to_openai(mcp_tool: MCPTool) -> Dict[str, Any]:
        """
        Convert an MCP tool definition to OpenAI tool format with enhanced schemas.
        Args:
            mcp_tool: MCP tool definition
        Returns:
            OpenAI-compatible tool definition with structured schemas
        """
        openai_tool = {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or f"Execute {mcp_tool.name} tool",
                "parameters": MCPToolConverter._convert_input_schema(
                    mcp_tool.inputSchema
                ),
            },
        }

        # Add output schema if available for better validation
        if hasattr(mcp_tool, "outputSchema") and mcp_tool.outputSchema:
            openai_tool["function"]["output_schema"] = (
                MCPToolConverter._convert_output_schema(mcp_tool.outputSchema)
            )

        # Add additional metadata for robustness
        openai_tool["function"][
            "strict"
        ] = True  # Enable strict mode for better validation

        return openai_tool

    @staticmethod
    def _convert_input_schema(input_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert MCP input schema to OpenAI format with validation.
        """
        if not input_schema:
            return {"type": "object", "properties": {}, "required": []}

        # Ensure required fields
        schema = {
            "type": input_schema.get("type", "object"),
            "properties": input_schema.get("properties", {}),
            "required": input_schema.get("required", []),
        }

        return schema

    @staticmethod
    def _convert_output_schema(
        output_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Convert MCP output schema to structured format.
        """
        if not output_schema:
            return {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Tool execution result",
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the operation succeeded",
                    },
                },
            }

        # Ensure the output schema is well-structured
        schema = dict(output_schema)

        # Add standard fields for consistency
        if schema.get("type") == "object" and "properties" in schema:
            properties = schema["properties"]
            # Ensure success indicator
            if "success" not in properties:
                properties["success"] = {
                    "type": "boolean",
                    "description": "Operation success status",
                }
            # Ensure error field for failed operations
            if "error" not in properties:
                properties["error"] = {
                    "type": "string",
                    "description": "Error message if operation failed",
                }

        return schema

    @staticmethod
    def convert_mcp_tools_to_openai(mcp_tools: List[MCPTool]) -> List[Dict[str, Any]]:
        """
        Convert multiple MCP tools to OpenAI format.
        Args:
            mcp_tools: List of MCP tool definitions
        Returns:
            List of OpenAI-compatible tool definitions
        """
        return [MCPToolConverter.convert_mcp_tool_to_openai(tool) for tool in mcp_tools]

    @staticmethod
    def convert_mcp_tool_to_meta(mcp_tool: MCPTool) -> Dict[str, Any]:
        """
        Convert an MCP tool definition to Meta-compatible format.
        Meta models require explicit 'items' property for array types.

        Args:
            mcp_tool: MCP tool definition
        Returns:
            Meta-compatible tool definition
        """

        def add_items_to_arrays(schema: Any) -> Any:
            """Recursively ensure all arrays have 'items' property."""
            if not isinstance(schema, dict):
                return schema

            result = {}
            for key, value in schema.items():
                if key == "type" and value == "array":
                    # Ensure parent dict has 'items' property
                    result[key] = value
                    if "items" not in schema:
                        # Add default items if not present
                        result["items"] = {"type": "string"}
                elif isinstance(value, dict):
                    result[key] = add_items_to_arrays(value)
                elif isinstance(value, list):
                    result[key] = [
                        add_items_to_arrays(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    result[key] = value

            # If this schema defines an array, ensure it has items
            if result.get("type") == "array" and "items" not in result:
                result["items"] = {"type": "string"}

            return result

        # Convert base tool structure
        meta_tool = {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or f"Execute {mcp_tool.name} tool",
                "parameters": add_items_to_arrays(
                    MCPToolConverter._convert_input_schema(mcp_tool.inputSchema)
                ),
            },
        }

        # Add output schema if available
        if hasattr(mcp_tool, "outputSchema") and mcp_tool.outputSchema:
            meta_tool["function"]["output_schema"] = add_items_to_arrays(
                MCPToolConverter._convert_output_schema(mcp_tool.outputSchema)
            )

        return meta_tool

    @staticmethod
    def convert_mcp_tools_to_meta(mcp_tools: List[MCPTool]) -> List[Dict[str, Any]]:
        """
        Convert multiple MCP tools to Meta-compatible format.

        Args:
            mcp_tools: List of MCP tool definitions
        Returns:
            List of Meta-compatible tool definitions
        """
        return [MCPToolConverter.convert_mcp_tool_to_meta(tool) for tool in mcp_tools]
