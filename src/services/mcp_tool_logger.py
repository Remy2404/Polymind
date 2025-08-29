"""
Enhanced MCP Tool Call Logger
Captures and formats MCP tool invocations for display in Telegram.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolCall:
    """Represents a single MCP tool call."""
    tool_name: str
    server_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = None
    duration_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MCPToolLogger:
    """
    Enhanced logger for MCP tool calls that provides user-friendly 
    formatting for Telegram display.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tool_calls: List[ToolCall] = []
        
    def log_tool_call(self, tool_name: str, server_name: str, 
                     arguments: Dict[str, Any]) -> ToolCall:
        """Log the start of a tool call."""
        tool_call = ToolCall(
            tool_name=tool_name,
            server_name=server_name,
            arguments=arguments,
            timestamp=datetime.now()
        )
        
        self.tool_calls.append(tool_call)
        
        # Log to system logger
        self.logger.info(
            f"MCP Tool Call Started - {server_name}:{tool_name} "
            f"with args: {json.dumps(arguments, default=str)[:200]}..."
        )
        
        return tool_call
        
    def log_tool_result(self, tool_call: ToolCall, result: Any, 
                       duration_ms: Optional[float] = None) -> None:
        """Log the successful result of a tool call."""
        tool_call.result = result
        tool_call.duration_ms = duration_ms
        
        # Log to system logger
        result_preview = str(result)[:200] if result else "None"
        self.logger.info(
            f"MCP Tool Call Completed - {tool_call.server_name}:{tool_call.tool_name} "
            f"in {duration_ms}ms, result: {result_preview}..."
        )
        
    def log_tool_error(self, tool_call: ToolCall, error: str, 
                      duration_ms: Optional[float] = None) -> None:
        """Log an error from a tool call."""
        tool_call.error = error
        tool_call.duration_ms = duration_ms
        
        # Log to system logger
        self.logger.error(
            f"MCP Tool Call Failed - {tool_call.server_name}:{tool_call.tool_name} "
            f"in {duration_ms}ms, error: {error}"
        )
        
    def format_tool_calls_for_telegram(self, show_details: bool = False) -> str:
        """
        Format tool calls for display in Telegram.
        
        Args:
            show_details: Whether to show detailed arguments and results
        """
        if not self.tool_calls:
            return ""
            
        message = "🔧 **Tool Calls:**\n"
        
        for i, call in enumerate(self.tool_calls, 1):
            # Status icon
            if call.error:
                status_icon = "❌"
                status = "Failed"
            elif call.result is not None:
                status_icon = "✅"
                status = "Success"
            else:
                status_icon = "🔄"
                status = "Running"
                
            # Basic info
            message += f"{status_icon} `{call.server_name}:{call.tool_name}` - {status}"
            
            if call.duration_ms:
                message += f" ({call.duration_ms:.0f}ms)"
                
            message += "\n"
            
            # Show details if requested
            if show_details:
                if call.arguments:
                    args_str = self._format_arguments(call.arguments)
                    message += f"   📝 Args: {args_str}\n"
                    
                if call.result:
                    result_str = self._format_result(call.result)
                    message += f"   📊 Result: {result_str}\n"
                    
                if call.error:
                    message += f"   ⚠️ Error: {call.error[:100]}...\n"
                    
                message += "\n"
                
        return message
        
    def _format_arguments(self, arguments: Dict[str, Any]) -> str:
        """Format tool arguments for display."""
        if not arguments:
            return "None"
            
        # Format common argument types
        formatted_args = []
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > 50:
                formatted_args.append(f"{key}={value[:50]}...")
            elif isinstance(value, (list, dict)):
                formatted_args.append(f"{key}={type(value).__name__}({len(value)})")
            else:
                formatted_args.append(f"{key}={value}")
                
        return ", ".join(formatted_args)
        
    def _format_result(self, result: Any) -> str:
        """Format tool result for display."""
        if result is None:
            return "None"
            
        if isinstance(result, str):
            if len(result) > 100:
                return f"{result[:100]}..."
            return result
            
        if isinstance(result, (list, dict)):
            return f"{type(result).__name__}({len(result)} items)"
            
        if isinstance(result, (int, float, bool)):
            return str(result)
            
        # For other types, show type and truncated string representation
        result_str = str(result)
        if len(result_str) > 100:
            return f"{type(result).__name__}: {result_str[:100]}..."
        return f"{type(result).__name__}: {result_str}"
        
    def get_tool_call_summary(self) -> Dict[str, Any]:
        """Get a summary of tool calls for logging."""
        total_calls = len(self.tool_calls)
        successful_calls = len([c for c in self.tool_calls if c.result is not None and not c.error])
        failed_calls = len([c for c in self.tool_calls if c.error])
        
        servers_used = list(set(c.server_name for c in self.tool_calls))
        tools_used = list(set(f"{c.server_name}:{c.tool_name}" for c in self.tool_calls))
        
        total_duration = sum(c.duration_ms for c in self.tool_calls if c.duration_ms)
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "servers_used": servers_used,
            "tools_used": tools_used,
            "total_duration_ms": total_duration
        }
        
    def clear(self) -> None:
        """Clear logged tool calls."""
        self.tool_calls.clear()
        
    def get_telegram_progress_message(self) -> str:
        """Get a progress message for ongoing tool calls."""
        running_calls = [c for c in self.tool_calls if c.result is None and not c.error]
        
        if not running_calls:
            return ""
            
        message = "🔄 **Processing...**\n"
        for call in running_calls:
            elapsed = (datetime.now() - call.timestamp).total_seconds() * 1000
            message += f"• `{call.server_name}:{call.tool_name}` ({elapsed:.0f}ms)\n"
            
        return message
