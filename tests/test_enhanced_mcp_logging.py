"""
Test script for Enhanced MCP Tool Call Logging
Tests the new MCP integration with tool call visibility.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.mcp_tool_logger import MCPToolLogger, ToolCall
from src.handlers.mcp_response_formatter import MCPResponseFormatter
from datetime import datetime


def test_tool_logger():
    """Test the MCPToolLogger functionality."""
    print("🧪 Testing MCPToolLogger...")
    
    # Create logger
    logger = MCPToolLogger()
    
    # Simulate tool calls
    call1 = logger.log_tool_call(
        "exa_search_web_search_exa", 
        "Exa Search", 
        {"query": "current time in Phnom Penh", "max_results": 5}
    )
    
    # Simulate successful result
    logger.log_tool_result(call1, "SearchResult with 5 entries", 1234.5)
    
    call2 = logger.log_tool_call(
        "resolve-library-id",
        "Context7",
        {"library_name": "FastAPI", "topic": "middleware"}
    )
    
    # Simulate error
    logger.log_tool_error(call2, "Library not found", 567.8)
    
    # Test formatting
    formatted = logger.format_tool_calls_for_telegram(show_details=True)
    print("📋 Formatted Tool Calls:")
    print(formatted)
    
    # Test summary
    summary = logger.get_tool_call_summary()
    print("📊 Tool Call Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("✅ MCPToolLogger test completed\n")


def test_response_formatter():
    """Test the MCPResponseFormatter functionality."""
    print("🎨 Testing MCPResponseFormatter...")
    
    # Create a logger with test data
    logger = MCPToolLogger()
    call = logger.log_tool_call(
        "exa_search_web_search_exa", 
        "Exa Search",
        {"query": "test query"}
    )
    logger.log_tool_result(call, "Test result data", 1000.0)
    
    # Test processing message
    processing_msg = MCPResponseFormatter.format_processing_message(
        "exa_search_web_search_exa", 
        "What is the current time in Phnom Penh?",
        "Exa Search"
    )
    print("⏳ Processing Message:")
    print(processing_msg)
    print()
    
    # Test complete response
    complete_response = MCPResponseFormatter.format_complete_response(
        "The current time in Phnom Penh is 1:25 AM ICT...",
        logger,
        "exa_search_web_search_exa",
        "Exa Search"
    )
    print("✅ Complete Response:")
    print(complete_response)
    print()
    
    # Test error response
    error_response = MCPResponseFormatter.format_error_response(
        "exa_search_web_search_exa",
        "Connection timeout",
        logger,
        "Exa Search"
    )
    print("❌ Error Response:")
    print(error_response)
    print()
    
    # Test status formatting
    servers_info = {
        "exa-search": {
            "server_name": "exa-search",
            "command": "npx",
            "args": ["-y", "@smithery/cli@latest", "run", "exa"],
            "tool_prefix": None,
            "status": "initialized"
        },
        "context7": {
            "server_name": "context7",
            "status": "disabled"
        },
        "failed-server": {
            "server_name": "failed-server",
            "status": "not_initialized"
        }
    }
    
    status_response = MCPResponseFormatter.format_mcp_status(servers_info, True)
    print("📊 MCP Status:")
    print(status_response)
    
    print("✅ MCPResponseFormatter test completed\n")


def test_integration():
    """Test the integration between components."""
    print("🔗 Testing Integration...")
    
    # Simulate a complete MCP workflow
    logger = MCPToolLogger()
    
    # Tool execution
    call = logger.log_tool_call(
        "exa_search_web_search_exa",
        "Exa Search", 
        {"query": "latest AI developments", "max_results": 10}
    )
    
    # Simulate some processing time
    import time
    time.sleep(0.1)  # Simulate 100ms processing
    
    logger.log_tool_result(
        call, 
        "Found 10 search results with recent AI developments",
        100.5
    )
    
    # Generate response
    ai_response = (
        "Based on my search, here are the latest AI developments:\n\n"
        "1. **Large Language Models**: Continued improvements in reasoning\n"
        "2. **Multimodal AI**: Better integration of text, image, and voice\n"
        "3. **AI Agents**: More sophisticated tool usage and planning\n\n"
        "The search found comprehensive information from recent sources."
    )
    
    # Format complete response
    final_response = MCPResponseFormatter.format_complete_response(
        ai_response,
        logger,
        "exa_search_web_search_exa",
        "Exa Search"
    )
    
    # Check length (Telegram limit simulation)
    final_response = MCPResponseFormatter.truncate_for_telegram(final_response)
    
    print("🎉 Final Integrated Response:")
    print("=" * 60)
    print(final_response)
    print("=" * 60)
    
    print(f"📏 Response length: {len(final_response)} characters")
    print("✅ Integration test completed\n")


def main():
    """Run all tests."""
    print("🚀 Starting Enhanced MCP Tool Logging Tests")
    print("=" * 50)
    
    try:
        test_tool_logger()
        test_response_formatter()
        test_integration()
        
        print("🎉 All tests completed successfully!")
        print("\n💡 The enhanced MCP tool logging system is ready for use!")
        print("\nTo test in Telegram bot:")
        print("  /search latest AI developments")
        print("  /Context7 FastAPI middleware")
        print("  /mcp_status")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
