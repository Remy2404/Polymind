#!/usr/bin/env python3
"""
Test script to check available MCP tools
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

async def test_mcp_tools():
    """Test MCP tool availability and list actual tool names."""
    try:
        from src.services.mcp import get_mcp_registry
        
        print("🔧 Testing MCP Registry...")
        registry = await get_mcp_registry()
        
        print(f"✅ Registry initialized with {len(registry.get_server_names())} servers")
        
        for server_name in registry.get_server_names():
            print(f"\n📡 Server: {server_name}")
            server = registry.get_server(server_name)
            if server:
                try:
                    # Get server capabilities using Pydantic AI MCP
                    print(f"   Server instance: {type(server).__name__}")
                    print(f"   Command: {registry.get_server_config(server_name).command}")
                    print(f"   Args: {registry.get_server_config(server_name).args}")
                except Exception as e:
                    print(f"   ❌ Error getting server info: {e}")
            else:
                print(f"   ❌ Server not available")
                
        # Test with actual Pydantic AI agent
        print("\n🤖 Testing with Pydantic AI Agent...")
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider
        
        # Create OpenRouter provider
        openrouter_provider = OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        # Create model
        model = OpenAIModel(
            model_name="qwen/qwen3-235b-a22b:free",
            provider=openrouter_provider
        )
        
        # Get toolsets
        toolsets = registry.get_toolsets()
        print(f"📦 Available toolsets: {len(toolsets)}")
        
        # Create agent with tools
        agent = Agent(
            model=model,
            system_prompt="You are a helpful assistant with access to MCP tools.",
            toolsets=toolsets if toolsets else None
        )
        
        print("✅ Agent created successfully")
        
        # Test a simple query that should trigger tool use
        print("\n🔍 Testing tool invocation...")
        result = await agent.run("List the available tools you have access to")
        print(f"Response: {result.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())
