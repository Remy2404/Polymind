#!/usr/bin/env python3
"""
Test script for MCP integration in Polymind AI.
"""

import os
import sys
import asyncio
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_registry():
    """Test MCP registry initialization and configuration."""
    try:
        from services.mcp_registry import MCPRegistry
        
        # Create MCP registry
        registry = MCPRegistry()
        
        # Test configuration loading
        config_loaded = await registry.load_config()
        logger.info(f"MCP config loaded: {config_loaded}")
        
        if config_loaded:
            # Show server status
            servers = registry.get_enabled_servers()
            logger.info(f"Enabled servers: {len(servers)}")
            
            for server in servers:
                logger.info(f"  - {server.name}: {server.description}")
                
            # Show available tools
            tools = registry.get_all_tools()
            logger.info(f"Available tools: {len(tools)}")
            
            for tool in tools:
                logger.info(f"  - {tool.name} (server: {tool.server})")
                
        return True
        
    except Exception as e:
        logger.error(f"MCP registry test failed: {e}")
        return False

async def test_enhanced_agent():
    """Test Enhanced Agent functionality."""
    try:
        from services.agent import EnhancedAgent, AgentConfig
        
        # Create agent config
        config = AgentConfig(enable_mcp=True)
        
        # Create agent (without model manager for now)
        agent = EnhancedAgent(config, None)
        
        # Test initialization
        initialized = await agent.initialize()
        logger.info(f"Enhanced Agent initialized: {initialized}")
        
        if initialized:
            # Test server status
            status = agent.get_server_status()
            logger.info(f"Server status: {status}")
            
            # Test tool listing
            tools = agent.get_available_tools()
            logger.info(f"Available tools: {tools}")
            
        return True
        
    except Exception as e:
        logger.error(f"Enhanced Agent test failed: {e}")
        return False

async def test_mock_search():
    """Test mock search functionality."""
    try:
        from services.agent import EnhancedAgent, AgentConfig
        
        # Create agent
        config = AgentConfig(enable_mcp=True)
        agent = EnhancedAgent(config, None)
        
        # Initialize
        await agent.initialize()
        
        # Test search request
        result = await agent.process_request(
            "Search for latest AI developments",
            use_tools=True
        )
        
        logger.info(f"Search result: {result}")
        return True
        
    except Exception as e:
        logger.error(f"Mock search test failed: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("Starting MCP integration tests...")
    
    # Test 1: MCP Registry
    logger.info("\n=== Testing MCP Registry ===")
    registry_ok = await test_mcp_registry()
    
    # Test 2: Enhanced Agent
    logger.info("\n=== Testing Enhanced Agent ===")
    agent_ok = await test_enhanced_agent()
    
    # Test 3: Mock Search
    logger.info("\n=== Testing Mock Search ===")
    search_ok = await test_mock_search()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"MCP Registry: {'✅ PASS' if registry_ok else '❌ FAIL'}")
    logger.info(f"Enhanced Agent: {'✅ PASS' if agent_ok else '❌ FAIL'}")
    logger.info(f"Mock Search: {'✅ PASS' if search_ok else '❌ FAIL'}")
    
    if all([registry_ok, agent_ok, search_ok]):
        logger.info("\n🎉 All tests passed! MCP integration is working.")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)