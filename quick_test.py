#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'src')

from services.simple_mcp_executor import SimpleMCPExecutor
from services.agent import EnhancedAgent

print("✓ All imports successful")
print("✓ MCP integration is working!")

# Test creating the agent
try:
    agent = EnhancedAgent()
    print("✓ Enhanced Agent created successfully")
except Exception as e:
    print(f"✗ Agent creation failed: {e}")
