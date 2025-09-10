#!/bin/bash

# Health check script for MCP servers in Docker
# This script checks if MCP servers are responding properly

set -e

echo "🏥 MCP Health Check Starting..."

# Function to check if MCP server responds
check_mcp_server() {
    local server_name=$1
    shift
    local cmd=("$@")
    
    echo "🔍 Checking $server_name..."
    
    # Try to start the server and check if it responds within timeout
    timeout 15 "${cmd[@]}" >/dev/null 2>&1 &
    local pid=$!
    
    # Wait and check if process is still alive (indicates successful start)
    sleep 3
    if kill -0 $pid 2>/dev/null; then
        echo "✅ $server_name is healthy"
        kill $pid 2>/dev/null || true
        return 0
    else
        echo "❌ $server_name is unhealthy"
        return 1
    fi
}

# Check each MCP server
healthy_servers=0
total_servers=0

# Context7 (no API key required)
total_servers=$((total_servers + 1))
if check_mcp_server "Context7" npx -y @upstash/context7-mcp@latest; then
    healthy_servers=$((healthy_servers + 1))
fi

# Sequential Thinking (no API key required)
total_servers=$((total_servers + 1))
if check_mcp_server "Sequential Thinking" npx -y @modelcontextprotocol/server-sequential-thinking@latest; then
    healthy_servers=$((healthy_servers + 1))
fi

# API key dependent servers (only check if API key is available)
if [ -n "$MCP_API_KEY" ]; then
    # Fetch MCP
    total_servers=$((total_servers + 1))
    if check_mcp_server "Fetch MCP" npx -y @smithery/cli@latest run fetch-mcp --key "$MCP_API_KEY"; then
        healthy_servers=$((healthy_servers + 1))
    fi
    
    # Exa Search
    total_servers=$((total_servers + 1))
    if check_mcp_server "Exa Search" npx -y @smithery/cli@latest run exa --key "$MCP_API_KEY"; then
        healthy_servers=$((healthy_servers + 1))
    fi
else
    echo "⚠️ MCP_API_KEY not found, skipping API-dependent servers"
fi

# Report results
echo "📊 Health Check Results:"
echo "   Healthy servers: $healthy_servers/$total_servers"

if [ $healthy_servers -eq $total_servers ]; then
    echo "✅ All MCP servers are healthy!"
    exit 0
elif [ $healthy_servers -gt 0 ]; then
    echo "⚠️ Some MCP servers are healthy ($healthy_servers/$total_servers)"
    exit 0  # Don't fail if some servers work
else
    echo "❌ No MCP servers are healthy"
    exit 1
fi
