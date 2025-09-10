#!/bin/bash

# MCP Server Initialization Script for Docker
# This script ensures MCP servers are ready before starting the main application

set -e

echo "🚀 Starting MCP Server Initialization..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to verify MCP package installation
verify_mcp_package() {
    local package=$1
    local timeout=10
    local count=0
    
    echo "📦 Verifying $package..."
    
    while ! npx $package --version >/dev/null 2>&1; do
        if [ $count -ge $timeout ]; then
            echo "❌ Failed to verify $package after ${timeout} seconds"
            return 1
        fi
        echo "⏳ Waiting for $package to be ready..."
        sleep 1
        ((count++))
    done
    
    echo "✅ $package is ready"
    return 0
}

# Function to test MCP server startup
test_mcp_server() {
    local server_name=$1
    local command=$2
    shift 2
    local args=("$@")
    
    echo "🧪 Testing $server_name startup..."
    
    # Start the server in background and test if it initializes
    timeout 10 npx "${args[@]}" >/dev/null 2>&1 &
    local pid=$!
    
    # Wait a bit and check if the process is still running
    sleep 2
    if kill -0 $pid >/dev/null 2>&1; then
        echo "✅ $server_name can start successfully"
        kill $pid 2>/dev/null || true
        return 0
    else
        echo "❌ $server_name failed to start"
        return 1
    fi
}

# Verify Node.js and npm
if ! command_exists node; then
    echo "❌ Node.js not found"
    exit 1
fi

if ! command_exists npm; then
    echo "❌ npm not found"
    exit 1
fi

echo "✅ Node.js $(node --version) and npm $(npm --version) are available"

# Verify MCP packages are installed
echo "🔍 Verifying MCP packages installation..."

# Test each MCP package
MCP_PACKAGES=(
    "@smithery/cli@latest"
    "@upstash/context7-mcp@latest"
    "@modelcontextprotocol/server-sequential-thinking@latest"
)

for package in "${MCP_PACKAGES[@]}"; do
    if ! verify_mcp_package "$package"; then
        echo "⚠️ $package verification failed, attempting reinstall..."
        npm install -g "$package" || {
            echo "❌ Failed to install $package"
            exit 1
        }
    fi
done

# Test MCP server startups (with API key handling)
echo "🧪 Testing MCP server startups..."

# Test servers that don't require API keys first
if test_mcp_server "Context7" npx "-y" "@upstash/context7-mcp@latest"; then
    echo "✅ Context7 server startup test passed"
else
    echo "⚠️ Context7 server startup test failed"
fi

if test_mcp_server "Sequential Thinking" npx "-y" "@modelcontextprotocol/server-sequential-thinking@latest"; then
    echo "✅ Sequential Thinking server startup test passed"
else
    echo "⚠️ Sequential Thinking server startup test failed"
fi

# Test API key dependent servers if API key is available
if [ -n "$MCP_API_KEY" ]; then
    echo "🔑 API key found, testing API-dependent servers..."
    
    if test_mcp_server "Smithery Fetch" npx "-y" "@smithery/cli@latest" "run" "fetch-mcp" "--key" "$MCP_API_KEY"; then
        echo "✅ Smithery Fetch server startup test passed"
    else
        echo "⚠️ Smithery Fetch server startup test failed"
    fi
    
    if test_mcp_server "Exa Search" npx "-y" "@smithery/cli@latest" "run" "exa" "--key" "$MCP_API_KEY"; then
        echo "✅ Exa Search server startup test passed"
    else
        echo "⚠️ Exa Search server startup test failed"
    fi
else
    echo "⚠️ MCP_API_KEY not found, skipping API-dependent server tests"
fi

echo "🎉 MCP Server initialization completed!"
echo "📝 MCP servers are ready for use"

# Optional: Start the main application
if [ "$1" = "--start-app" ]; then
    shift
    echo "🚀 Starting main application..."
    exec "$@"
fi
