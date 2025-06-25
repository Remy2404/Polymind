# Setup Instructions

## Installation

1. Install dependencies:
```bash
uv sync
```

2. Verify installation:
```bash
python -c "print('✅ Setup complete!')"
```

3. For MCP Server integration (optional):
```bash
# Install Node.js dependencies for MCP server
npm install -g @smithery/cli
```

## Bot Configuration

The bot is now optimized for low-resource servers and uses fast regex-based intent detection.
No additional language models or heavy dependencies are required.

## MCP Server Integration

The bot supports Model Context Protocol (MCP) servers to enhance AI model capabilities:

1. Set up MCP-enabled models in your `.env`:
```
OPENROUTER_API_KEY=your_openrouter_api_key
EXA_API_KEY=your_exa_api_key  # Optional for enhanced Exa search capability
```

2. Use MCP-enabled models like:
- `claude-3-opus-mcp-exa` (Claude 3 Opus with Exa search)
- `claude-3-sonnet-mcp-exa` (Claude 3 Sonnet with Exa search)

3. For more information, see: `docs/MCP_SERVER_INTEGRATION_GUIDE.md`
