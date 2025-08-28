# MCP Integration Guide

## Overview

The Polymind AI project now includes comprehensive Model Context Protocol (MCP) integration, providing enhanced search, research, and tool capabilities via Smithery.ai and other MCP servers.

## Features

### 🔍 Enhanced Search Capabilities
- **Web Search**: Real-time web search via Exa AI
- **Company Research**: Business intelligence and company analysis
- **URL Crawling**: Content extraction from web pages
- **Multi-server Support**: Extensible MCP server architecture

### 🤖 AI Agent Orchestration
- **Enhanced Agent**: Combines 54+ AI models with MCP tools
- **Intelligent Tool Selection**: Automatically selects appropriate tools based on queries
- **Context-aware Processing**: Integrates tool results with AI model responses
- **Fallback Mechanisms**: Graceful degradation when MCP services are unavailable

### 👥 Group Collaboration
- **Collaborative Research**: Shared search results in group chats
- **Group Preferences**: Configurable MCP settings per group
- **Search History**: Track and share research within groups
- **Auto-suggestions**: Smart MCP command suggestions

## Quick Start

### 1. Environment Setup

Create a `.env` file with required variables:

```bash
# Copy from .env.example
cp .env.example .env

# Edit with your API keys
SMITHERY_API_KEY=your_smithery_api_key_here
EXA_PROFILE=your_exa_profile_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
# ... other variables
```

### 2. MCP Configuration

The `mcp.json` file configures available MCP servers:

```json
{
  "servers": {
    "exa-search": {
      "type": "streamable_http",
      "enabled": true,
      "description": "Exa AI Search via Smithery.ai",
      "url": "https://server.smithery.ai/exa/mcp"
    }
  },
  "tools": {
    "exa_search_web_search_exa": {
      "server": "exa-search",
      "description": "Perform web search using Exa"
    }
  }
}
```

### 3. Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/search <query>` | Web search via Exa AI | `/search latest AI developments 2024` |
| `/company <name>` | Company research | `/company Tesla` |
| `/crawl <url>` | Extract content from URL | `/crawl https://example.com/article` |
| `/mcp <subcommand>` | MCP server management | `/mcp status` |

## MCP Commands Reference

### `/search <query>`
Performs web search using Exa AI integration.

**Features:**
- Real-time web results
- AI-enhanced result formatting
- Source attribution
- Fallback to AI knowledge when MCP unavailable

**Example:**
```
/search quantum computing breakthroughs 2024
```

### `/company <name>`
Researches companies using business intelligence tools.

**Features:**
- Company overview and analysis
- Business model insights
- Recent developments
- Market position analysis

**Example:**
```
/company Microsoft
```

### `/crawl <url>`
Extracts and analyzes content from web pages.

**Features:**
- Content extraction
- Text analysis
- Summary generation
- Document processing

**Example:**
```
/crawl https://arxiv.org/abs/2023.12345
```

### `/mcp <subcommand>`
Manages MCP servers and tools.

**Subcommands:**
- `status` - Show MCP system status
- `servers` - List available servers
- `tools` - List available tools
- `query <server> <query>` - Query specific server

**Examples:**
```
/mcp status
/mcp servers
/mcp query exa-search artificial intelligence
```

## Architecture

### MCP Registry (`src/services/mcp_registry.py`)
- **Server Management**: Dynamic loading and configuration of MCP servers
- **Tool Registry**: Centralized tool discovery and execution
- **Connection Handling**: Manages connections to Smithery.ai and other providers
- **Error Handling**: Graceful fallback when servers are unavailable

### Enhanced Agent (`src/services/agent.py`)
- **Multi-modal Integration**: Combines AI models with MCP tools
- **Request Orchestration**: Intelligently routes requests to appropriate tools
- **Context Enhancement**: Augments prompts with tool results
- **Fallback Logic**: Maintains functionality when MCP is unavailable

### Command Integration (`src/handlers/commands/mcp_commands.py`)
- **Telegram Interface**: User-friendly command handling
- **Error Recovery**: Comprehensive error handling and user feedback
- **Group Support**: Enhanced functionality for group chats
- **Progressive Enhancement**: Works with or without MCP configuration

### Group Integration (`src/services/group_mcp_integration.py`)
- **Collaborative Features**: Shared research and search capabilities
- **Group Preferences**: Per-group MCP configuration
- **Search History**: Persistent search tracking
- **Auto-detection**: Smart command recognition in natural language

## Configuration

### MCP Servers

Current supported servers:

| Server | Type | Status | Description |
|--------|------|--------|-------------|
| exa-search | streamable_http | ✅ Active | Exa AI Search via Smithery.ai |
| fetch-mcp | stdio | ✅ Active | Web content fetching |
| context7 | stdio | ✅ Active | Context7 library docs |
| docfork | stdio | ✅ Active | Document analysis |
| duckduckgo-search | stdio | ✅ Active | DuckDuckGo web search |
| sequential-thinking | stdio | ✅ Active | Problem-solving tools |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SMITHERY_API_KEY` | ✅ | Smithery.ai API key for MCP access |
| `EXA_PROFILE` | ⚠️ | Optional Exa profile for enhanced search |
| `TELEGRAM_BOT_TOKEN` | ✅ | Telegram bot token |
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |

## Error Handling

The MCP integration includes comprehensive error handling:

### Graceful Degradation
- **MCP Unavailable**: Falls back to AI model responses
- **Network Issues**: Provides helpful error messages
- **Invalid Queries**: Suggests corrections and alternatives

### User Feedback
- **Status Updates**: Real-time progress indicators
- **Error Messages**: Clear explanations of issues
- **Fallback Options**: Alternative approaches when tools fail

### Logging
- **Debug Information**: Comprehensive logging for troubleshooting
- **Performance Metrics**: Track MCP response times
- **Usage Analytics**: Monitor command usage patterns

## Testing

### Running Tests

```bash
# Test MCP integration
python test_mcp_integration.py

# Test individual components
python -c "
import sys
sys.path.append('src')
from services.mcp_registry import MCPRegistry
print('MCP Registry test passed')
"
```

### Expected Results

```
INFO:__main__:MCP Registry: ✅ PASS
INFO:__main__:Available tools: 3
INFO:__main__:  - exa_search_company_research_exa (server: exa-search)
INFO:__main__:  - exa_search_web_search_exa (server: exa-search)
INFO:__main__:  - exa_search_content_crawler (server: exa-search)
```

## Troubleshooting

### Common Issues

**1. MCP Commands Not Working**
- Check environment variables are set
- Verify `mcp.json` configuration
- Test with `/mcp status`

**2. Search Results Empty**
- Ensure SMITHERY_API_KEY is valid
- Check network connectivity
- Try fallback AI responses

**3. Group Features Not Available**
- Verify group chat permissions
- Check group integration is enabled
- Review group preferences

### Debug Commands

```bash
# Check MCP status
/mcp status

# List available tools
/mcp tools

# Test specific server
/mcp query exa-search test query
```

## Contributing

### Adding New MCP Servers

1. Update `mcp.json` with server configuration
2. Add server-specific tools
3. Update documentation
4. Test integration

### Extending Commands

1. Add new command methods to `MCPCommands`
2. Register commands in `CommandHandlers`
3. Update help documentation
4. Add error handling

## Security Considerations

- **API Key Protection**: Store sensitive keys in environment variables
- **Input Validation**: Sanitize all user inputs
- **Rate Limiting**: Respect API rate limits
- **Error Disclosure**: Avoid exposing sensitive error details

## Performance

### Optimization Features
- **Caching**: Results cached for frequently requested queries
- **Async Processing**: Non-blocking command execution
- **Connection Pooling**: Efficient MCP server connections
- **Fallback Speed**: Fast fallback to AI models

### Monitoring
- **Response Times**: Track MCP server performance
- **Success Rates**: Monitor tool execution success
- **Usage Patterns**: Analyze command usage trends

## Future Enhancements

### Planned Features
- **Custom MCP Servers**: User-defined MCP server integration
- **Advanced Caching**: Intelligent result caching strategies
- **Search Orchestration**: Multi-server search aggregation
- **Analytics Dashboard**: Usage and performance analytics

### Integration Opportunities
- **Knowledge Graphs**: Enhanced context understanding
- **Document Analysis**: Deep document intelligence
- **Real-time Collaboration**: Live research sharing
- **AI Model Routing**: Smart model selection for queries