from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MCPTool:
    name: str
    server: str
    description: Optional[str] = None


class MCPRegistry:
    """
    Lightweight registry that reads mcp.json and exposes available servers/tools.

    This does not spawn MCP processes itself; it provides metadata so the Pydantic AI agent
    can register and invoke tools via the client integration you use at runtime.
    """

    def __init__(self, config_path: str | Path = "mcp.json") -> None:
        self.config_path = Path(config_path)
        self._raw: Dict[str, Any] = {}
        self._servers: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"MCP config not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as f:
            self._raw = json.load(f)

        self._servers = self._raw.get("mcpServers", {})

    def reload(self) -> None:
        self._load()

    def list_servers(self) -> List[str]:
        return list(self._servers.keys())

    def server_config(self, name: str) -> Dict[str, Any]:
        return self._servers.get(name, {})

    def list_tools(self, server: Optional[str] = None) -> List[MCPTool]:
        tools: List[MCPTool] = []
        servers = [server] if server else self.list_servers()
        for s in servers:
            cfg = self.server_config(s)
            for tool in cfg.get("tools", []) or []:
                tools.append(MCPTool(name=tool, server=s))
        return tools

    def enabled_servers(self) -> List[str]:
        enabled = set(self._raw.get("globalSettings", {}).get("enabledByDefault", []) or [])
        # If a server has explicit enabled flag false (for http servers), skip
        result = []
        for name, cfg in self._servers.items():
            if cfg.get("enabled", True) and (not enabled or name in enabled):
                result.append(name)
        return result

    def health(self) -> Dict[str, Any]:
        required_env = ["SMITHERY_API_KEY"]
        missing_env = [k for k in required_env if not os.getenv(k)]
        return {
            "config_path": str(self.config_path),
            "servers": self.list_servers(),
            "enabled_servers": self.enabled_servers(),
            "has_tools": any(self.list_tools()),
            "missing_env": missing_env,
        }


__all__ = ["MCPRegistry", "MCPTool"]
