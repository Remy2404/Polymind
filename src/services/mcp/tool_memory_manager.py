#!/usr/bin/env python3
"""
Tool Call Memory Manager
Provides workspace-wide memory and learning for MCP tool calls.
Tracks tool usage patterns, success rates, and enables the agent to learn
from tool interactions across all conversations and users.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.database.connection import get_database
from src.utils.log.telegramlog import telegram_logger

logger = logging.getLogger(__name__)

@dataclass
class ToolCallRecord:
    """Record of a single tool call execution"""
    tool_name: str
    server_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    context_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "arguments": self.arguments,
            "result": self.result,
            "success": self.success,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp,
            "context_keywords": self.context_keywords
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallRecord":
        """Create from dictionary"""
        return cls(
            tool_name=data["tool_name"],
            server_name=data["server_name"],
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            success=data.get("success", False),
            execution_time=data.get("execution_time", 0.0),
            error_message=data.get("error_message"),
            user_id=data.get("user_id"),
            conversation_id=data.get("conversation_id"),
            timestamp=data.get("timestamp", time.time()),
            context_keywords=data.get("context_keywords", [])
        )

@dataclass
class ToolPerformanceStats:
    """Performance statistics for a tool"""
    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_execution_time: float = 0.0
    last_used: float = 0.0
    common_arguments: Dict[str, int] = field(default_factory=dict)
    common_errors: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0

    def update_stats(self, record: ToolCallRecord):
        """Update statistics with a new record"""
        self.total_calls += 1
        self.last_used = record.timestamp

        if record.success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1

        # Update success rate
        self.success_rate = self.successful_calls / self.total_calls

        # Track common arguments (simplified)
        for arg_name in record.arguments.keys():
            self.common_arguments[arg_name] = self.common_arguments.get(arg_name, 0) + 1

        # Track common errors
        if record.error_message:
            self.common_errors[record.error_message] = self.common_errors.get(record.error_message, 0) + 1

        # Update average execution time
        if self.total_calls == 1:
            self.average_execution_time = record.execution_time
        else:
            self.average_execution_time = (
                (self.average_execution_time * (self.total_calls - 1)) + record.execution_time
            ) / self.total_calls

class ToolMemoryManager:
    """Manages workspace-wide tool call memory and learning"""

    def __init__(self):
        self.db, self.client = get_database()
        self.tool_calls_collection = self.db.tool_calls if self.db is not None else None
        self.tool_stats_collection = self.db.tool_stats if self.db is not None else None

        # In-memory cache for performance
        self._stats_cache: Dict[str, ToolPerformanceStats] = {}
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes

    async def record_tool_call(
        self,
        tool_name: str,
        server_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
        execution_time: float,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        context_keywords: Optional[List[str]] = None
    ) -> bool:
        """
        Record a tool call for learning and analytics

        Args:
            tool_name: Name of the tool called
            server_name: Name of the MCP server
            arguments: Arguments passed to the tool
            result: Tool execution result
            success: Whether the call was successful
            execution_time: Time taken to execute (seconds)
            error_message: Error message if failed
            user_id: ID of the user who made the call
            conversation_id: ID of the conversation
            context_keywords: Keywords from the conversation context

        Returns:
            True if recorded successfully
        """
        try:
            record = ToolCallRecord(
                tool_name=tool_name,
                server_name=server_name,
                arguments=arguments,
                result=result,
                success=success,
                execution_time=execution_time,
                error_message=error_message,
                user_id=user_id,
                conversation_id=conversation_id,
                context_keywords=context_keywords or []
            )

            # Store in database
            if self.tool_calls_collection is not None:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.tool_calls_collection.insert_one(record.to_dict())
                )

            # Update performance statistics
            await self._update_tool_stats(record)

            logger.info(f"Recorded tool call: {tool_name} ({'success' if success else 'failed'})")
            return True

        except Exception as e:
            logger.error(f"Failed to record tool call: {e}")
            telegram_logger.log_error(f"Tool call recording failed: {e}", 0)
            return False

    async def _update_tool_stats(self, record: ToolCallRecord):
        """Update tool performance statistics"""
        try:
            stats = self._stats_cache.get(record.tool_name)
            if stats is None:
                stats = await self._load_tool_stats(record.tool_name)
                if stats is None:
                    stats = ToolPerformanceStats(tool_name=record.tool_name)

            stats.update_stats(record)
            self._stats_cache[record.tool_name] = stats

            # Persist updated stats
            if self.tool_stats_collection is not None:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.tool_stats_collection.update_one(
                        {"tool_name": record.tool_name},
                        {"$set": {
                            "total_calls": stats.total_calls,
                            "successful_calls": stats.successful_calls,
                            "failed_calls": stats.failed_calls,
                            "average_execution_time": stats.average_execution_time,
                            "last_used": stats.last_used,
                            "common_arguments": stats.common_arguments,
                            "common_errors": stats.common_errors,
                            "success_rate": stats.success_rate,
                            "last_updated": time.time()
                        }},
                        upsert=True
                    )
                )

        except Exception as e:
            logger.error(f"Failed to update tool stats: {e}")

    async def _load_tool_stats(self, tool_name: str) -> Optional[ToolPerformanceStats]:
        """Load tool statistics from database"""
        try:
            if self.tool_stats_collection is not None:
                data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.tool_stats_collection.find_one({"tool_name": tool_name})
                )

                if data:
                    stats = ToolPerformanceStats(
                        tool_name=data["tool_name"],
                        total_calls=data.get("total_calls", 0),
                        successful_calls=data.get("successful_calls", 0),
                        failed_calls=data.get("failed_calls", 0),
                        average_execution_time=data.get("average_execution_time", 0.0),
                        last_used=data.get("last_used", 0.0),
                        common_arguments=data.get("common_arguments", {}),
                        common_errors=data.get("common_errors", {}),
                        success_rate=data.get("success_rate", 0.0)
                    )
                    return stats

        except Exception as e:
            logger.error(f"Failed to load tool stats for {tool_name}: {e}")

        return None

    async def get_tool_recommendations(
        self,
        context_keywords: List[str],
        limit: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Get tool recommendations based on context and performance

        Args:
            context_keywords: Keywords from current context
            limit: Maximum number of recommendations

        Returns:
            List of (tool_name, confidence_score, metadata) tuples
        """
        try:
            # Get all tool stats
            all_stats = await self.get_all_tool_stats()

            recommendations = []
            for tool_name, stats in all_stats.items():
                if stats.total_calls == 0:
                    continue

                # Calculate confidence score based on:
                # - Success rate (40%)
                # - Recent usage (30%)
                # - Total usage (20%)
                # - Context relevance (10%)
                success_score = stats.success_rate * 0.4

                # Recent usage score (higher for recently used tools)
                days_since_last_use = (time.time() - stats.last_used) / (24 * 3600)
                recency_score = max(0, 1 - (days_since_last_use / 30)) * 0.3  # 30 days decay

                # Usage frequency score
                usage_score = min(1.0, stats.total_calls / 100) * 0.2  # Cap at 100 calls

                # Context relevance (simplified keyword matching)
                context_score = 0.0
                if context_keywords:
                    # Check if tool name or common args match context
                    tool_words = set(tool_name.lower().split('_'))
                    context_words = set(kw.lower() for kw in context_keywords)
                    overlap = len(tool_words.intersection(context_words))
                    context_score = min(1.0, overlap / len(tool_words)) * 0.1

                confidence = success_score + recency_score + usage_score + context_score

                if confidence > 0.1:  # Minimum threshold
                    metadata = {
                        "success_rate": stats.success_rate,
                        "total_calls": stats.total_calls,
                        "avg_execution_time": stats.average_execution_time,
                        "last_used_days": days_since_last_use
                    }
                    recommendations.append((tool_name, confidence, metadata))

            # Sort by confidence and return top recommendations
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:limit]

        except Exception as e:
            logger.error(f"Failed to get tool recommendations: {e}")
            return []

    async def get_all_tool_stats(self) -> Dict[str, ToolPerformanceStats]:
        """Get statistics for all tools"""
        try:
            # Check cache first
            if time.time() - self._cache_timestamp < self._cache_ttl and self._stats_cache:
                return self._stats_cache.copy()

            # Load from database
            all_stats = {}
            if self.tool_stats_collection is not None:
                cursor = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: list(self.tool_stats_collection.find({}))
                )

                for data in cursor:
                    stats = ToolPerformanceStats(
                        tool_name=data["tool_name"],
                        total_calls=data.get("total_calls", 0),
                        successful_calls=data.get("successful_calls", 0),
                        failed_calls=data.get("failed_calls", 0),
                        average_execution_time=data.get("average_execution_time", 0.0),
                        last_used=data.get("last_used", 0.0),
                        common_arguments=data.get("common_arguments", {}),
                        common_errors=data.get("common_errors", {}),
                        success_rate=data.get("success_rate", 0.0)
                    )
                    all_stats[data["tool_name"]] = stats

            # Update cache
            self._stats_cache = all_stats.copy()
            self._cache_timestamp = time.time()

            return all_stats

        except Exception as e:
            logger.error(f"Failed to get all tool stats: {e}")
            return {}

    async def get_tool_history(
        self,
        tool_name: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        days_back: int = 30
    ) -> List[ToolCallRecord]:
        """
        Get tool call history with optional filtering

        Args:
            tool_name: Filter by specific tool
            user_id: Filter by specific user
            limit: Maximum number of records
            days_back: Number of days to look back

        Returns:
            List of tool call records
        """
        try:
            if self.tool_calls_collection is None:
                return []

            # Build query
            query = {
                "timestamp": {
                    "$gte": time.time() - (days_back * 24 * 3600)
                }
            }

            if tool_name:
                query["tool_name"] = tool_name
            if user_id:
                query["user_id"] = user_id

            # Execute query
            cursor = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(self.tool_calls_collection.find(query)
                           .sort("timestamp", -1)
                           .limit(limit))
            )

            records = [ToolCallRecord.from_dict(data) for data in cursor]
            return records

        except Exception as e:
            logger.error(f"Failed to get tool history: {e}")
            return []

    async def get_tool_insights(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed insights for a specific tool

        Args:
            tool_name: Name of the tool to analyze

        Returns:
            Dictionary with tool insights
        """
        try:
            stats = await self._load_tool_stats(tool_name)
            if not stats:
                return {"error": f"No data found for tool: {tool_name}"}

            history = await self.get_tool_history(tool_name=tool_name, limit=100)

            insights = {
                "tool_name": tool_name,
                "performance": {
                    "total_calls": stats.total_calls,
                    "success_rate": stats.success_rate,
                    "average_execution_time": stats.average_execution_time,
                    "last_used": datetime.fromtimestamp(stats.last_used).isoformat()
                },
                "usage_patterns": {
                    "common_arguments": stats.common_arguments,
                    "common_errors": stats.common_errors
                },
                "recent_activity": [
                    {
                        "timestamp": datetime.fromtimestamp(record.timestamp).isoformat(),
                        "success": record.success,
                        "execution_time": record.execution_time,
                        "user_id": record.user_id
                    }
                    for record in history[:10]  # Last 10 calls
                ]
            }

            return insights

        except Exception as e:
            logger.error(f"Failed to get tool insights for {tool_name}: {e}")
            return {"error": str(e)}

    async def cleanup_old_records(self, days_to_keep: int = 90):
        """
        Clean up old tool call records to manage database size

        Args:
            days_to_keep: Number of days of records to keep
        """
        try:
            if not self.tool_calls_collection:
                return

            cutoff_time = time.time() - (days_to_keep * 24 * 3600)

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.tool_calls_collection.delete_many(
                    {"timestamp": {"$lt": cutoff_time}}
                )
            )

            logger.info(f"Cleaned up {result.deleted_count} old tool call records")

        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")

    async def get_workspace_tool_stats(self) -> Dict[str, Any]:
        """
        Get overall workspace statistics for tool usage

        Returns:
            Dictionary with workspace-wide tool statistics
        """
        try:
            all_stats = await self.get_all_tool_stats()

            if not all_stats:
                return {"error": "No tool statistics available"}

            total_tools = len(all_stats)
            total_calls = sum(stats.total_calls for stats in all_stats.values())
            total_successful = sum(stats.successful_calls for stats in all_stats.values())

            overall_success_rate = total_successful / total_calls if total_calls > 0 else 0.0

            # Most used tools
            most_used = sorted(
                [(name, stats.total_calls) for name, stats in all_stats.items()],
                key=lambda x: x[1],
                reverse=True
            )

            # Best performing tools (by success rate, min 5 calls)
            best_performing = sorted(
                [(name, stats.success_rate) for name, stats in all_stats.items()
                 if stats.total_calls >= 5],
                key=lambda x: x[1],
                reverse=True
            )[:5]

            return {
                "total_tools": total_tools,
                "total_calls": total_calls,
                "overall_success_rate": overall_success_rate,
                "most_used_tools": most_used,
                "best_performing_tools": best_performing
            }

        except Exception as e:
            logger.error(f"Failed to get workspace tool stats: {e}")
            return {"error": str(e)}

# Global instance
tool_memory_manager = ToolMemoryManager()