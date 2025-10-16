"""
Enhanced error handling and monitoring for streaming responses.
Provides comprehensive error tracking, metrics collection, and observability.
"""

import time
import json
import logging
import traceback
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of different error types."""
    VALIDATION_ERROR = "validation_error"
    MODEL_ERROR = "model_error"
    STREAMING_ERROR = "streaming_error"
    DATABASE_ERROR = "database_error"
    AUTHENTICATION_ERROR = "auth_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorEvent:
    """Structured error event for tracking and analysis."""
    timestamp: float
    error_type: ErrorType
    severity: ErrorSeverity
    user_id: Optional[str]
    endpoint: str
    error_message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class MetricsSnapshot:
    """Performance and error metrics snapshot."""
    timestamp: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    error_rate_percent: float
    active_streams: int
    errors_by_type: Dict[str, int]
    errors_by_severity: Dict[str, int]


class ErrorTracker:
    """Tracks and analyzes application errors for monitoring and alerting."""
    
    def __init__(self, max_events: int = 1000, alert_threshold: int = 10):
        self.max_events = max_events
        self.alert_threshold = alert_threshold
        self.error_events: deque = deque(maxlen=max_events)
        self.error_counts: Dict[ErrorType, int] = defaultdict(int)
        self.severity_counts: Dict[ErrorSeverity, int] = defaultdict(int)
        self.recent_errors: deque = deque(maxlen=100)  # Last 100 errors for alerting
        
    def track_error(self, error_event: ErrorEvent) -> None:
        """Track a new error event."""
        self.error_events.append(error_event)
        self.error_counts[error_event.error_type] += 1
        self.severity_counts[error_event.severity] += 1
        self.recent_errors.append(error_event)
        
        # Log based on severity
        if error_event.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error_event.error_message}", 
                          extra={"error_event": asdict(error_event)})
        elif error_event.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR: {error_event.error_message}",
                        extra={"error_event": asdict(error_event)})
        elif error_event.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR: {error_event.error_message}")
        else:
            logger.info(f"LOW SEVERITY ERROR: {error_event.error_message}")
        
        # Check for alert conditions
        self._check_alert_conditions()
    
    def _check_alert_conditions(self) -> None:
        """Check if alert conditions are met."""
        # Check for error spike in last 5 minutes
        current_time = time.time()
        recent_errors_5min = [
            e for e in self.recent_errors 
            if current_time - e.timestamp < 300  # 5 minutes
        ]
        
        if len(recent_errors_5min) >= self.alert_threshold:
            logger.warning(
                f"ERROR SPIKE DETECTED: {len(recent_errors_5min)} errors in last 5 minutes"
            )
        
        # Check for critical errors
        critical_errors = [e for e in recent_errors_5min if e.severity == ErrorSeverity.CRITICAL]
        if critical_errors:
            logger.critical(f"CRITICAL ERROR ALERT: {len(critical_errors)} critical errors detected")
    
    def get_error_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get error summary for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.error_events if e.timestamp >= cutoff_time]
        
        return {
            "time_period_hours": hours,
            "total_errors": len(recent_events),
            "errors_by_type": {
                error_type.value: len([e for e in recent_events if e.error_type == error_type])
                for error_type in ErrorType
            },
            "errors_by_severity": {
                severity.value: len([e for e in recent_events if e.severity == severity])
                for severity in ErrorSeverity
            },
            "most_common_errors": self._get_most_common_errors(recent_events),
            "affected_users": len(set(e.user_id for e in recent_events if e.user_id)),
            "affected_endpoints": len(set(e.endpoint for e in recent_events))
        }
    
    def _get_most_common_errors(self, events: List[ErrorEvent], limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most common error messages."""
        error_counts = defaultdict(int)
        for event in events:
            error_counts[event.error_message] += 1
        
        return [
            {"message": message, "count": count}
            for message, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]


class PerformanceMonitor:
    """Monitors application performance metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.request_times: deque = deque(maxlen=window_size)
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.active_streams = 0
        self.start_time = time.time()
        
    def record_request(self, duration_ms: float, success: bool) -> None:
        """Record a request completion."""
        self.request_times.append(duration_ms)
        self.request_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def start_stream(self) -> None:
        """Record the start of a streaming request."""
        self.active_streams += 1
    
    def end_stream(self) -> None:
        """Record the end of a streaming request."""
        self.active_streams = max(0, self.active_streams - 1)
    
    def get_metrics(self) -> MetricsSnapshot:
        """Get current performance metrics."""
        avg_response_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        error_rate = (self.failure_count / self.request_count * 100) if self.request_count > 0 else 0
        
        return MetricsSnapshot(
            timestamp=time.time(),
            total_requests=self.request_count,
            successful_requests=self.success_count,
            failed_requests=self.failure_count,
            avg_response_time_ms=avg_response_time,
            error_rate_percent=error_rate,
            active_streams=self.active_streams,
            errors_by_type={},  # Will be filled by error tracker
            errors_by_severity={}  # Will be filled by error tracker
        )


class StreamingErrorHandler:
    """Specialized error handler for streaming responses."""
    
    def __init__(self, error_tracker: ErrorTracker, performance_monitor: PerformanceMonitor):
        self.error_tracker = error_tracker
        self.performance_monitor = performance_monitor
    
    @asynccontextmanager
    async def handle_streaming_errors(
        self,
        user_id: str,
        endpoint: str,
        model_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """Context manager for handling streaming errors."""
        start_time = time.time()
        self.performance_monitor.start_stream()
        
        try:
            yield
            # Success
            duration_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_request(duration_ms, success=True)
            
        except asyncio.CancelledError:
            # Client disconnected - not really an error
            duration_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_request(duration_ms, success=True)
            logger.info(f"Streaming request cancelled by client for user {user_id}")
            raise
            
        except Exception as e:
            # Actual error
            duration_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_request(duration_ms, success=False)
            
            # Classify error
            error_type, severity = self._classify_error(e)
            
            # Create error event
            error_event = ErrorEvent(
                timestamp=time.time(),
                error_type=error_type,
                severity=severity,
                user_id=user_id,
                endpoint=endpoint,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                context={
                    "model_name": model_name,
                    "duration_ms": duration_ms,
                    "error_class": e.__class__.__name__
                },
                request_id=request_id,
                model_name=model_name,
                duration_ms=duration_ms
            )
            
            # Track the error
            self.error_tracker.track_error(error_event)
            raise
            
        finally:
            self.performance_monitor.end_stream()
    
    def _classify_error(self, error: Exception) -> tuple[ErrorType, ErrorSeverity]:
        """Classify an error by type and severity."""
        error_class = error.__class__.__name__
        error_message = str(error).lower()
        
        # Classification logic
        if "validation" in error_message or "pydantic" in error_class.lower():
            return ErrorType.VALIDATION_ERROR, ErrorSeverity.LOW
        
        elif "rate limit" in error_message or "429" in error_message:
            return ErrorType.RATE_LIMIT_ERROR, ErrorSeverity.MEDIUM
        
        elif "auth" in error_message or "unauthorized" in error_message or "403" in error_message:
            return ErrorType.AUTHENTICATION_ERROR, ErrorSeverity.MEDIUM
        
        elif "database" in error_message or "mongo" in error_message:
            return ErrorType.DATABASE_ERROR, ErrorSeverity.HIGH
        
        elif "model" in error_message or "openai" in error_message or "gemini" in error_message:
            return ErrorType.MODEL_ERROR, ErrorSeverity.MEDIUM
        
        elif "network" in error_message or "connection" in error_message or "timeout" in error_message:
            return ErrorType.NETWORK_ERROR, ErrorSeverity.MEDIUM
        
        elif "stream" in error_message:
            return ErrorType.STREAMING_ERROR, ErrorSeverity.HIGH
        
        else:
            # Unknown error - could be critical
            if "critical" in error_message or "fatal" in error_message:
                return ErrorType.UNKNOWN_ERROR, ErrorSeverity.CRITICAL
            else:
                return ErrorType.UNKNOWN_ERROR, ErrorSeverity.MEDIUM
    
    async def create_error_response_stream(self, error_message: str, error_type: str = "error") -> str:
        """Create a standardized error response for streaming endpoints."""
        error_response = {
            "type": "error",
            "error": error_message,
            "error_type": error_type,
            "timestamp": datetime.now().timestamp(),
            "recoverable": error_type in ["rate_limit_error", "validation_error"]
        }
        
        return f'data: {json.dumps(error_response)}\n\n'


# Global instances
error_tracker = ErrorTracker()
performance_monitor = PerformanceMonitor()
streaming_error_handler = StreamingErrorHandler(error_tracker, performance_monitor)


def get_monitoring_stats() -> Dict[str, Any]:
    """Get comprehensive monitoring statistics."""
    metrics = performance_monitor.get_metrics()
    error_summary = error_tracker.get_error_summary(hours=1)
    
    return {
        "performance": asdict(metrics),
        "errors": error_summary,
        "uptime_seconds": time.time() - performance_monitor.start_time,
        "health_status": _calculate_health_status(metrics, error_summary)
    }


def _calculate_health_status(metrics: MetricsSnapshot, error_summary: Dict[str, Any]) -> str:
    """Calculate overall health status based on metrics."""
    # Health criteria
    if metrics.error_rate_percent > 10:
        return "unhealthy"
    elif metrics.error_rate_percent > 5 or error_summary["errors_by_severity"]["critical"] > 0:
        return "degraded"
    elif metrics.avg_response_time_ms > 5000:
        return "slow"
    else:
        return "healthy"