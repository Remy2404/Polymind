"""
Health and root endpoint routes for the Telegram Bot API.
Provides basic application status information and health checks.
"""

import time
import psutil
import os
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

router = APIRouter()
__version__ = "1.0.0"  # Application version defined directly in code

@router.get("/")
@router.head("/")
async def root_get():
    """Root endpoint for health checks."""
    return JSONResponse(
        content={
            "status": "ok", 
            "message": "Telegram Bot API is running",
            "version": __version__
        },
        status_code=200,
    )

@router.post("/")
async def root_post():
    """Root endpoint for POST requests."""
    return JSONResponse(
        content={"status": "ok", "message": "Telegram Bot API is running"},
        status_code=200,
    )

@router.get("/health")
@router.head("/health")
async def health_check(bot=Depends(lambda: None)):  # bot will be injected later
    """
    Enhanced health check endpoint with detailed status information.
    Provides system metrics and component status.
    """
    health_data = {
        "status": "ok",
        "timestamp": time.time(),
        "version": __version__,
        "components": {},
    }

    # Add system metrics
    health_data["system"] = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage("/").percent,
    }

    # Components will be checked with actual bot instance when dependency is injected
    if bot:
        # Check components here when bot instance is available
        pass

    # Return health check response
    return JSONResponse(content=health_data, status_code=200)