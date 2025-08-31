#!/usr/bin/env python3
"""
HTTP API routes for TTS Provider monitoring
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import psutil
import time
from pathlib import Path
import os

from services.tts_service import TTSService

router = APIRouter()

# Global variable to hold the TTS service instance
tts_service = None

def set_tts_service(service):
    """Set the TTS service instance for the routes to use"""
    global tts_service
    tts_service = service

class ServiceStatus(BaseModel):
    """Model for individual service status."""
    
    name: str
    status: str  # "healthy", "degraded", "down"
    details: Optional[Dict[str, Any]] = None
    last_updated: str


class SystemMetrics(BaseModel):
    """Model for system metrics."""
    
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    uptime: Optional[str] = None


class MonitoringResponse(BaseModel):
    """Response model for system monitoring."""
    
    status: str  # "healthy", "degraded", "down"
    service_name: str
    version: str
    timestamp: str
    metrics: Optional[SystemMetrics] = None
    services: Optional[List[ServiceStatus]] = None
    details: Optional[Dict[str, Any]] = None


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    if tts_service is None:
        return {
            "status": "degraded",
            "service": "TTS-Provider",
            "version": "1.0.0",
            "tts_ready": False,
            "message": "TTS service not initialized"
        }
    return {
        "status": "healthy",
        "service": "TTS-Provider",
        "version": "1.0.0",
        "tts_ready": tts_service.is_ready() if tts_service else False
    }


@router.get("/monitoring")
async def get_monitoring_status() -> MonitoringResponse:
    """
    Get comprehensive monitoring status of the TTS Provider service.
    
    Returns:
        MonitoringResponse: Detailed system status and metrics
    """
    try:
        import psutil
        import time
        from datetime import datetime, timedelta
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        # Calculate uptime
        boot_time = psutil.boot_time()
        uptime = str(timedelta(seconds=time.time() - boot_time))
        
        metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_info.percent,
            disk_usage=(disk_info.used / disk_info.total) * 100,
            uptime=uptime
        )
        
        # Get service status (TTS)
        tts_status = ServiceStatus(
            name="TTS Service",
            status="healthy" if tts_service and tts_service.is_ready() else "degraded",
            details={
                "ready": tts_service is not None,
                "model_loaded": tts_service.is_ready() if tts_service else False,
                "model_name": tts_service.model_name if tts_service else None
            },
            last_updated=datetime.now().isoformat()
        )
        
        # Overall status
        overall_status = "healthy" if tts_service and tts_service.is_ready() else "degraded"
        
        return MonitoringResponse(
            status=overall_status,
            service_name="TTS-Provider",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            services=[tts_status]
        )
        
    except ImportError:
        # If psutil is not available, return basic status
        return MonitoringResponse(
            status="healthy",
            service_name="TTS-Provider",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            details={"message": "Detailed metrics not available (psutil not installed)"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring status: {str(e)}")


@router.get("/models")
async def get_available_models():
    """Get information about available TTS models."""
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    try:
        models = tts_service.list_available_models()
        return {
            "models": models,
            "current_model": tts_service.model_name if tts_service else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model information: {str(e)}")


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to TTS-Provider API 🎵",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /monitoring": "Comprehensive system monitoring",
            "GET /models": "Get available TTS models"
        },
        "features": [
            "Text-to-speech conversion with multiple models",
            "WebSocket-based streaming audio delivery",
            "Support for Edge TTS and Zonos models",
            "Dynamic model loading"
        ],
        "health_endpoint": "/health"
    }