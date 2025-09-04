from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

class TTSRequest(BaseModel):
    text: str
    speaker: int = 0
    sample_rate: int = 24000
    model: str = "edge"
    lang: str = "en-US"

def create_http_routes(tts_service):
    router = APIRouter()
    logger = logging.getLogger("TTS-HTTP-Routes")

    @router.post("/tts")
    async def http_tts(request: TTSRequest):
        try:
            logger.info(f"Received HTTP TTS request for text: {request.text[:50]}...")
            audio_bytes = await tts_service.generate_speech(
                text=request.text,
                speaker=request.speaker,
                sample_rate=request.sample_rate,
                model=request.model,
                lang=request.lang
            )
            return {"audio": audio_bytes}
        except Exception as e:
            logger.error(f"Error processing HTTP TTS request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "ok"}

    @router.get("/ready")
    async def ready_check():
        """Readiness check endpoint"""
        if tts_service.is_ready():
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")

    return router