from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import soundfile as sf
from TTS.api import TTS
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from pathlib import Path
from typing import List, Dict
import shutil
import uuid
import os


# App Initialization
app = FastAPI(
    title="Neural Text-to-Speech and Speech-to-Text API",
    version="2.0",
    description="Convert text to natural-sounding speech using a neural TTS model."
)

# Load the neural TTS model once at startup for efficiency
tts_model = TTS(
    model_name="tts_models/en/ljspeech/tacotron2-DDC",
    progress_bar=False,
    gpu=False
)

# Request Schema
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    rate: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    volume: float = Field(1.0, ge=0.5, le=2.0, description="Volume multiplier")


def synthesize_speech(text: str, rate: float, volume: float, output_path: str):
    # Generate waveform
    wav = tts_model.tts(text)

    # Convert to NumPy float32
    wav = np.array(wav, dtype=np.float32)

    # Apply volume
    wav *= volume

    # Clip to prevent distortion/crash
    wav = np.clip(wav, -1.0, 1.0)

    # Save using soundfile (more reliable than save_wav)
    sf.write(output_path, wav, samplerate=22050)

# API Endpoints
@app.post("/generate-audio", response_class=FileResponse, tags=["TTS"])
async def generate_audio(request: TTSRequest):
    """
    Generate speech from text and return the audio file.
    """
    try:
        # Ensure output directory exists
        os.makedirs("audio_files", exist_ok=True)

        # Generate unique file name
        file_name = f"{uuid.uuid4()}.wav"
        output_path = os.path.join("audio_files", file_name)

        # Synthesize speech
        synthesize_speech(
            text=request.text,
            rate=request.rate,
            volume=request.volume,
            output_path=output_path
        )

        # Check if file was successfully created
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise HTTPException(status_code=500, detail="Audio generation failed")

        # Return the WAV file
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="tts_response.wav"
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class SpeechToText:
    """
    Speech-to-Text engine using Faster-Whisper.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_audio(self, audio_path: str) -> Dict:
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Perform transcription
        segments, info = self.model.transcribe(str(audio_file))

        # Combine all segments into full text
        full_text = " ".join(segment.text for segment in segments).strip()

        # Segment timestamps
        segment_data = [
            {"start": float(segment.start), "end": float(segment.end), "text": str(segment.text)}
            for segment in segments
        ]

        # Metadata info — convert to JSON-serializable dict
        if hasattr(info, "_asdict"):
            metadata_raw = info._asdict()
        else:
            metadata_raw = vars(info)

        # Convert all non-serializable fields to string
        metadata = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    for k, v in metadata_raw.items()}

        return {
            "full_text": full_text,
            "metadata": metadata,
            "segments": segment_data
        }

# Initialize STT once (reuse model for all requests)
stt_engine = SpeechToText(model_size="base", device="cpu", compute_type="int8")

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/transcribe", response_class=FileResponse, tags=["STT"])
async def transcribe(file: UploadFile = File(...)):
    """
    Upload an audio file and get transcription with metadata and segment timestamps.
    """
    # Validate file type (basic check)
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # Save uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe audio
        result = stt_engine.transcribe_audio(temp_filename)
        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.get("/", tags=["ROOT"])
def root():
    """
    Health check endpoint to verify API is running.
    """
    return {"message": "Neural TTS API is running"}
