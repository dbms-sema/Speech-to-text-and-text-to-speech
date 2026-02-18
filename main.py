from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import soundfile as sf
from TTS.api import TTS
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Neural Speech API</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; padding: 50px; }
        .container { max-width: 800px; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { margin-bottom: 30px; color: #333; }
        .section { margin-bottom: 40px; }
        .btn-action { min-width: 150px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center">Neural Speech API Interface</h1>
        
        <div class="section">
            <h3>Text to Speech</h3>
            <div class="mb-3">
                <textarea id="tts-text" class="form-control" rows="3" placeholder="Enter text here..."></textarea>
            </div>
            <div class="row mb-3">
                <div class="col">
                    <label class="form-label">Rate (0.5 - 2.0)</label>
                    <input type="number" id="tts-rate" class="form-control" value="1.0" step="0.1" min="0.5" max="2.0">
                </div>
                <div class="col">
                    <label class="form-label">Volume (0.5 - 2.0)</label>
                    <input type="number" id="tts-volume" class="form-control" value="1.0" step="0.1" min="0.5" max="2.0">
                </div>
            </div>
            <button onclick="generateAudio()" class="btn btn-primary btn-action">Generate Audio</button>
            <div class="mt-3">
                <audio id="audio-player" controls style="display:none; width: 100%;"></audio>
            </div>
        </div>

        <hr>

        <div class="section">
            <h3>Speech to Text (Transcription)</h3>
            <div class="mb-3">
                <input type="file" id="stt-file" class="form-control" accept=".wav,.mp3,.m4a,.flac">
            </div>
            <button id="stt-btn" onclick="transcribeAudio()" class="btn btn-success btn-action">Transcribe</button>
            <div id="stt-result" class="mt-3 p-3 bg-light border rounded" style="display:none;">
                <h5>Result:</h5>
                <pre id="stt-json" style="white-space: pre-wrap;"></pre>
            </div>
        </div>
    </div>

    <script>
        function generateAudio() {
            const text = document.getElementById('tts-text').value;
            const rate = document.getElementById('tts-rate').value;
            const volume = document.getElementById('tts-volume').value;
            if (!text) return alert('Please enter text');
            
            const player = document.getElementById('audio-player');
            player.style.display = 'block';
            player.src = `/generate-audio?text=${encodeURIComponent(text)}&rate=${rate}&volume=${volume}`;
            player.play();
        }

        async function transcribeAudio() {
            const fileInput = document.getElementById('stt-file');
            const btn = document.getElementById('stt-btn');
            if (fileInput.files.length === 0) return alert('Please select a file');

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const resultDiv = document.getElementById('stt-result');
            const jsonPre = document.getElementById('stt-json');
            
            resultDiv.style.display = 'block';
            jsonPre.innerText = 'Transcribing...';
            btn.disabled = true;

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                jsonPre.innerText = JSON.stringify(result, null, 2);
            } catch (error) {
                jsonPre.innerText = 'Error: ' + error.message;
            } finally {
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

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


@app.get("/generate-audio", response_class=FileResponse, tags=["TTS"])
async def generate_audio_get(
    text: str = Query(..., description="Text to convert to speech"),
    rate: float = Query(1.0, ge=0.5, le=2.0, description="Speech speed multiplier"),
    volume: float = Query(1.0, ge=0.5, le=2.0, description="Volume multiplier")
):
    """
    Generate speech from text via GET request, returning the audio file directly.
    """
    try:
        os.makedirs("audio_files", exist_ok=True)
        file_name = f"{uuid.uuid4()}.wav"
        output_path = os.path.join("audio_files", file_name)

        synthesize_speech(text=text, rate=rate, volume=volume, output_path=output_path)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise HTTPException(status_code=500, detail="Audio generation failed")

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


@app.get("/", tags=["ROOT"], response_class=HTMLResponse)
def root():
    """
    Serve the browser interface.
    """
    return HTML_TEMPLATE
