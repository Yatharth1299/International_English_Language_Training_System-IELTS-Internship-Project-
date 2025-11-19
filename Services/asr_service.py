import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ASR_MODE = os.getenv("ASR_MODE", "local")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASR_MODEL_ID = os.getenv("ASR_MODEL_ID", "scribe_v1")

# Load local Whisper model only if needed
whisper_model = None
if ASR_MODE == "local":
    import whisper
    logger.info("Loading Whisper base model for local ASR...")
    whisper_model = whisper.load_model("base")

def transcribe_audio(audio_file: str) -> str:
    """
    Transcribe audio file to text.

    Modes:
    - local: Whisper
    - cloud: ElevenLabs ASR
    """
    try:
        if ASR_MODE == "local":
            logger.info("Using local Whisper ASR...")
            result = whisper_model.transcribe(audio_file)
            return result.get("text", "")

        elif ASR_MODE == "cloud":
            logger.info("Using ElevenLabs Cloud ASR...")
            url = "https://api.elevenlabs.io/v1/speech-to-text"
            headers = {"xi-api-key": ELEVENLABS_API_KEY}

            with open(audio_file, "rb") as f:
                files = {"file": f}
                data = {"model_id": ASR_MODEL_ID}
                response = requests.post(url, headers=headers, files=files, data=data)

            response.raise_for_status()
            result = response.json()
            return result.get("text", "")

        else:
            return "Error: Invalid ASR_MODE. Must be 'local' or 'cloud'."

    except Exception as e:
        logger.error(f"ASR Error: {e}")
        return f"Error in transcription: {e}"
