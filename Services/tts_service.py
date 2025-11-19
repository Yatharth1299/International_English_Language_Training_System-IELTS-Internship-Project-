import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TTS_MODE = os.getenv("TTS_MODE", "local")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

def speak_text(text: str, output_file: str = "output.mp3") -> str:
    """
    Convert text to speech and save as MP3.

    Modes:
    - local: pyttsx3
    - cloud: ElevenLabs TTS
    """
    try:
        if TTS_MODE == "local":
            logger.info("Using local pyttsx3 TTS...")
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            return output_file

        elif TTS_MODE == "cloud":
            logger.info("Using ElevenLabs Cloud TTS...")
            if not ELEVENLABS_VOICE_ID:
                raise ValueError("ELEVENLABS_VOICE_ID not set in environment")

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {"text": text}
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            with open(output_file, "wb") as f:
                f.write(response.content)

            return output_file

        else:
            raise ValueError(f"Invalid TTS_MODE: {TTS_MODE}")

    except Exception as e:
        logger.error(f"TTS Error: {str(e)}")
        return f"Error in TTS: {str(e)}"
