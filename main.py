import os
import time
from workflow.practice_module_flow import generate_task1,generate_task2
from fastapi import FastAPI,UploadFile, File, Form
from agents.speaking_agent import speaking_agent, format_output
from pydantic import BaseModel
from agents.scoring_agent import score_task,combine_results
from typing import List
from typing import Optional
from agents.writing_agent import evaluate_task
import base64
import json
from fastapi import HTTPException,status
from fastapi.responses import JSONResponse, FileResponse
from services.asr_service import transcribe_audio
from services.tts_service import speak_text
from dotenv import load_dotenv

load_dotenv()
app= FastAPI(title="IELTS Writing Test API",
             description="Generate IELTS writing tasks and submit answers for scoring and feedback",
    version="1.0.0",
    docs_url="/docs",        
    redoc_url="/redoc",      
    openapi_url="/openapi.json"
)
#requestmodels
class UserRequest(BaseModel):
    mode: str
    test_type: str

    class Config:
        json_schema_extra = {
            "example": {
                "mode": "practice",
                "test_type": "academic"
            }
        }
#response models
class Task1Response(BaseModel):
    question: str
    image: Optional[str] = None  # base64 or None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "The chart below shows the number of students studying abroad from 2000 to 2020.",
                "image": "iVBORw0KGgoAAAANSUhEUg..."  # base64 encoded image
            }
        }


class Task2Response(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Some people think globalization helps preserve cultures, while others argue it erodes them. Discuss both views and give your opinion."
            }
        }


class WritingTestResponse(BaseModel):
    message: str
    task1: Task1Response
    task2: Task2Response

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Starting practice test for academic writing",
                "task1": {
                    "question": "The chart below shows the number of internet users from 2000 to 2020.",
                    "image": "iVBORw0KGgoAAAANSUhEUg..."
                },
                "task2": {
                    "question": "Some people believe cultural traditions are losing importance due to globalization, while others think globalization helps preserve them. Discuss both views and give your opinion."
                }
            }
        }

print("Server Started")
@app.post("/ielts/writing-tests",
          response_model=WritingTestResponse,
    summary="Generate two writing tasks (task1 & task2)"
)
def start_module(request:UserRequest):
    print("Endpoint called with:",request.mode,request.test_type)

    task1=generate_task1(request.mode,request.test_type)
    task2=generate_task2(request.mode,request.test_type)

    return {
        "message": f"Starting {request.mode} test for {request.test_type} writing",
        "task1": task1,
        "task2": task2
        }


#request model
class TaskSubmission(BaseModel):
    test_type: str  
    task1_question: str
    task1_answer: str = None
    task2_question: str
    task2_answer: str
    task1_image: str = None
    class Config:
        json_schema_extra = {
            "example": {
                "test_type": "academic",
                "task1_question": "Describe the chart about global water usage.",
                "task1_answer": "Water usage is highest in agriculture.",
                "task1_image": "<base64-image-string>",
                "task2_question": "Some people believe exams are unfair.",
                "task2_answer": "Exams are useful but have drawbacks."
            }
        }
#responsemodel
class TaskResult(BaseModel):
    band: float
    feedback: str
    improvements: List[str]

@app.post("/ielts/writing-submission",
          response_model=TaskResult,
          summary="Submit answers for scoring (returns band, feedback, improvements)")


        
def writing_submission(request: TaskSubmission):
    
    #testtype validation
    if request.test_type not in ["academic","general training"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="test_type must be 'academic' or 'general training'")
    #task1 validation
    if request.test_type.lower() =="academic":
        if not request.task1_question or not request.task1_answer or  not request.task1_image:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Task 1 requires question,answer and image for academic test")
        
            
    else:
        if not request.task1_question or  not request.task1_answer:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Task 1 requires question and answer for general training test")
    if not request.task2_question or not request.task2_answer:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Task 2 always requires question and answer for both academic and general training test")
    try:
            return evaluate_task(request)
    except request.Timeout:
            raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service unavailable. Please try again later."
        )
    except request.ConnectionError:
            raise HTTPException(
            status_code=status.HTTP_504_SERVICE_UNAVAILABLE,
            detail="service timeout. Please try again later.")

    except Exception as e:
            raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error. Please try again later."
        )
        


AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# app = FastAPI(
#     title="Speech Processing API",
#     description="""
# This API provides **ASR (Automatic Speech Recognition)** and **TTS (Text-to-Speech)** services.

# ### 🔹 Modes
# - **ASR_MODE**
#   - `local` → Whisper base model (runs locally)
#   - `cloud` → ElevenLabs ASR (requires API key)
# - **TTS_MODE**
#   - `local` → pyttsx3 (runs locally)
#   - `cloud` → ElevenLabs TTS (requires API key & voice ID)

# All uploaded and generated audio files are stored in `audio_files/`.
# """,
#     version="1.0.0"
# )

@app.post(
    "/asr/transcribe",
    summary="Transcribe Audio to Text",
    description="Upload an audio file (WAV/MP3) and receive its transcript. Mode controlled via ASR_MODE in .env.",
    response_description="Transcribed text from audio"
)
async def asr_transcribe(file: UploadFile = File(..., description="Audio file to transcribe")):
    try:
        filename = f"asr_{int(time.time())}_{file.filename}"
        file_path = os.path.join(AUDIO_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        transcript = transcribe_audio(file_path)
        return JSONResponse({"transcript": transcript})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post(
    "/tts/speak",
    summary="Convert Text to Speech",
    description="Send text and receive an MP3 audio file. Mode controlled via TTS_MODE in .env.",
    response_description="MP3 audio file of the text"
)
async def tts_speak(
    text: str = Form(..., description="Text to convert into speech", example="Hello, this is a test speech.")
):
    try:
        filename = f"tts_{int(time.time())}.mp3"
        output_file = os.path.join(AUDIO_DIR, filename)

        audio_path = speak_text(text, output_file)
        if audio_path.startswith("Error"):
            return JSONResponse({"error": audio_path}, status_code=500)

        return FileResponse(audio_path, media_type="audio/mpeg", filename=filename)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
@app.post("/agent/speaking", summary="Evaluate IELTS speaking (parts 1-3)")
async def agent_speaking_endpoint(
    test_id: str = Form(..., description="Test identifier"),
    user_id: str = Form(..., description="User identifier"),
    part_1: Optional[UploadFile] = File(None),
    part_2: Optional[UploadFile] = File(None),
    part_3: Optional[UploadFile] = File(None),
):
    try:
        
        responses = {}
        # Save uploaded files to audio_files/
        for part_key, upload in (("part_1", part_1), ("part_2", part_2), ("part_3", part_3)):
            if upload is not None:
                filename = f"{user_id}_{int(time.time())}_{os.path.basename(upload.filename)}"
                path = os.path.join(AUDIO_DIR, filename)
                with open(path, "wb") as f:
                    f.write(await upload.read())
                responses[part_key] = path

        if not responses:
            return JSONResponse({"error": "No audio files uploaded (part_1/part_2/part_3)."}, status_code=400)

        # Build state and invoke LangGraph speaking_agent
        state = {"test_id": test_id, "user_id": user_id, "responses": responses}
        result_state = speaking_agent.invoke(state)
        output = format_output(result_state)
        return JSONResponse(output)

    except Exception as e:
        print("Error in /agent/speaking:", e)
        return JSONResponse({"error": str(e)}, status_code=500)
