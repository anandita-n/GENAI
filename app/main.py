"""
SignBridge AI - FastAPI Application Server
Integrates Computer Vision, ASL Gesture Recognition, Generative AI (LLM),
and Text-To-Speech into an end-to-end communication assistant.
"""

import time
import json
import base64
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import (
    STATIC_DIR,
    DATA_DIR,
    ALL_CLASSES,
    ASL_ALPHABET,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_STABILITY_FRAMES,
    METRICS_PATH,
    GEMINI_API_KEY,
    GEMINI_MODEL
)
from app.landmark_extractor import LandmarkExtractor
from app.gesture_classifier import GestureClassifier, TemporalStabilizer
from app.llm_engine import LLMEngine
from app.tts_engine import TTSEngine

app = FastAPI(
    title="SignBridge AI API",
    description="Generative AI-Powered Sign Language Recognition and Communication Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core subsystems
landmark_extractor = LandmarkExtractor()
gesture_classifier = GestureClassifier()
stabilizer = TemporalStabilizer(
    stability_threshold=DEFAULT_STABILITY_FRAMES,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
)
llm_engine = LLMEngine()
tts_engine = TTSEngine()

# In-memory translation history
TRANSLATION_HISTORY: List[Dict[str, Any]] = []


# --- Request/Response Models ---
class LandmarkItem(BaseModel):
    x: float
    y: float
    z: float = 0.0

class RecognizeLandmarksRequest(BaseModel):
    landmarks: List[Dict[str, float]]
    confidence_threshold: Optional[float] = DEFAULT_CONFIDENCE_THRESHOLD

class FrameRecognizeRequest(BaseModel):
    image_base64: str  # JPEG/PNG data URL or raw base64
    confidence_threshold: Optional[float] = DEFAULT_CONFIDENCE_THRESHOLD

class RefineRequest(BaseModel):
    raw_text: str
    style: Optional[str] = "casual"
    context: Optional[List[Dict[str, str]]] = None

class ReplyRequest(BaseModel):
    last_message: str
    role: Optional[str] = "hearing_partner"

class SpeakRequest(BaseModel):
    text: str
    voice: Optional[str] = "Samantha"
    rate: Optional[int] = 175

class SettingsUpdateRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    confidence_threshold: Optional[float] = None
    stability_frames: Optional[int] = None

class BufferSetRequest(BaseModel):
    text: str

class HistoryItemRequest(BaseModel):
    raw_text: str
    refined_text: str
    style: str
    provider: str


# --- API Routes ---

@app.get("/api/status")
def get_status():
    """Returns system status, active models, and configuration."""
    return {
        "status": "online",
        "system": "SignBridge AI Assistant",
        "classifier_ready": gesture_classifier.clf is not None,
        "classes_count": len(gesture_classifier.classes_),
        "classes": gesture_classifier.classes_,
        "gemini_active": bool(llm_engine.api_key),
        "llm_model": llm_engine.model_name,
        "confidence_threshold": stabilizer.confidence_threshold,
        "stability_frames": stabilizer.stability_threshold,
        "voices_count": len(tts_engine.get_available_voices())
    }


@app.get("/api/metrics")
def get_metrics():
    """Returns evaluation metrics report including accuracy, precision, recall, and confusion matrix."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {"error": "Metrics file not found. Train model first."}


@app.get("/api/alphabet")
def get_alphabet():
    """Returns ASL alphabet reference directory."""
    return {
        "alphabet": ASL_ALPHABET,
        "special_gestures": ["SPACE", "DEL", "CLEAR"],
        "total_classes": len(ALL_CLASSES)
    }


@app.post("/api/recognize/landmarks")
def recognize_landmarks(req: RecognizeLandmarksRequest):
    """
    Classifies raw MediaPipe 21 hand landmarks passed from client.
    Stabilizes character stream and returns updated text accumulator.
    """
    if len(req.landmarks) != 21:
        raise HTTPException(status_code=400, detail="Expected exactly 21 landmarks")

    threshold = req.confidence_threshold or stabilizer.confidence_threshold
    pred = gesture_classifier.predict_raw_landmarks(req.landmarks, confidence_threshold=threshold)
    stab_res = stabilizer.process_prediction(pred)

    if stab_res.get("committed_action"):
        print(f"[COMMITTED SIGN] ===> {stab_res['committed_action']} (Confidence: {pred['confidence']})")
    return {
        "prediction": pred,
        "stabilizer": stab_res
    }


@app.post("/api/recognize/frame")
def recognize_frame(req: FrameRecognizeRequest):
    """
    Decodes an image frame, runs backend MediaPipe LandmarkExtractor,
    and returns predictions and bounding box.
    """
    try:
        # Strip header if data URL
        img_data = req.image_base64
        if "base64," in img_data:
            img_data = img_data.split("base64,")[1]
        decoded = base64.b64decode(img_data)
        np_arr = np.frombuffer(decoded, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    extracted = landmark_extractor.extract_from_frame(frame_bgr)
    if not extracted:
        return {
            "hand_detected": False,
            "prediction": {"sign": "NONE", "confidence": 0.0, "is_confident": False, "candidates": []},
            "stabilizer": stabilizer.process_prediction({"raw_sign": "NONE", "confidence": 0.0, "is_confident": False})
        }

    threshold = req.confidence_threshold or stabilizer.confidence_threshold
    pred = gesture_classifier.predict_features(extracted["features"], confidence_threshold=threshold)
    stab_res = stabilizer.process_prediction(pred)

    if stab_res.get("committed_action"):
        print(f"[COMMITTED SIGN] ===> {stab_res['committed_action']} (Confidence: {pred['confidence']})")
    else:
        print(f"[RECOGNIZE FRAME] Pred: {pred['raw_sign']} ({pred['confidence']:.2f}) | Confident: {pred['is_confident']} | Active: {stab_res.get('active_sign')} | Stab: {stab_res.get('stability_ratio')}")

    return {
        "hand_detected": True,
        "bbox": extracted["bbox"],
        "landmarks_pixel": extracted["landmarks_pixel"],
        "prediction": pred,
        "stabilizer": stab_res
    }


# --- Text Buffer Management ---

@app.get("/api/buffer")
def get_buffer():
    return stabilizer.get_text_state()

@app.post("/api/buffer/commit")
def buffer_commit(req: Optional[BufferSetRequest] = None):
    sign = req.text if req and req.text else None
    return stabilizer.force_commit_sign(sign)

@app.post("/api/buffer/space")
def buffer_space():
    stabilizer._apply_sign_action("SPACE")
    return stabilizer.get_text_state()

@app.post("/api/buffer/backspace")
def buffer_backspace():
    stabilizer._apply_sign_action("DEL")
    return stabilizer.get_text_state()

@app.post("/api/buffer/clear")
def buffer_clear():
    stabilizer.clear()
    return stabilizer.get_text_state()

@app.post("/api/buffer/set")
def buffer_set(req: BufferSetRequest):
    stabilizer.set_text(req.text)
    return stabilizer.get_text_state()


# --- Generative AI Endpoints ---

@app.post("/api/refine")
def refine_text(req: RefineRequest):
    """
    Transforms raw sign tokens into grammatical English according to chosen style.
    """
    result = llm_engine.refine_sentence(
        raw_text=req.raw_text,
        style=req.style or "casual",
        conversation_context=req.context
    )
    return result


@app.post("/api/chat/reply")
def chat_replies(req: ReplyRequest):
    """
    Generates 3 contextual replies for two-way communication.
    """
    replies = llm_engine.generate_replies(req.last_message, role=req.role or "hearing_partner")
    return {"suggestions": replies}


# --- Text-to-Speech Endpoints ---

@app.post("/api/tts/speak")
def speak(req: SpeakRequest):
    """Audible speech on device speakers."""
    ok = tts_engine.speak_direct(req.text, voice=req.voice, rate=req.rate or 175)
    return {"spoken": ok, "text": req.text}

@app.get("/api/tts/voices")
def get_voices():
    return {"voices": tts_engine.get_available_voices()}


# --- Translation History ---

@app.get("/api/history")
def get_history():
    return {"history": TRANSLATION_HISTORY}

@app.post("/api/history")
def add_history(item: HistoryItemRequest):
    entry = {
        "id": len(TRANSLATION_HISTORY) + 1,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_text": item.raw_text,
        "refined_text": item.refined_text,
        "style": item.style,
        "provider": item.provider
    }
    TRANSLATION_HISTORY.insert(0, entry)  # latest first
    return {"status": "saved", "entry": entry}

@app.delete("/api/history")
def clear_history():
    TRANSLATION_HISTORY.clear()
    return {"status": "cleared"}


# --- Settings Endpoint ---

@app.post("/api/settings")
def update_settings(req: SettingsUpdateRequest):
    if req.gemini_api_key is not None:
        llm_engine.update_api_key(req.gemini_api_key)
    if req.confidence_threshold is not None:
        stabilizer.confidence_threshold = req.confidence_threshold
    if req.stability_frames is not None:
        stabilizer.stability_threshold = max(2, min(req.stability_frames, 20))
        stabilizer.history = type(stabilizer.history)(maxlen=stabilizer.stability_threshold)

    return {
        "status": "updated",
        "gemini_active": bool(llm_engine.api_key),
        "confidence_threshold": stabilizer.confidence_threshold,
        "stability_frames": stabilizer.stability_threshold
    }


# --- WebSocket Stream ---

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)

            msg_type = data.get("type", "landmarks")
            if msg_type == "landmarks":
                raw_lms = data.get("landmarks", [])
                threshold = data.get("confidence_threshold", stabilizer.confidence_threshold)
                if len(raw_lms) == 21:
                    pred = gesture_classifier.predict_raw_landmarks(raw_lms, confidence_threshold=threshold)
                    stab_res = stabilizer.process_prediction(pred)
                    await websocket.send_json({
                        "type": "prediction_result",
                        "prediction": pred,
                        "stabilizer": stab_res
                    })
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Error: {e}")


# --- Static Files / SPA Mounting ---

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
@app.head("/")
def read_root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(content={"message": "SignBridge AI API is running. index.html not found."})
