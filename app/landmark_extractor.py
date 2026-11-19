"""
SignBridge AI - Landmark Extractor and Normalizer
Extracts and normalizes 21 3D hand landmarks from MediaPipe.
Provides full scale, translation, 2D rotation (tilt), and handedness invariance
plus rich geometric biomechanical features.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.config import HAND_LANDMARKER_MODEL_PATH


class LandmarkExtractor:
    """Extracts and preprocesses 21 hand landmarks using MediaPipe LiteRT tasks."""

    FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    FINGER_PIPS = [2, 6, 10, 14, 18]
    FINGER_MCPS = [1, 5, 9, 13, 17]

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(HAND_LANDMARKER_MODEL_PATH)
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[LandmarkExtractor] Warning: Detector initialization failed ({e}). "
                  f"Client-side landmark processing will still work.")
            self.detector = None

    def extract_from_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Processes an OpenCV BGR image frame and extracts hand landmarks and bounding box.
        """
        if self.detector is None or frame_bgr is None:
            return None

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)
        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None

        # Take first detected hand
        landmarks = result.hand_landmarks[0]
        h, w, _ = frame_bgr.shape

        raw_points = []
        pixel_points = []
        for lm in landmarks:
            raw_points.append([lm.x, lm.y, lm.z])
            pixel_points.append([int(lm.x * w), int(lm.y * h)])

        raw_np = np.array(raw_points, dtype=np.float32)  # shape (21, 3)
        pixel_np = np.array(pixel_points, dtype=np.int32)

        # Compute bounding box
        x_min, y_min = np.min(pixel_np, axis=0)
        x_max, y_max = np.max(pixel_np, axis=0)

        feature_vector = self.normalize_landmarks(raw_np)

        return {
            "landmarks_raw": raw_points,
            "landmarks_pixel": pixel_points,
            "features": feature_vector.tolist(),
            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
        }

    @classmethod
    def normalize_landmarks(cls, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalizes 21 3D landmarks into a canonical invariant feature vector:
        1. Translation invariance: Wrist (index 0) becomes origin (0, 0, 0).
        2. Scale normalization: Scale coordinates by distance between Wrist and Middle MCP (9).
        3. 2D In-plane Rotation: Rotate landmarks so Middle MCP (9) aligns strictly with (0, -1, 0).
        4. Handedness/Mirror Invariance: Canonicalize so Index is always negative X and Pinky is positive X.
        5. Biomechanical Geometric features:
           - Tip-to-wrist normalized distances (5)
           - Pairwise fingertip distances (10 combinations)
           - Curl angles/ratios for all 5 fingers (5)
        Total features: 63 (normalized coords) + 20 (geometric features) = 83 features.
        """
        lm = np.array(landmarks, dtype=np.float32).copy()
        if lm.shape != (21, 3):
            lm = lm.reshape(21, 3)

        # 1. Translation: center at wrist (landmark 0)
        wrist = lm[0].copy()
        centered = lm - wrist

        # 2. Scale normalization based on Middle MCP (landmark 9) distance in 2D
        ref_dist = np.linalg.norm(centered[9, :2])
        if ref_dist < 1e-5:
            ref_dist = np.linalg.norm(centered[9])
        if ref_dist < 1e-5:
            ref_dist = np.max(np.linalg.norm(centered, axis=1))
        if ref_dist < 1e-5:
            ref_dist = 1.0

        scaled = centered / ref_dist

        # 3. In-plane rotation alignment: align Wrist->MiddleMCP with (0, -1)
        vx, vy = scaled[9, 0], scaled[9, 1]
        angle = np.arctan2(vx, -vy)
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        scaled[:, :2] = scaled[:, :2] @ R.T

        # 4. Handedness / Mirror reflection: ensure Index MCP (5) is negative X relative to Pinky MCP (17)
        if scaled[5, 0] > scaled[17, 0]:
            scaled[:, 0] = -scaled[:, 0]

        coords_flat = scaled.flatten()  # 63 features

        # 5. Biomechanical Geometric features
        geo_features = []

        # (a) Distances from wrist to 5 fingertips
        for tip_idx in cls.FINGER_TIPS:
            d = np.linalg.norm(scaled[tip_idx])
            geo_features.append(d)

        # (b) Pairwise fingertip distances (10 pairs: C(5, 2))
        tips = [scaled[i] for i in cls.FINGER_TIPS]
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                d = np.linalg.norm(tips[i] - tips[j])
                geo_features.append(d)

        # (c) Extension ratio for each finger
        for mcp_idx, pip_idx, tip_idx in zip(cls.FINGER_MCPS, cls.FINGER_PIPS, cls.FINGER_TIPS):
            dip_idx = pip_idx + 1 if pip_idx < 18 else 19
            direct_dist = np.linalg.norm(scaled[tip_idx] - scaled[mcp_idx])
            seg1 = np.linalg.norm(scaled[pip_idx] - scaled[mcp_idx])
            seg2 = np.linalg.norm(scaled[dip_idx] - scaled[pip_idx])
            seg3 = np.linalg.norm(scaled[tip_idx] - scaled[dip_idx])
            total_seg = seg1 + seg2 + seg3
            ratio = direct_dist / (total_seg + 1e-6)
            geo_features.append(ratio)

        geo_np = np.array(geo_features, dtype=np.float32)

        return np.concatenate([coords_flat, geo_np])

    @classmethod
    def from_client_landmarks(cls, raw_landmarks: List[Dict[str, float]]) -> np.ndarray:
        """
        Converts MediaPipe JS landmark list ([{'x':..., 'y':..., 'z':...}, ...])
        into the normalized 83-dimensional feature vector.
        """
        arr = np.zeros((21, 3), dtype=np.float32)
        for i, pt in enumerate(raw_landmarks):
            arr[i, 0] = pt.get("x", 0.0)
            arr[i, 1] = pt.get("y", 0.0)
            arr[i, 2] = pt.get("z", 0.0)
        return cls.normalize_landmarks(arr)
