"""
SignBridge AI - Motion letter detector (J and Z)

J and Z are the only ASL letters signed with movement, so a single-frame classifier
cannot recognise them: J starts as the "I" handshape and Z starts as a pointing index
finger. This module tracks the fingertip path over roughly the last 1.6 seconds and
recognises:
  * J - "I" handshape (only the little finger extended) whose tip draws a downward
        stroke followed by a sideways hook.
  * Z - pointing handshape (index extended, middle/ring curled) whose tip draws
        right, then diagonal down-left, then right again (or the mirror image).

It works on any consistent 2D coordinates (normalised 0-1 or pixels) and is
mirror-invariant, so it does not matter whether the camera image is flipped.
The thresholds are heuristics measured on real hand landmarks; tune them against
your own camera if detection feels too eager or too shy.
"""

import time
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

WINDOW_SECONDS = 1.6
COOLDOWN_SECONDS = 1.5
MIN_FRAMES = 8

# Finger extension ratio = |tip - wrist| / |knuckle - wrist| (measured on real hands:
# extended fingers are ~1.7-1.9, curled fingers ~0.5-1.2).
EXTENDED_MIN = 1.5
CURLED_MAX = 1.3

# Landmark indices (MediaPipe): wrist=0, middle MCP=9
_FINGERS = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}


def _ratios(points: np.ndarray) -> dict:
    wrist = points[0]
    out = {}
    for name, (mcp, tip) in _FINGERS.items():
        base = np.linalg.norm(points[mcp] - wrist)
        out[name] = np.linalg.norm(points[tip] - wrist) / base if base > 1e-9 else 0.0
    return out


def handshape(points: np.ndarray) -> Optional[str]:
    """Returns 'J' (I-like pinky handshape), 'Z' (pointing handshape) or None."""
    r = _ratios(points)
    if (r["pinky"] >= EXTENDED_MIN and r["index"] <= CURLED_MAX
            and r["middle"] <= CURLED_MAX and r["ring"] <= CURLED_MAX):
        return "J"
    if (r["index"] >= EXTENDED_MIN and r["middle"] <= CURLED_MAX
            and r["ring"] <= CURLED_MAX and r["pinky"] <= EXTENDED_MIN):
        return "Z"
    return None


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    line = end - start
    norm = np.linalg.norm(line)
    if norm < 1e-9:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        dists = np.abs(line[0] * (points[:, 1] - start[1]) - line[1] * (points[:, 0] - start[0])) / norm
    idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left = _rdp(points[: idx + 1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def _segments(path: np.ndarray, epsilon: float = 0.2, min_len: float = 0.3) -> List[np.ndarray]:
    simple = _rdp(path, epsilon)
    segs = [simple[i + 1] - simple[i] for i in range(len(simple) - 1)]
    return [s for s in segs if np.linalg.norm(s) >= min_len]


def looks_like_z(path: np.ndarray) -> bool:
    """path: fingertip positions in hand-size units (y grows downward)."""
    segs = _segments(path)
    if len(segs) < 3:
        return False
    for i in range(len(segs) - 2):
        s1, s2, s3 = segs[i], segs[i + 1], segs[i + 2]
        sign = np.sign(s1[0])
        if sign == 0:
            continue
        top_ok = abs(s1[1]) <= 0.7 * abs(s1[0]) and np.linalg.norm(s1) >= 0.45
        bottom_ok = abs(s3[1]) <= 0.7 * abs(s3[0]) and np.linalg.norm(s3) >= 0.45 and np.sign(s3[0]) == sign
        diag_ok = (np.sign(s2[0]) == -sign and s2[1] > 0
                   and s2[1] >= 0.5 * abs(s2[0]) and np.linalg.norm(s2) >= 0.5)
        if top_ok and diag_ok and bottom_ok:
            return True
    return False


def looks_like_j(path: np.ndarray) -> bool:
    """A downward stroke followed by a sideways hook (curving back up or across)."""
    segs = _segments(path)
    for i in range(len(segs) - 1):
        down, hook = segs[i], segs[i + 1]
        down_len, hook_len = np.linalg.norm(down), np.linalg.norm(hook)
        if not (down[1] > 0 and abs(down[1]) >= 1.2 * abs(down[0]) and down_len >= 0.5):
            continue
        if hook_len < 0.3 or abs(hook[0]) < 0.4 * hook_len:
            continue  # the return stroke must have a real sideways component
        cos = float(np.dot(down, hook) / (down_len * hook_len))
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        if 50 <= angle <= 150:
            return True
    return False


class MotionTracker:
    """Feed it each frame's 21 hand points; it returns 'J' or 'Z' when one is drawn."""

    def __init__(self, window_seconds: float = WINDOW_SECONDS, cooldown: float = COOLDOWN_SECONDS):
        self.window = window_seconds
        self.cooldown = cooldown
        self._shape: Optional[str] = None
        self._buf: deque = deque()  # (time, x, y, hand_size)
        self._last_trigger = float("-inf")

    def reset(self):
        self._shape = None
        self._buf.clear()

    def update(self, landmarks: Sequence[Sequence[float]], now: Optional[float] = None) -> Optional[str]:
        now = time.time() if now is None else now
        pts = np.asarray([[p[0], p[1]] for p in landmarks], dtype=np.float64)
        if pts.shape != (21, 2):
            self.reset()
            return None

        hand_size = float(np.linalg.norm(pts[0] - pts[9]))
        shape = handshape(pts) if hand_size > 1e-9 else None
        if shape is None:
            self.reset()
            return None
        if shape != self._shape:
            self.reset()
            self._shape = shape

        tip = pts[20] if shape == "J" else pts[8]
        self._buf.append((now, tip[0], tip[1], hand_size))
        while self._buf and now - self._buf[0][0] > self.window:
            self._buf.popleft()

        if len(self._buf) < MIN_FRAMES or now - self._last_trigger < self.cooldown:
            return None

        arr = np.array(self._buf)
        size = float(np.median(arr[:, 3]))
        path = arr[:, 1:3] / size
        path = self._smooth(path)

        detected = looks_like_j(path) if shape == "J" else looks_like_z(path)
        if detected:
            self._last_trigger = now
            self.reset()
            return shape
        return None

    @staticmethod
    def _smooth(path: np.ndarray, k: int = 3) -> np.ndarray:
        if len(path) < k:
            return path
        kernel = np.ones(k) / k
        sm = np.column_stack([np.convolve(path[:, c], kernel, mode="valid") for c in range(2)])
        return np.vstack([path[:1], sm, path[-1:]])
