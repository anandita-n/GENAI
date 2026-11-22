"""Unit tests for the J/Z motion detector using synthetic hand paths."""

import unittest

import numpy as np

from app.motion_detector import MotionTracker, handshape

FPS = 30.0
EXTENDED, CURLED = 1.9, 0.8


def make_hand(extended_fingers, offset=(0.0, 0.0)):
    """21 landmarks: wrist at origin, finger MCPs at distance 1, tips at ratio 1.9 or 0.8."""
    pts = np.zeros((21, 2))
    angles = {"index": 100, "middle": 90, "ring": 80, "pinky": 70}  # degrees, pointing "up" (-y)
    idx = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}
    for name, (mcp, tip) in idx.items():
        a = np.radians(angles[name])
        d = np.array([np.cos(a), -np.sin(a)])
        pts[mcp] = d
        pts[tip] = d * (EXTENDED if name in extended_fingers else CURLED)
    return pts + np.asarray(offset)


def interpolate(waypoints, n_per_leg=12):
    out = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        for t in np.linspace(0, 1, n_per_leg, endpoint=False):
            out.append((1 - t) * np.asarray(a) + t * np.asarray(b))
    out.append(np.asarray(waypoints[-1]))
    return out


def run(tracker, fingers, path, noise=0.0, seed=0):
    """Feed a fingertip path; the whole hand translates so the drawing tip follows it."""
    rng = np.random.default_rng(seed)
    base = make_hand(fingers)
    tip_idx = 20 if "pinky" in fingers and "index" not in fingers else 8
    results = []
    for i, p in enumerate(path):
        hand = make_hand(fingers, offset=np.asarray(p) - base[tip_idx])
        hand = hand + rng.normal(0, noise, hand.shape)
        results.append(tracker.update(hand, now=i / FPS))
    return [r for r in results if r]


Z_PATH = [(0, 0), (1.3, 0), (0, 1.3), (1.3, 1.3)]            # right, down-left, right
J_PATH = [(0, 0), (0, 1.1), (-0.6, 0.9)]                       # down, then hook sideways/up


class TestHandshape(unittest.TestCase):
    def test_i_shape_is_j(self):
        self.assertEqual(handshape(make_hand({"pinky"})), "J")

    def test_pointing_shape_is_z(self):
        self.assertEqual(handshape(make_hand({"index"})), "Z")

    def test_open_hand_is_neither(self):
        self.assertIsNone(handshape(make_hand({"index", "middle", "ring", "pinky"})))


class TestMotionTracker(unittest.TestCase):

    def test_detects_z(self):
        self.assertEqual(run(MotionTracker(), {"index"}, interpolate(Z_PATH), noise=0.01), ["Z"])

    def test_detects_mirrored_z(self):
        mirrored = [(-x, y) for x, y in Z_PATH]
        self.assertEqual(run(MotionTracker(), {"index"}, interpolate(mirrored), noise=0.01), ["Z"])

    def test_detects_j(self):
        self.assertEqual(run(MotionTracker(), {"pinky"}, interpolate(J_PATH), noise=0.01), ["J"])

    def test_detects_mirrored_j(self):
        mirrored = [(-x, y) for x, y in J_PATH]
        self.assertEqual(run(MotionTracker(), {"pinky"}, interpolate(mirrored), noise=0.01), ["J"])

    def test_static_pose_does_nothing(self):
        still = [(0.0, 0.0)] * 60
        self.assertEqual(run(MotionTracker(), {"pinky"}, still, noise=0.005), [])
        self.assertEqual(run(MotionTracker(), {"index"}, still, noise=0.005), [])

    def test_horizontal_wave_is_not_z(self):
        wave = interpolate([(0, 0), (1.3, 0), (0, 0), (1.3, 0)])
        self.assertEqual(run(MotionTracker(), {"index"}, wave), [])

    def test_vertical_down_and_back_is_not_j(self):
        down_up = interpolate([(0, 0), (0, 1.1), (0, 0)])
        self.assertEqual(run(MotionTracker(), {"pinky"}, down_up), [])

    def test_wrong_handshape_never_triggers(self):
        open_hand = {"index", "middle", "ring", "pinky"}
        self.assertEqual(run(MotionTracker(), open_hand, interpolate(Z_PATH)), [])

    def test_cooldown_prevents_double_trigger(self):
        path = interpolate(Z_PATH) + interpolate(Z_PATH)[:5]
        self.assertEqual(len(run(MotionTracker(), {"index"}, path)), 1)


class TestMotionThroughApi(unittest.TestCase):
    """Drawing a Z / J through /api/recognize/landmarks commits the letter."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app)

    def _draw(self, fingers, waypoints):
        from app.main import motion_tracker, stabilizer
        motion_tracker.reset()
        stabilizer.clear()
        base = make_hand(fingers)
        tip_idx = 20 if "index" not in fingers else 8
        committed, last_text = [], ""
        for p in interpolate(waypoints):
            hand = make_hand(fingers, offset=np.asarray(p) - base[tip_idx]) * 0.1 + 0.5
            body = {"landmarks": [{"x": float(x), "y": float(y), "z": 0.0} for x, y in hand]}
            data = self.client.post("/api/recognize/landmarks", json=body).json()
            if data["stabilizer"].get("committed_action"):
                committed.append(data["stabilizer"]["committed_action"])
            last_text = data["stabilizer"]["text_state"]["full_text"]
        return committed, last_text

    def test_z_is_committed_via_api(self):
        committed, text = self._draw({"index"}, Z_PATH)
        self.assertIn("Z", committed)
        self.assertTrue(text.endswith("Z"))

    def test_j_is_committed_via_api(self):
        committed, text = self._draw({"pinky"}, J_PATH)
        self.assertIn("J", committed)
        self.assertTrue(text.endswith("J"))

    def test_static_prediction_never_commits_j_or_z(self):
        from app.main import motion_tracker, stabilizer
        motion_tracker.reset()
        stabilizer.clear()
        hand = make_hand({"pinky"}) * 0.1 + 0.5
        body = {"landmarks": [{"x": float(x), "y": float(y), "z": 0.0} for x, y in hand]}
        for _ in range(30):
            data = self.client.post("/api/recognize/landmarks", json=body).json()
            self.assertNotIn(data["stabilizer"].get("committed_action"), ("J", "Z"))


if __name__ == "__main__":
    unittest.main()
