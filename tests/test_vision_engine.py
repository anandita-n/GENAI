"""Unit tests for the VLM image captioning engine (offline paths, no network)."""

import base64
import unittest

from app.vision_engine import VisionEngine, HANDSHAPE_DESCRIPTIONS


class _NoClientLLM:
    """Stand-in LLM engine with no Gemini client, forcing the offline path."""
    client = None
    model_name = "test-model"


class TestVisionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = VisionEngine(_NoClientLLM())

    def test_offline_caption_uses_top_candidate(self):
        result = self.engine.caption_sign(
            b"fake-image-bytes",
            [{"sign": "M", "confidence": 0.6}, {"sign": "N", "confidence": 0.3}],
        )
        self.assertEqual(result["vlm_sign"], "M")
        self.assertIn("M", result["caption"])
        self.assertTrue(result["agrees_with_classifier"])
        self.assertIn("offline", result["provider"].lower())

    def test_offline_caption_without_candidates(self):
        result = self.engine.caption_sign(b"fake-image-bytes", None)
        self.assertEqual(result["vlm_sign"], "NONE")
        self.assertEqual(result["candidates"], [])

    def test_invalid_candidates_are_dropped(self):
        result = self.engine.caption_sign(
            b"x", [{"sign": "??", "confidence": 0.9}, {"sign": "B", "confidence": "bad"}, {"sign": "A", "confidence": 0.5}]
        )
        self.assertEqual([c["sign"] for c in result["candidates"]], ["A"])

    def test_parse_json_handles_code_fences(self):
        parsed = VisionEngine._parse_json('```json\n{"caption": "a fist", "sign": "A"}\n```')
        self.assertEqual(parsed["sign"], "A")

    def test_every_letter_has_a_description(self):
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.assertIn(letter, HANDSHAPE_DESCRIPTIONS)


class TestCaptionEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app, vision_engine
        cls.client = TestClient(app)
        # Force offline mode so the test never calls the network.
        cls._saved_llm = vision_engine.llm
        vision_engine.llm = _NoClientLLM()

    @classmethod
    def tearDownClass(cls):
        from app.main import vision_engine
        vision_engine.llm = cls._saved_llm

    def test_caption_endpoint_offline(self):
        image = "data:image/jpeg;base64," + base64.b64encode(b"fake").decode()
        resp = self.client.post(
            "/api/caption",
            json={"image_base64": image, "candidates": [{"sign": "L", "confidence": 0.8}]},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["vlm_sign"], "L")
        self.assertIn("caption", data)

    def test_caption_endpoint_blank_frame_has_no_hand(self):
        import numpy as np
        import cv2
        ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 200, np.uint8))
        image = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        resp = self.client.post(
            "/api/caption",
            json={"image_base64": image, "candidates": [{"sign": "M", "confidence": 0.5}]},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["vlm_sign"], "NONE")
        self.assertIn("hand check", data["provider"])

    def test_caption_endpoint_rejects_bad_image(self):
        resp = self.client.post("/api/caption", json={"image_base64": "!!!not base64!!!"})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
