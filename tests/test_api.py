"""Unit tests for FastAPI endpoints."""

import unittest
from fastapi.testclient import TestClient
from app.main import app, llm_engine


class TestSignBridgeAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Tests must not depend on a real API key, the network or live-model wording.
        cls._saved_llm_state = (llm_engine.api_key, llm_engine.client)
        llm_engine.api_key, llm_engine.client = "", None

    @classmethod
    def tearDownClass(cls):
        llm_engine.api_key, llm_engine.client = cls._saved_llm_state

    def test_status_endpoint(self):
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["classifier_ready"])
        self.assertGreaterEqual(data["classes_count"], 26)

    def test_metrics_endpoint(self):
        resp = self.client.get("/api/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("overall", data)
        self.assertGreater(data["overall"]["accuracy"], 0.95)

    def test_alphabet_endpoint(self):
        resp = self.client.get("/api/alphabet")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["alphabet"]), 26)

    def test_refine_endpoint(self):
        resp = self.client.post("/api/refine", json={
            "raw_text": "T H A N K   Y O U",
            "style": "formal"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("refined_text", data)
        self.assertIn("thank", data["refined_text"].lower())

    def test_buffer_endpoints(self):
        # Clear
        self.client.post("/api/buffer/clear")
        resp = self.client.get("/api/buffer")
        self.assertEqual(resp.json()["full_text"], "")

        # Set
        self.client.post("/api/buffer/set", json={"text": "HELLO WORLD"})
        resp = self.client.get("/api/buffer")
        self.assertEqual(resp.json()["full_text"], "HELLO WORLD")

        # Space
        self.client.post("/api/buffer/space")
        # Backspace
        self.client.post("/api/buffer/backspace")

    def test_chat_reply_endpoint(self):
        resp = self.client.post("/api/chat/reply", json={
            "last_message": "Hello, how are you?"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("suggestions", data)
        self.assertEqual(len(data["suggestions"]), 3)

    def test_history_endpoints(self):
        self.client.post("/api/history", json={
            "raw_text": "H E L O",
            "refined_text": "Hello there!",
            "style": "casual",
            "provider": "Test"
        })
        resp = self.client.get("/api/history")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(len(data["history"]), 0)


if __name__ == "__main__":
    unittest.main()
