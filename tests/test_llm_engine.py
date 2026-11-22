"""Unit tests for Generative AI & Prompt Engineering Engine."""

import unittest
from app.llm_engine import LLMEngine


class TestLLMEngine(unittest.TestCase):

    def setUp(self):
        # Offline engine: tests must not depend on a real API key or network quota.
        self.engine = LLMEngine(api_key="")
        self.engine.api_key = ""
        self.engine.client = None

    def test_refine_medical_styles(self):
        raw = "N E D   D O C T E R   W H E R E   H O S P I T A L"

        res_casual = self.engine.refine_sentence(raw, style="casual")
        self.assertIn("doctor", res_casual["refined_text"].lower())
        self.assertGreater(len(res_casual["suggestions"]), 0)

        res_formal = self.engine.refine_sentence(raw, style="formal")
        self.assertIn("require", res_formal["refined_text"].lower())

        res_concise = self.engine.refine_sentence(raw, style="concise")
        self.assertLess(len(res_concise["refined_text"]), len(res_formal["refined_text"]))

    def test_refine_greeting_styles(self):
        raw = "H E L O   M Y   N A M E   A L E X"

        res_casual = self.engine.refine_sentence(raw, style="casual")
        self.assertIn("alex", res_casual["refined_text"].lower())

        res_empathetic = self.engine.refine_sentence(raw, style="empathetic")
        self.assertTrue(any(w in res_empathetic["refined_text"].lower() for w in ["delighted", "glad", "meet"]))

    def test_conversational_reply_generation(self):
        replies = self.engine.generate_replies("Where is the hospital?", role="hearing_partner")
        self.assertEqual(len(replies), 3)
        self.assertTrue(any("hospital" in r.lower() or "blocks" in r.lower() or "clinic" in r.lower() for r in replies))


if __name__ == "__main__":
    unittest.main()
