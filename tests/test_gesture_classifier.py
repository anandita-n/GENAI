"""Unit tests for Gesture Classifier & Temporal Stabilizer."""

import unittest
import numpy as np
from app.gesture_classifier import GestureClassifier, TemporalStabilizer
from app.asl_dataset import ASLKinematics
from app.landmark_extractor import LandmarkExtractor


class TestGestureClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier = GestureClassifier()

    def test_model_loaded(self):
        self.assertIsNotNone(self.classifier.clf)
        self.assertGreaterEqual(len(self.classifier.classes_), 26)

    def test_canonical_prediction_accuracy(self):
        """Test that canonical hand landmarks for key letters predict accurately."""
        test_signs = ["A", "B", "C", "L", "V", "Y", "SPACE"]
        for sign in test_signs:
            raw_pts = ASLKinematics.get_canonical_hand(sign)
            feats = LandmarkExtractor.normalize_landmarks(raw_pts)
            pred = self.classifier.predict_features(feats, confidence_threshold=0.5)
            self.assertEqual(pred["raw_sign"], sign, f"Expected {sign}, got {pred['raw_sign']}")
            self.assertGreater(pred["confidence"], 0.60)

    def test_temporal_stabilizer(self):
        stabilizer = TemporalStabilizer(stability_threshold=5, confidence_threshold=0.7)
        # Frames 1-4: should not commit yet
        for _ in range(4):
            res = stabilizer.process_prediction({"raw_sign": "H", "confidence": 0.95, "is_confident": True})
            self.assertIsNone(res["committed_action"])

        # Frame 5: reaches threshold of 5, commits 'H' ONCE
        res5 = stabilizer.process_prediction({"raw_sign": "H", "confidence": 0.95, "is_confident": True})
        self.assertEqual(res5["committed_action"], "H")
        self.assertEqual(res5["text_state"]["current_word"], "H")

        # Frame 6 & 7 (still holding H): locked out! Will NOT commit again!
        res6 = stabilizer.process_prediction({"raw_sign": "H", "confidence": 0.95, "is_confident": True})
        self.assertIsNone(res6["committed_action"])
        self.assertEqual(res6["text_state"]["current_word"], "H")

    def test_actions_space_and_del(self):
        stabilizer = TemporalStabilizer()
        stabilizer.set_text("HELO")
        # Apply space
        stabilizer._apply_sign_action("SPACE")
        self.assertEqual(stabilizer.current_sentence, "HELO")
        self.assertEqual(stabilizer.current_word, "")

        # Add more letters
        stabilizer._apply_sign_action("W")
        stabilizer._apply_sign_action("O")
        self.assertEqual(stabilizer.current_word, "WO")

        # Del
        stabilizer._apply_sign_action("DEL")
        self.assertEqual(stabilizer.current_word, "W")


if __name__ == "__main__":
    unittest.main()
