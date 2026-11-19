"""Unit tests for Landmark Extractor and Normalizer."""

import unittest
import numpy as np
from app.landmark_extractor import LandmarkExtractor


class TestLandmarkExtractor(unittest.TestCase):

    def setUp(self):
        # 21 3D points
        self.raw_landmarks = np.zeros((21, 3), dtype=np.float32)
        # Wrist at origin
        self.raw_landmarks[0] = [0.0, 0.0, 0.0]
        # Middle MCP
        self.raw_landmarks[9] = [0.0, -1.0, 0.0]
        # Random positions for others
        for i in range(1, 21):
            if i != 9:
                self.raw_landmarks[i] = [np.sin(i), -np.cos(i), 0.1 * i]

    def test_normalization_shape(self):
        features = LandmarkExtractor.normalize_landmarks(self.raw_landmarks)
        self.assertEqual(features.shape, (83,))
        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isinf(features).any())

    def test_translation_invariance(self):
        # Shift all landmarks by offset (+5.0, -3.2, 8.4)
        shifted = self.raw_landmarks + np.array([5.0, -3.2, 8.4], dtype=np.float32)
        feats_original = LandmarkExtractor.normalize_landmarks(self.raw_landmarks)
        feats_shifted = LandmarkExtractor.normalize_landmarks(shifted)
        # Should be identical within numerical tolerance
        np.testing.assert_allclose(feats_original, feats_shifted, atol=1e-5)

    def test_scale_invariance(self):
        # Scale all landmarks by 2.5x
        scaled = self.raw_landmarks * 2.5
        feats_original = LandmarkExtractor.normalize_landmarks(self.raw_landmarks)
        feats_scaled = LandmarkExtractor.normalize_landmarks(scaled)
        np.testing.assert_allclose(feats_original, feats_scaled, atol=1e-5)

    def test_client_landmarks_conversion(self):
        client_pts = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in self.raw_landmarks]
        feats = LandmarkExtractor.from_client_landmarks(client_pts)
        self.assertEqual(feats.shape, (83,))


if __name__ == "__main__":
    unittest.main()
