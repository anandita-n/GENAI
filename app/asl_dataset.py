"""
SignBridge AI - ASL Biomechanical Dataset Generator & Kinematic Hand Model
Generates realistic 3D hand landmark distributions for 26 ASL alphabet signs
plus functional gestures (SPACE, DEL, CLEAR) based on canonical ASL finger configurations,
anthropometric finger length variances, 3D rotations, and joint perturbation noise.
"""

from typing import Dict, List, Tuple
import numpy as np
from app.config import ASL_ALPHABET, SPECIAL_GESTURES, ALL_CLASSES
from app.landmark_extractor import LandmarkExtractor


class ASLKinematics:
    """Defines canonical 3D finger kinematics for ASL gestures."""

    @staticmethod
    def get_canonical_hand(sign: str) -> np.ndarray:
        """
        Returns canonical (21, 3) landmarks for a right hand executing `sign`.
        Wrist is at (0, 0, 0). Up is -Y, Right is +X, Forward is -Z.
        """
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = [0.0, 0.0, 0.0]  # Wrist

        # Base MCP knuckle positions (wrist to base of each finger)
        # 1: Thumb CMC, 5: Index MCP, 9: Middle MCP, 13: Ring MCP, 17: Pinky MCP
        pts[1] = [-0.25, -0.22, -0.05]  # Thumb CMC
        pts[2] = [-0.42, -0.45, -0.10]  # Thumb MCP
        pts[5] = [-0.28, -0.95, -0.05]  # Index MCP
        pts[9] = [0.00, -1.00, 0.00]    # Middle MCP
        pts[13] = [0.25, -0.92, 0.02]   # Ring MCP
        pts[17] = [0.45, -0.80, 0.05]   # Pinky MCP

        def set_finger(mcp_idx: int, dir_vec: np.ndarray, curl_amount: float, spread: float = 0.0):
            """
            Builds PIP, DIP, TIP from MCP.
            dir_vec: unit vector for extension.
            curl_amount: 0.0 (fully straight) to 1.0 (fully curled into palm).
            spread: lateral angle in radians.
            """
            bone_lengths = [0.35, 0.25, 0.22]  # PIP, DIP, TIP
            base = pts[mcp_idx].copy()

            # Lateral spread rotation matrix around Z-axis
            c, s = np.cos(spread), np.sin(spread)
            R_spread = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            d = R_spread @ dir_vec

            # Normal straight orientation
            p_curr = base.copy()
            for step, blen in enumerate(bone_lengths):
                next_idx = mcp_idx + 1 + step
                if curl_amount < 0.2:
                    # Straight
                    p_curr = p_curr + d * blen
                elif curl_amount > 0.8:
                    # Fully curled inward towards palm (+Z and +Y)
                    curl_phase = (step + 1) * 1.05
                    p_curr = p_curr + np.array([
                        d[0] * 0.2,
                        blen * np.cos(curl_phase) * 0.7,
                        blen * np.sin(curl_phase) * 0.9 + 0.1
                    ], dtype=np.float32)
                else:
                    # Half-curled (e.g. C, E, X)
                    p_curr = p_curr + np.array([
                        d[0] * 0.5 * blen,
                        d[1] * (1.0 - curl_amount) * blen,
                        curl_amount * blen * 0.8
                    ], dtype=np.float32)
                pts[next_idx] = p_curr

        def set_thumb(thumb_state: str):
            """
            States:
            'up': alongside index finger (A)
            'across_palm': folded over fingers (B, S, etc.)
            'open_wide': extended out at 90 deg (L, Y, SPACE)
            'tucked_under_index': (N, M, T)
            'pinched_index': touching index tip (O, F)
            'pinched_down': touching middle/ring (D)
            """
            if thumb_state == 'up':
                pts[3] = [-0.38, -0.70, -0.05]
                pts[4] = [-0.30, -0.92, -0.05]  # Thumb tip resting against index MCP
            elif thumb_state == 'across_palm':
                pts[3] = [-0.15, -0.55, 0.20]
                pts[4] = [0.10, -0.58, 0.25]   # Folded over middle of palm
            elif thumb_state == 'open_wide':
                pts[3] = [-0.65, -0.55, -0.05]
                pts[4] = [-0.90, -0.65, -0.08]  # Extended out to the left
            elif thumb_state == 'tucked_under_index':
                pts[3] = [-0.15, -0.65, 0.15]
                pts[4] = [-0.05, -0.75, 0.15]   # Poking up between fingers
            elif thumb_state == 'pinched_index':
                pts[3] = [-0.25, -0.70, 0.20]
                pts[4] = [-0.15, -0.85, 0.25]
            elif thumb_state == 'pinched_middle':
                pts[3] = [-0.15, -0.65, 0.20]
                pts[4] = [0.05, -0.75, 0.25]
            elif thumb_state == 'flat_open':
                pts[3] = [-0.55, -0.50, 0.0]
                pts[4] = [-0.75, -0.60, 0.0]
            else:
                pts[3] = [-0.35, -0.60, 0.0]
                pts[4] = [-0.30, -0.75, 0.1]

        # Finger directions (canonical up is [0, -1, 0])
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        side_right = np.array([1.0, -0.2, 0.0], dtype=np.float32)

        # Build each sign configuration
        if sign == 'A':
            # Fist closed, thumb straight up along index
            set_thumb('up')
            set_finger(5, up, curl_amount=1.0)
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'B':
            # 4 fingers straight up touching together, thumb tucked across palm
            set_thumb('across_palm')
            set_finger(5, up, curl_amount=0.0, spread=-0.05)
            set_finger(9, up, curl_amount=0.0, spread=0.0)
            set_finger(13, up, curl_amount=0.0, spread=0.04)
            set_finger(17, up, curl_amount=0.0, spread=0.08)

        elif sign == 'C':
            # All fingers curved into a C cup shape
            set_thumb('open_wide')
            pts[4] = [-0.45, -0.45, 0.35]
            set_finger(5, up, curl_amount=0.55, spread=-0.1)
            set_finger(9, up, curl_amount=0.55, spread=0.0)
            set_finger(13, up, curl_amount=0.55, spread=0.08)
            set_finger(17, up, curl_amount=0.55, spread=0.15)

        elif sign == 'D':
            # Index straight up, middle/ring/pinky curled touching thumb
            set_thumb('pinched_middle')
            set_finger(5, up, curl_amount=0.0, spread=0.0)
            set_finger(9, up, curl_amount=0.9)
            set_finger(13, up, curl_amount=0.95)
            set_finger(17, up, curl_amount=0.95)
            # Ensure middle tip touches thumb tip
            pts[12] = pts[4] + [0.02, 0.02, 0.0]

        elif sign == 'E':
            # All fingers tightly curled with tips resting on top edge of thumb
            set_thumb('across_palm')
            pts[4] = [-0.10, -0.60, 0.15]
            set_finger(5, up, curl_amount=0.9)
            set_finger(9, up, curl_amount=0.9)
            set_finger(13, up, curl_amount=0.9)
            set_finger(17, up, curl_amount=0.9)
            # Move tips to thumb line
            for tip_idx in [8, 12, 16, 20]:
                pts[tip_idx][1] = -0.65
                pts[tip_idx][2] = 0.20

        elif sign == 'F':
            # Index and thumb touch in circle, middle/ring/pinky extended up spread
            set_thumb('pinched_index')
            pts[8] = pts[4].copy()  # Index tip meets thumb tip
            pts[7] = pts[8] + [-0.1, 0.15, 0.05]
            pts[6] = pts[5] + [-0.1, -0.2, 0.15]
            set_finger(9, up, curl_amount=0.0, spread=0.05)
            set_finger(13, up, curl_amount=0.0, spread=0.15)
            set_finger(17, up, curl_amount=0.0, spread=0.25)

        elif sign == 'G':
            # Index pointing forward/sideways horizontally, thumb parallel, others in fist
            pts[3] = [-0.25, -0.55, -0.10]
            pts[4] = [0.10, -0.60, -0.10]
            pts[6] = [-0.10, -0.85, -0.10]
            pts[7] = [0.15, -0.85, -0.10]
            pts[8] = [0.45, -0.85, -0.10]  # Pointing to right
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'H':
            # Index and middle pointing horizontal together, thumb tucked, others curled
            pts[4] = [-0.15, -0.55, 0.15]
            # Index
            pts[6] = [-0.10, -0.85, -0.05]
            pts[7] = [0.15, -0.85, -0.05]
            pts[8] = [0.45, -0.85, -0.05]
            # Middle
            pts[10] = [0.05, -0.92, -0.05]
            pts[11] = [0.25, -0.92, -0.05]
            pts[12] = [0.48, -0.92, -0.05]
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'I':
            # Pinky straight up, all others in closed fist, thumb across
            set_thumb('across_palm')
            set_finger(5, up, curl_amount=1.0)
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=0.0, spread=0.1)

        elif sign == 'J':
            # Pinky extended and curved in J motion
            set_thumb('across_palm')
            set_finger(5, up, curl_amount=1.0)
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            pts[18] = [0.48, -1.05, 0.0]
            pts[19] = [0.55, -1.20, -0.1]
            pts[20] = [0.40, -1.30, -0.2]  # Hooked tip

        elif sign == 'K':
            # Index straight up, middle finger forward at 45 deg, thumb between them
            pts[4] = [-0.15, -0.80, 0.10]
            set_finger(5, up, curl_amount=0.0, spread=-0.05)
            # Middle angled forward
            pts[10] = [0.0, -1.15, 0.15]
            pts[11] = [0.0, -1.30, 0.30]
            pts[12] = [0.0, -1.45, 0.45]
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'L':
            # Index up, thumb at 90 deg ("L")
            set_thumb('open_wide')
            set_finger(5, up, curl_amount=0.0, spread=0.0)
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'M':
            # Fist with thumb tucked under 3 fingers (index, middle, ring)
            set_thumb('tucked_under_index')
            pts[4] = [0.20, -0.65, 0.20]  # Poking out under ring finger
            set_finger(5, up, curl_amount=0.95)
            set_finger(9, up, curl_amount=0.95)
            set_finger(13, up, curl_amount=0.95)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'N':
            # Fist with thumb tucked under 2 fingers (index, middle)
            set_thumb('tucked_under_index')
            pts[4] = [0.05, -0.68, 0.20]  # Poking out between middle and ring
            set_finger(5, up, curl_amount=0.95)
            set_finger(9, up, curl_amount=0.95)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'O':
            # All fingers curled to meet thumb tip in O circle
            set_thumb('pinched_index')
            pts[4] = [-0.15, -0.75, 0.25]
            pts[8] = pts[4] + [0.0, 0.05, 0.0]
            pts[12] = pts[4] + [0.08, 0.05, 0.0]
            pts[16] = pts[4] + [0.15, 0.05, 0.0]
            pts[20] = pts[4] + [0.22, 0.05, 0.0]
            # Intermediate joints curved
            for mcp, tip in zip([5, 9, 13, 17], [8, 12, 16, 20]):
                pts[mcp+1] = pts[mcp] + [0.0, -0.25, 0.30]
                pts[mcp+2] = pts[tip] + [0.0, -0.15, 0.15]

        elif sign == 'P':
            # Similar to K but pointed downwards (-Y inverted)
            pts[4] = [-0.15, -0.50, 0.10]
            pts[6] = [-0.25, -0.65, 0.0]
            pts[7] = [-0.25, -0.40, 0.0]
            pts[8] = [-0.25, -0.15, 0.0]  # Downward
            pts[10] = [0.0, -0.70, 0.15]
            pts[11] = [0.0, -0.50, 0.30]
            pts[12] = [0.0, -0.30, 0.40]
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'Q':
            # Similar to G but pointing downwards
            pts[3] = [-0.25, -0.45, -0.10]
            pts[4] = [-0.10, -0.25, -0.10]  # Thumb down
            pts[6] = [-0.28, -0.65, -0.05]
            pts[7] = [-0.28, -0.45, -0.05]
            pts[8] = [-0.28, -0.20, -0.05]  # Index down
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'R':
            # Index and middle straight up and crossed over
            set_thumb('across_palm')
            # Index crosses behind or to the right of middle
            pts[6] = [-0.22, -1.25, 0.05]
            pts[7] = [-0.10, -1.45, 0.05]
            pts[8] = [0.08, -1.65, 0.05]
            # Middle crosses in front to the left
            pts[10] = [-0.05, -1.28, -0.05]
            pts[11] = [-0.18, -1.50, -0.05]
            pts[12] = [-0.25, -1.70, -0.05]
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'S':
            # Fist with thumb wrapped across front of fingers
            set_thumb('across_palm')
            pts[4] = [0.15, -0.72, 0.30]  # Wrapped across middle and ring
            set_finger(5, up, curl_amount=1.0)
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'T':
            # Fist with thumb poking up between index and middle fingers
            set_thumb('tucked_under_index')
            pts[4] = [-0.12, -0.80, 0.18]  # Between index and middle
            set_finger(5, up, curl_amount=1.0)
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'U':
            # Index and middle straight up touching together, others folded
            set_thumb('across_palm')
            set_finger(5, up, curl_amount=0.0, spread=-0.03)
            set_finger(9, up, curl_amount=0.0, spread=0.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'V':
            # Index and middle straight up spread in peace sign (V)
            set_thumb('across_palm')
            set_finger(5, up, curl_amount=0.0, spread=-0.22)
            set_finger(9, up, curl_amount=0.0, spread=0.22)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'W':
            # Index, middle, ring straight up spread apart
            set_thumb('across_palm')
            pts[4] = [0.35, -0.65, 0.20]  # Touching pinky
            set_finger(5, up, curl_amount=0.0, spread=-0.25)
            set_finger(9, up, curl_amount=0.0, spread=0.0)
            set_finger(13, up, curl_amount=0.0, spread=0.25)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'X':
            # Index hooked like a claw/bent, others folded in fist
            set_thumb('across_palm')
            pts[6] = [-0.28, -1.25, 0.0]
            pts[7] = [-0.28, -1.45, 0.20]  # Bends forward
            pts[8] = [-0.28, -1.30, 0.35]  # Hooked tip
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'Y':
            # Thumb and pinky extended wide, middle 3 curled (hang loose)
            set_thumb('open_wide')
            set_finger(5, up, curl_amount=1.0)
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=0.0, spread=0.35)

        elif sign == 'Z':
            # Index pointing forward tracing Z, others curled
            set_thumb('across_palm')
            pts[6] = [-0.28, -1.25, -0.10]
            pts[7] = [-0.20, -1.50, -0.20]
            pts[8] = [-0.10, -1.75, -0.30]  # Dynamic angle forward
            set_finger(9, up, curl_amount=1.0)
            set_finger(13, up, curl_amount=1.0)
            set_finger(17, up, curl_amount=1.0)

        elif sign == 'SPACE':
            # Flat open palm pushed forward
            set_thumb('flat_open')
            set_finger(5, up, curl_amount=0.0, spread=-0.12)
            set_finger(9, up, curl_amount=0.0, spread=-0.02)
            set_finger(13, up, curl_amount=0.0, spread=0.08)
            set_finger(17, up, curl_amount=0.0, spread=0.18)

        elif sign == 'DEL':
            # Thumb pointing left / hand rotated side gesture
            set_thumb('open_wide')
            pts[4] = [-1.0, -0.30, 0.0]
            set_finger(5, up, curl_amount=0.85)
            set_finger(9, up, curl_amount=0.85)
            set_finger(13, up, curl_amount=0.85)
            set_finger(17, up, curl_amount=0.85)

        elif sign == 'CLEAR':
            # Open hand tilted horizontally / wave
            set_thumb('flat_open')
            pts[4] = [-0.5, -0.3, 0.0]
            set_finger(5, side_right, curl_amount=0.0)
            set_finger(9, side_right, curl_amount=0.0)
            set_finger(13, side_right, curl_amount=0.0)
            set_finger(17, side_right, curl_amount=0.0)

        else:
            # Default relaxed neutral hand
            set_thumb('up')
            set_finger(5, up, curl_amount=0.3)
            set_finger(9, up, curl_amount=0.35)
            set_finger(13, up, curl_amount=0.4)
            set_finger(17, up, curl_amount=0.45)

        return pts

    @classmethod
    def generate_sample(cls, sign: str, noise_level: float = 0.035) -> np.ndarray:
        """
        Generates a perturbed sample for `sign` with 3D rotation, scaling,
        bone length variance, and Gaussian joint noise.
        """
        canonical = cls.get_canonical_hand(sign).copy()

        # 1. Anthropometric bone length variance (per finger stretch)
        scale_factors = 1.0 + np.random.uniform(-0.08, 0.08, size=5)
        for f_idx, s in enumerate(scale_factors):
            mcp_idx = 1 if f_idx == 0 else (5 + (f_idx - 1) * 4)
            for offset in range(1, 4):
                canonical[mcp_idx + offset] += (canonical[mcp_idx + offset] - canonical[mcp_idx]) * (s - 1.0)

        # 2. 3D Rotation (pitch, yaw, roll within natural hand orientation limits)
        roll = np.radians(np.random.normal(0, 10))   # Z-axis
        pitch = np.radians(np.random.normal(0, 8))  # X-axis
        yaw = np.radians(np.random.normal(0, 12))   # Y-axis

        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        rotated = canonical @ R.T

        # 3. Overall scale variation (hand distance from camera)
        scale = np.random.uniform(0.75, 1.35)
        rotated *= scale

        # 4. Translation offset (hand position in video frame)
        offset = np.random.normal(0, 0.25, size=(1, 3))
        translated = rotated + offset

        # 5. Gaussian joint jitter
        noise = np.random.normal(0, noise_level * scale, size=translated.shape)
        final_landmarks = translated + noise

        # Flip augmentation (50% probability for left hand)
        if np.random.rand() > 0.5:
            final_landmarks[:, 0] = -final_landmarks[:, 0]

        return final_landmarks.astype(np.float32)

    @classmethod
    def generate_dataset(cls, samples_per_class: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes a full normalized dataset of ASL gestures across ALL_CLASSES.
        Returns:
            X: shape (num_samples, 83) feature vectors
            y: shape (num_samples,) class label strings
        """
        features_list = []
        labels_list = []

        for label in ALL_CLASSES:
            for _ in range(samples_per_class):
                raw_pts = cls.generate_sample(label)
                feats = LandmarkExtractor.normalize_landmarks(raw_pts)
                features_list.append(feats)
                labels_list.append(label)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=object)
        return X, y
