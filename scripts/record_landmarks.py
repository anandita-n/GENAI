"""
SignBridge AI - Interactive Live Landmark Data Collector
Allows users to record custom hand gestures from their webcam and save them
into data/custom_landmarks.json for continuous training.
"""

import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np

from app.config import DATA_DIR, ALL_CLASSES
from app.landmark_extractor import LandmarkExtractor


def record_sign(target_sign: str, samples_to_record: int = 50):
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"\n=======================================================")
    print(f" Recording Live Landmarks for Sign: [{target_sign}]")
    print(f" Target: {samples_to_record} valid hand frames")
    print(f" Press 'r' to start recording, 'q' to quit.")
    print(f"=======================================================\n")

    recorded = []
    is_recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display = frame.copy()

        extracted = extractor.extract_from_frame(frame)
        status_text = f"Sign: {target_sign} | Collected: {len(recorded)}/{samples_to_record}"

        if extracted:
            # Draw bbox and points
            bbox = extracted["bbox"]
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            for pt in extracted["landmarks_pixel"]:
                cv2.circle(display, tuple(pt), 4, (255, 0, 0), -1)

            if is_recording:
                recorded.append(extracted["features"])
                status_text += " [RECORDING...]"
                if len(recorded) >= samples_to_record:
                    print(f"Successfully recorded {samples_to_record} samples for {target_sign}!")
                    break
        else:
            status_text += " [No Hand Detected]"

        cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("SignBridge Live Data Collector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = True

    cap.release()
    cv2.destroyAllWindows()

    if recorded:
        out_file = DATA_DIR / f"recorded_{target_sign}_{int(time.time())}.json"
        with open(out_file, "w") as f:
            json.dump({"sign": target_sign, "features": recorded}, f)
        print(f"Saved {len(recorded)} samples to {out_file}")


if __name__ == "__main__":
    sign = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    record_sign(sign, count)
