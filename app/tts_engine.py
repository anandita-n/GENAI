"""
SignBridge AI - Text-to-Speech (TTS) Engine
Provides audible speech synthesis on macOS via native NSSpeechSynthesizer / say
and pyttsx3 with audio file generation for web clients.
"""

import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional
from app.config import DATA_DIR


class TTSEngine:
    """Manages audio speech generation and playback."""

    def __init__(self):
        self.audio_cache_dir = DATA_DIR / "tts_cache"
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def speak_direct(self, text: str, voice: Optional[str] = "Samantha", rate: int = 175) -> bool:
        """
        Uses macOS native `say` utility for instant, reliable audible speech on host speakers.
        """
        clean_text = text.strip()
        if not clean_text:
            return False

        try:
            # macOS built-in say command
            cmd = ["say"]
            if voice:
                cmd.extend(["-v", voice])
            if rate:
                cmd.extend(["-r", str(rate)])
            cmd.append(clean_text)

            threading.Thread(target=subprocess.run, args=(cmd,), kwargs={"check": False}).start()
            return True
        except Exception as e:
            print(f"[TTSEngine] Direct speak error: {e}")
            return False

    def synthesize_wav(self, text: str, voice: Optional[str] = "Samantha") -> Optional[Path]:
        """
        Synthesizes text into an audio AIFF/WAV file that can be served to web clients.
        """
        clean_text = text.strip()
        if not clean_text:
            return None

        with self._lock:
            # Generate deterministic or unique filename
            safe_name = "".join(c for c in clean_text[:24] if c.isalnum() or c in " _-").strip()
            out_file = self.audio_cache_dir / f"tts_{hash(clean_text) & 0xFFFFFFFF}.aiff"

            try:
                cmd = ["say", "-o", str(out_file)]
                if voice:
                    cmd.extend(["-v", voice])
                cmd.append(clean_text)
                subprocess.run(cmd, check=True)
                return out_file
            except Exception as e:
                print(f"[TTSEngine] Synthesis error: {e}")
                return None

    def get_available_voices(self) -> List[Dict[str, str]]:
        """
        Lists available high-quality voices installed on macOS.
        """
        voices = []
        try:
            res = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True)
            for line in res.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    name = parts[0]
                    locale = parts[1]
                    voices.append({"name": name, "locale": locale})
        except Exception:
            voices = [
                {"name": "Samantha", "locale": "en_US"},
                {"name": "Alex", "locale": "en_US"},
                {"name": "Daniel", "locale": "en_GB"},
                {"name": "Karen", "locale": "en_AU"},
            ]
        return voices
