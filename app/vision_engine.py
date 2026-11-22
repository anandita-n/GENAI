"""
SignBridge AI - Vision-Language Model (VLM) Image Captioning Engine
Takes a camera frame of a hand sign plus the fast landmark classifier's top
candidate letters, and asks a vision-language model (Gemini) to:
  1. caption the image (describe the handshape in natural language), and
  2. choose which candidate letter the image really shows (disambiguating
     look-alike signs such as M/N, U/V, D/1, K/P).

If no API key is configured, or the VLM call fails, an offline caption is built
from the classifier's candidates so the endpoint always returns something useful.
"""

import json
import re
from typing import Any, Dict, List, Optional

from app.config import ALL_CLASSES

# Plain-language description of each ASL handshape. Used both to ground the
# VLM prompt and to build the offline fallback caption.
HANDSHAPE_DESCRIPTIONS: Dict[str, str] = {
    "A": "a closed fist with the thumb resting against the side of the index finger",
    "B": "a flat hand with fingers together pointing up and the thumb folded across the palm",
    "C": "a curved hand forming the shape of the letter C",
    "D": "the index finger pointing up while the other fingers and thumb touch to form a circle",
    "E": "fingers bent down with the fingertips touching the thumb, like a claw-shaped fist",
    "F": "the index finger and thumb touching in a circle while the other three fingers stay extended",
    "G": "the index finger and thumb pointing sideways, parallel to each other",
    "H": "the index and middle fingers extended sideways together",
    "I": "a closed fist with only the little finger extended upward",
    "J": "the little finger extended, tracing the letter J in the air",
    "K": "the index and middle fingers up in a V with the thumb between them",
    "L": "the index finger pointing up and the thumb out, forming the letter L",
    "M": "a fist with the thumb tucked under the index, middle and ring fingers",
    "N": "a fist with the thumb tucked under the index and middle fingers",
    "O": "all fingers curved to meet the thumb, forming the letter O",
    "P": "like K, but pointing downward",
    "Q": "like G, but pointing downward",
    "R": "the index and middle fingers crossed",
    "S": "a closed fist with the thumb across the front of the fingers",
    "T": "a fist with the thumb tucked between the index and middle fingers",
    "U": "the index and middle fingers extended upward, held together",
    "V": "the index and middle fingers extended upward and spread apart",
    "W": "the index, middle and ring fingers extended and spread apart",
    "X": "the index finger bent into a hook",
    "Y": "the thumb and little finger extended with the other fingers curled",
    "Z": "the index finger tracing the letter Z in the air",
    "SPACE": "a gesture meaning a space between words",
    "DEL": "a gesture meaning delete the last character",
    "CLEAR": "a gesture meaning clear all text",
}


class VisionEngine:
    """VLM-based sign image captioner with an offline fallback."""

    def __init__(self, llm_engine):
        # Reuse the LLM engine's Gemini client and model so that updating the
        # API key from the settings dialog also updates the vision engine.
        self.llm = llm_engine

    # ------------------------------------------------------------------ public
    def caption_sign(
        self,
        image_bytes: bytes,
        candidates: Optional[List[Dict[str, Any]]] = None,
        mime_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """
        Returns:
            {
              "caption": str,            # natural-language description of the image
              "vlm_sign": str,           # letter the VLM picked
              "agrees_with_classifier": bool,
              "reasoning": str,
              "candidates": [...],       # classifier candidates that were supplied
              "provider": str
            }
        """
        candidates = self._clean_candidates(candidates)

        client = getattr(self.llm, "client", None)
        if client is not None:
            try:
                result = self._caption_with_gemini(client, image_bytes, candidates, mime_type)
                if result:
                    return result
            except Exception as e:
                print(f"[VisionEngine] Gemini vision call failed: {e}. Using offline caption.")

        return self._offline_caption(candidates)

    # ----------------------------------------------------------------- Gemini
    def _caption_with_gemini(
        self, client, image_bytes: bytes, candidates: List[Dict[str, Any]], mime_type: str
    ) -> Optional[Dict[str, Any]]:
        from google.genai import types

        prompt = self._build_prompt(candidates)
        response = client.models.generate_content(
            model=self.llm.model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        parsed = self._parse_json(response.text or "")
        if not parsed:
            return None

        vlm_sign = str(parsed.get("sign", "")).strip().upper()
        if vlm_sign not in ALL_CLASSES:
            vlm_sign = candidates[0]["sign"] if candidates else "NONE"

        top = candidates[0]["sign"] if candidates else None
        return {
            "caption": str(parsed.get("caption", "")).strip(),
            "vlm_sign": vlm_sign,
            "agrees_with_classifier": (vlm_sign == top) if top else False,
            "reasoning": str(parsed.get("reasoning", "")).strip(),
            "candidates": candidates,
            "provider": f"Gemini VLM ({self.llm.model_name})",
        }

    def _build_prompt(self, candidates: List[Dict[str, Any]]) -> str:
        reference = "\n".join(
            f"- {sign}: {desc}"
            for sign, desc in HANDSHAPE_DESCRIPTIONS.items()
            if sign in ALL_CLASSES and len(sign) == 1
        )
        if candidates:
            cand_text = ", ".join(f"{c['sign']} ({c['confidence'] * 100:.0f}%)" for c in candidates)
        else:
            cand_text = "none provided"

        return (
            "You are the vision-language component of a sign language assistant. "
            "The image shows a person making an American Sign Language (ASL) fingerspelling "
            "handshape for one letter.\n\n"
            f"A fast landmark classifier proposed these candidate letters: {cand_text}.\n\n"
            "Reference handshapes:\n"
            f"{reference}\n\n"
            "The candidates are only hints from a simple model and may be wrong or even "
            "irrelevant. Base your answer on what is actually visible in the image.\n\n"
            "Tasks:\n"
            "1. First decide whether a human hand is clearly visible. If it is not (blank "
            "image, no hand, hand too blurry or cut off), set sign to NONE and caption it as "
            '"No hand is visible" plus what you do see. Never invent a hand.\n'
            "2. Otherwise write a one-sentence caption describing the hand: which fingers are "
            "extended or curled, the thumb position, and the hand orientation.\n"
            "3. Decide which single letter the handshape shows. Choose a candidate only if it "
            "genuinely matches the image; choose another letter if the image shows it instead.\n"
            "4. Give a short reason for the choice.\n\n"
            'Reply with JSON only: {"caption": "...", "sign": "A", "reasoning": "..."}.'
        )

    # ---------------------------------------------------------------- fallback
    def _offline_caption(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            return {
                "caption": "No classifier candidates available, so no caption could be generated offline.",
                "vlm_sign": "NONE",
                "agrees_with_classifier": False,
                "reasoning": "Add a Gemini API key to enable the vision-language model.",
                "candidates": [],
                "provider": "SignBridge offline captioner",
            }
        top = candidates[0]
        desc = HANDSHAPE_DESCRIPTIONS.get(top["sign"], "a hand gesture")
        return {
            "caption": f"A hand showing {desc} (letter {top['sign']}).",
            "vlm_sign": top["sign"],
            "agrees_with_classifier": True,
            "reasoning": (
                f"Offline mode: using the classifier's top guess ({top['confidence'] * 100:.0f}% "
                "confidence). Add a Gemini API key to let the vision-language model verify it."
            ),
            "candidates": candidates,
            "provider": "SignBridge offline captioner",
        }

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _clean_candidates(candidates: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        cleaned = []
        for c in (candidates or [])[:3]:
            try:
                sign = str(c.get("sign", "")).upper()
                conf = float(c.get("confidence", 0.0))
            except Exception:
                continue
            if sign in ALL_CLASSES:
                cleaned.append({"sign": sign, "confidence": conf})
        return cleaned

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    return None
        return None
