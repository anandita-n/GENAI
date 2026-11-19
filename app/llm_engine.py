"""
SignBridge AI - Generative AI & LLM Prompt Engineering Engine
Transforms raw recognized sign language tokens and fingerspelling fragments
into grammatically fluent, context-aware English sentences across multiple styles,
and generates context-aware conversational replies.
Supports Google Gemini API with seamless offline heuristic fallbacks.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from app.config import GEMINI_API_KEY, GEMINI_MODEL


# Style specifications and prompt system directives
STYLE_PROMPTS = {
    "casual": (
        "Convert the raw sign language tokens into a friendly, natural, and conversational everyday English sentence. "
        "Keep the tone warm, approachable, and colloquial."
    ),
    "formal": (
        "Convert the raw sign language tokens into a polite, professional, and grammatically polished English sentence. "
        "Suitable for workplace, healthcare, legal, or educational environments."
    ),
    "concise": (
        "Convert the raw sign language tokens into a brief, clear, and direct sentence. "
        "Omit unnecessary words while preserving complete clarity and intent."
    ),
    "empathetic": (
        "Convert the raw sign language tokens into a warm, polite, and considerate English sentence. "
        "Highlight appreciation, courtesy, and respect."
    )
}

FEW_SHOT_EXAMPLES = [
    {
        "raw": "N E D D O C T E R W H E R E H O S P I T A L",
        "casual": "I need a doctor, could you tell me where the nearest hospital is?",
        "formal": "I require medical attention. Could you please direct me to the nearest hospital?",
        "concise": "Need a doctor. Where is the hospital?",
        "empathetic": "I'm not feeling well and need a doctor. Could you please help guide me to the hospital?"
    },
    {
        "raw": "H E L O M Y N A M E A L E X N I C E M E T Y O U",
        "casual": "Hello! My name is Alex, really nice to meet you!",
        "formal": "Hello, my name is Alex. It is a pleasure to make your acquaintance.",
        "concise": "Hello, I'm Alex. Nice to meet you.",
        "empathetic": "Hi there! My name is Alex. I'm truly delighted to meet you!"
    },
    {
        "raw": "W H A T T I M E M E T I N G S T A R T",
        "casual": "Hey, what time does the meeting start?",
        "formal": "Could you please inform me at what time the meeting is scheduled to begin?",
        "concise": "What time does the meeting begin?",
        "empathetic": "Excuse me, could you kindly let me know when our meeting starts?"
    },
    {
        "raw": "T H A N K Y O U F O R H E L P",
        "casual": "Thanks a lot for your help!",
        "formal": "Thank you very much for your kind assistance.",
        "concise": "Thank you for the help.",
        "empathetic": "Thank you so much for your wonderful help and support, I truly appreciate it!"
    }
]


class LLMEngine:
    """Generative AI assistant engine powered by Gemini or built-in grammar rules."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_MODEL):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self._init_gemini()

    def _init_gemini(self):
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                print(f"[LLMEngine] Initialized Google Gemini client with model: {self.model_name}")
            except Exception as e:
                print(f"[LLMEngine] Notice: Gemini client init with google-genai failed ({e}). "
                      f"Will use REST / offline fallback mode.")
                self.client = None

    def update_api_key(self, api_key: str):
        """Updates the API key dynamically at runtime from UI settings."""
        self.api_key = api_key.strip()
        self._init_gemini()

    def refine_sentence(
        self,
        raw_text: str,
        style: str = "casual",
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Refines raw sign language tokens into grammatical English according to the requested style.
        Returns:
            {
                "raw_text": ...,
                "refined_text": ...,
                "style": ...,
                "provider": "gemini" | "offline_rules",
                "suggestions": [str, str, str],
                "prompt_used": str
            }
        """
        clean_raw = raw_text.strip()
        if not clean_raw:
            return {
                "raw_text": "",
                "refined_text": "",
                "style": style,
                "provider": "none",
                "suggestions": [],
                "prompt_used": ""
            }

        style_instruction = STYLE_PROMPTS.get(style.lower(), STYLE_PROMPTS["casual"])
        prompt = self._build_refinement_prompt(clean_raw, style, style_instruction, conversation_context)

        # 1. Try Gemini if configured
        if self.client and self.api_key:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                response_text = response.text.strip()
                parsed = self._parse_json_or_text(response_text)
                if parsed and "refined_sentence" in parsed:
                    return {
                        "raw_text": clean_raw,
                        "refined_text": parsed.get("refined_sentence", clean_raw),
                        "style": style,
                        "provider": f"Gemini ({self.model_name})",
                        "suggestions": parsed.get("reply_suggestions", []),
                        "prompt_used": prompt
                    }
            except Exception as e:
                print(f"[LLMEngine] Gemini API call error: {e}. Falling back to linguistic rule engine.")

        # 2. Offline Linguistic & Heuristic Grammar Fallback
        fallback_result = self._offline_refine(clean_raw, style)
        return {
            "raw_text": clean_raw,
            "refined_text": fallback_result["refined_text"],
            "style": style,
            "provider": "SignBridge Built-in Grammar Engine",
            "suggestions": fallback_result["suggestions"],
            "prompt_used": prompt
        }

    def generate_replies(
        self,
        last_message: str,
        role: str = "hearing_partner"
    ) -> List[str]:
        """
        Generates 3 contextual response options for two-way communication.
        """
        if not last_message.strip():
            return ["Hello! How can I help you?", "Nice to meet you.", "Could you please repeat that?"]

        if self.client and self.api_key:
            try:
                prompt = (
                    f"You are SignBridge AI, a communication assistant bridging Deaf and hearing people.\n"
                    f"The other person just communicated: '{last_message}'\n"
                    f"Generate 3 natural, concise, contextually appropriate conversational replies for the {role}.\n"
                    f"Return ONLY a JSON array of 3 strings: [\"reply1\", \"reply2\", \"reply3\"]."
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                parsed = json.loads(self._clean_json_str(response.text))
                if isinstance(parsed, list) and len(parsed) >= 1:
                    return parsed[:3]
            except Exception as e:
                print(f"[LLMEngine] Reply generation fallback: {e}")

        # Heuristic conversational replies
        msg_lower = last_message.lower()
        if any(w in msg_lower for w in ["hospital", "doctor", "emergency", "help", "pain"]):
            return [
                "The hospital is 2 blocks away. I can accompany you right now.",
                "Would you like me to call emergency services or a doctor?",
                "Please sit down and rest while I get medical assistance."
            ]
        elif any(w in msg_lower for w in ["hello", "hi", "hey", "morning", "meet"]):
            return [
                "Hello! Nice to meet you too. How can I help you today?",
                "Good day! Glad to communicate with you.",
                "Hi there! Welcome, let me know what you need."
            ]
        elif any(w in msg_lower for w in ["where", "how", "what time", "restroom", "station"]):
            return [
                "It is located down the corridor to your right.",
                "Let me guide you there directly.",
                "It starts in about 15 minutes."
            ]
        elif any(w in msg_lower for w in ["thank", "appreciate", "grateful"]):
            return [
                "You are very welcome! Always happy to assist.",
                "Glad I could help. Have a wonderful day!",
                "Anytime! Please reach out if you need anything else."
            ]
        else:
            return [
                "Understood. Let me know if there's anything else you'd like to discuss.",
                "Thank you for sharing. How would you like to proceed?",
                "I am here to help, please take your time."
            ]

    def _build_refinement_prompt(
        self,
        raw_text: str,
        style: str,
        style_instruction: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        few_shots_str = ""
        for ex in FEW_SHOT_EXAMPLES[:3]:
            few_shots_str += (
                f"- Input Sign Tokens: \"{ex['raw']}\"\n"
                f"  Casual: \"{ex['casual']}\"\n"
                f"  Formal: \"{ex['formal']}\"\n"
                f"  Concise: \"{ex['concise']}\"\n"
                f"  Empathetic: \"{ex['empathetic']}\"\n\n"
            )

        context_str = ""
        if conversation_context and len(conversation_context) > 0:
            context_str = "Prior Conversation History:\n"
            for msg in conversation_context[-3:]:
                context_str += f"- {msg.get('sender', 'Partner')}: {msg.get('text', '')}\n"
            context_str += "\n"

        prompt = (
            f"You are the language processing engine of SignBridge AI, an assistive communication system.\n"
            f"The user is fingerspelling or signing in American Sign Language (ASL). The input may contain:\n"
            f"- Space-separated letters representing fingerspelled words (e.g. 'H E L O' -> 'Hello')\n"
            f"- Phonetic or visual substitution typos (e.g. 'NED' for 'need', 'DOCTER' for 'doctor')\n"
            f"- Telegram-style gloss or missing function words (e.g. 'WHERE HOSPITAL')\n\n"
            f"Target Communication Style: {style.upper()}\n"
            f"Style Directive: {style_instruction}\n\n"
            f"{context_str}"
            f"Few-Shot Examples:\n{few_shots_str}"
            f"Current Raw Input:\n\"{raw_text}\"\n\n"
            f"Output Requirement:\n"
            f"Respond ONLY with a valid JSON object in this exact schema, without markdown code fences:\n"
            f'{{"refined_sentence": "<complete grammatically correct sentence>", '
            f'"reply_suggestions": ["<suggestion 1>", "<suggestion 2>", "<suggestion 3>"]}}'
        )
        return prompt

    def _clean_json_str(self, text: str) -> str:
        s = text.strip()
        if s.startswith("```json"):
            s = s[7:]
        if s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    def _parse_json_or_text(self, text: str) -> Optional[Dict[str, Any]]:
        cleaned = self._clean_json_str(text)
        try:
            return json.loads(cleaned)
        except Exception:
            # Try regex extraction
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return {"refined_sentence": cleaned, "reply_suggestions": []}

    def _offline_refine(self, raw_text: str, style: str) -> Dict[str, Any]:
        """
        Intelligent offline heuristic correction engine.
        Converts spaced characters (e.g. "H E L O   M Y   N A M E") into words,
        applies common spelling and phonetic corrections, and structures grammatical sentences.
        """
        # Split by multiple spaces (double space or more represents word separator in fingerspelling)
        raw_words = re.split(r'\s{2,}', raw_text.strip())
        assembled_words = []

        for rw in raw_words:
            tokens = rw.split()
            if all(len(t) == 1 for t in tokens):
                # e.g. "H E L O" -> "HELO"
                assembled_words.append("".join(tokens))
            else:
                assembled_words.append(rw.replace(" ", ""))

        assembled_str = " ".join(assembled_words) if assembled_words else raw_text.strip()

        # Dictionary of ASL fingerspelling / phonetic spelling corrections
        CORRECTIONS = {
            "HELO": "hello",
            "HELOU": "hello",
            "NED": "need",
            "DOCTER": "doctor",
            "HOSPITAL": "hospital",
            "THX": "thanks",
            "THNK": "thank",
            "PLZ": "please",
            "PLS": "please",
            "WHR": "where",
            "WHER": "where",
            "METING": "meeting",
            "TIM": "time",
            "RESTROM": "restroom",
            "BATHROM": "bathroom",
            "WATR": "water",
            "HLP": "help",
            "NME": "name",
            "APPT": "appointment",
            "EMRGN": "emergency"
        }

        fixed_tokens = []
        for w in assembled_str.split():
            clean_w = w.upper()
            if clean_w in CORRECTIONS:
                fixed_tokens.append(CORRECTIONS[clean_w])
            else:
                fixed_tokens.append(w.lower())

        base_sentence = " ".join(fixed_tokens)

        # Contextual Sentence Construction based on keyword intent
        lower_s = base_sentence.lower()

        if "doctor" in lower_s or "hospital" in lower_s or "medical" in lower_s:
            if style == "formal":
                refined = "I require medical attention. Could you please direct me to the nearest doctor or hospital?"
            elif style == "concise":
                refined = "I need a doctor or hospital immediately."
            elif style == "empathetic":
                refined = "I am feeling unwell and need medical help. Could you kindly guide me to a hospital?"
            else:
                refined = "I need to see a doctor. Where is the nearest hospital?"
            suggestions = [
                "The clinic is on the first floor.",
                "Let me call an ambulance for you immediately.",
                "I will walk with you to the emergency department."
            ]

        elif "hello" in lower_s or "hi" in lower_s or "name" in lower_s:
            name_part = ""
            for i, tok in enumerate(fixed_tokens):
                if tok == "name" and i + 1 < len(fixed_tokens):
                    name_part = fixed_tokens[i+1].capitalize()

            if name_part:
                if style == "formal":
                    refined = f"Good day. My name is {name_part}. It is an honor to meet you."
                elif style == "concise":
                    refined = f"Hello, I am {name_part}."
                elif style == "empathetic":
                    refined = f"Hello! My name is {name_part}. I am truly delighted to connect with you."
                else:
                    refined = f"Hello! My name is {name_part}, nice to meet you."
            else:
                if style == "formal":
                    refined = "Good day. It is a pleasure to meet you."
                elif style == "concise":
                    refined = "Hello, nice to meet you."
                elif style == "empathetic":
                    refined = "Hi there! I am so glad to meet you today."
                else:
                    refined = "Hello, it is great to meet you!"
            suggestions = [
                "Hello! Very nice to meet you.",
                "Welcome! How can I assist you today?",
                "Hi there, pleased to meet you as well."
            ]

        elif "where" in lower_s and ("restroom" in lower_s or "bathroom" in lower_s):
            if style == "formal":
                refined = "Pardon me, could you please direct me to the nearest restroom?"
            elif style == "concise":
                refined = "Where is the restroom?"
            elif style == "empathetic":
                refined = "Excuse me, would you mind showing me where the restroom is?"
            else:
                refined = "Could you tell me where the restroom is?"
            suggestions = [
                "The restroom is down the corridor to your left.",
                "It is located right next to the elevators.",
                "Follow this hallway straight ahead."
            ]

        elif "water" in lower_s or "thirsty" in lower_s:
            if style == "formal":
                refined = "Could I please request a glass of water?"
            elif style == "concise":
                refined = "May I have some water?"
            elif style == "empathetic":
                refined = "Would you be kind enough to provide some water? Thank you."
            else:
                refined = "Could I please get some water?"
            suggestions = [
                "Here is some fresh water for you.",
                "The water dispenser is right around the corner.",
                "Let me get you a bottle right away."
            ]

        elif "thank" in lower_s:
            if style == "formal":
                refined = "Thank you very much for your time and assistance."
            elif style == "concise":
                refined = "Thank you for your help."
            elif style == "empathetic":
                refined = "I am deeply grateful for your kindness and help!"
            else:
                refined = "Thanks a lot for your help!"
            suggestions = [
                "You are very welcome!",
                "Happy to help anytime.",
                "Have a wonderful rest of your day!"
            ]

        else:
            # Capitalize and add proper punctuation
            cap = base_sentence.capitalize()
            if not cap.endswith((".", "?", "!")):
                cap += "."

            if style == "formal":
                refined = f"I would like to state: {cap}"
            elif style == "concise":
                refined = cap
            elif style == "empathetic":
                refined = f"Kindly note: {cap} Thank you for your understanding."
            else:
                refined = cap
            suggestions = [
                "Understood, thank you.",
                "Could you tell me more about that?",
                "I will take care of this right away."
            ]

        return {
            "refined_text": refined,
            "suggestions": suggestions
        }
