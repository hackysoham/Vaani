"""
Sarvam AI API Client Wrapper
Provides robust bindings for:
1. Speech-to-Text-Translate (Saaras)
2. Chat Completions (sarvam-m / sarvam-105b)
3. Translation (Mayura v1)
4. Text-to-Speech (Bulbul v3 / v2)
Includes automatic retries, fallback strategies, and offline simulation for hackathon judging.
"""

import os
import json
import base64
import re
import io
import wave
import struct
import math
import requests
from typing import Dict, Any, Optional, Tuple


def generate_fallback_chime(duration_sec: float = 1.8) -> bytes:
    """
    Generates a melodic 2-tone notification chime in pure Python standard library.
    Ensures judges can test the audio player even before configuring their API key.
    """
    sample_rate = 22050
    num_samples = int(sample_rate * duration_sec)
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Generate a pleasant 3-chord major arpeggio (C5 -> E5 -> G5)
        frequencies = [523.25, 659.25, 783.99]
        samples_per_note = num_samples // len(frequencies)

        frames = bytearray()
        for i, freq in enumerate(frequencies):
            for t in range(samples_per_note):
                decay = math.exp(-3.5 * (t / samples_per_note))
                val = int(24000 * decay * math.sin(2.0 * math.pi * freq * (t / sample_rate)))
                frames.extend(struct.pack("<h", max(-32767, min(32767, val))))

        wav_file.writeframes(frames)

    return buffer.getvalue()


class SarvamClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "").strip()

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 8)

    def _get_headers(self) -> Dict[str, str]:
        return {
            "api-subscription-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # =========================================================================
    # 1. SPEECH-TO-TEXT TRANSLATE (Saaras)
    # =========================================================================
    def speech_to_text_translate(self, audio_bytes: bytes, filename: str = "input_audio.wav") -> Dict[str, Any]:
        """
        Translates vernacular spoken audio directly into English text.
        Endpoints tested:
        1. https://api.sarvam.ai/speech-to-text-translate
        2. https://api.sarvam.ai/speech-to-text (mode='translate')
        """
        if not self.is_configured():
            return {
                "success": False,
                "transcript": "",
                "language_code": "unknown",
                "error": "Sarvam API Key is not configured."
            }

        headers = {
            "api-subscription-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}"
        }

        # Attempt 1: Direct speech-to-text-translate endpoint
        try:
            files = {"file": (filename, audio_bytes, "audio/wav")}
            data = {"model": "saaras:v2"}
            resp = requests.post(
                "https://api.sarvam.ai/speech-to-text-translate",
                headers=headers,
                files=files,
                data=data,
                timeout=30
            )

            if resp.status_code == 200:
                res_json = resp.json()
                transcript = res_json.get("transcript") or res_json.get("translated_text") or res_json.get("text", "")
                lang = res_json.get("language_code", "detected")
                return {"success": True, "transcript": transcript.strip(), "language_code": lang, "error": None}
        except Exception:
            pass

        # Attempt 2: speech-to-text with mode='translate'
        try:
            files = {"file": (filename, audio_bytes, "audio/wav")}
            data = {"model": "saaras:v2", "mode": "translate"}
            resp = requests.post(
                "https://api.sarvam.ai/speech-to-text",
                headers=headers,
                files=files,
                data=data,
                timeout=30
            )

            if resp.status_code == 200:
                res_json = resp.json()
                transcript = res_json.get("transcript") or res_json.get("translated_text") or res_json.get("text", "")
                lang = res_json.get("language_code", "detected")
                return {"success": True, "transcript": transcript.strip(), "language_code": lang, "error": None}
            else:
                return {
                    "success": False,
                    "transcript": "",
                    "language_code": "unknown",
                    "error": f"Saaras STT HTTP {resp.status_code}: {resp.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "transcript": "",
                "language_code": "unknown",
                "error": f"STT connection error: {str(e)}"
            }

    # =========================================================================
    # 2. CHAT COMPLETIONS (sarvam-m / sarvam-105b)
    # =========================================================================
    def generate_pandas_query(
        self,
        schema_markdown: str,
        dtypes_info: str,
        sample_rows_markdown: str,
        user_query: str
    ) -> Dict[str, Any]:
        """
        Sends ONLY the schema + sample rows + query to Sarvam Chat Completions.
        Returns strict JSON with:
        pandas_code, chart_type, x_col, y_col, english_insight
        """
        system_prompt = (
            "You are an expert Chief Quantitative Data Analyst and Pandas specialist.\n"
            "You are given a pandas DataFrame `df` containing structured data.\n"
            "Your task is to translate the user's analytical question into a precise, single-line Python expression using `df`.\n"
            "\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY a valid JSON object. Do NOT add any conversational preamble or outro.\n"
            "2. `pandas_code` MUST be a single-line valid python expression using `df` that evaluates to a DataFrame, Series, or scalar.\n"
            "   Examples of valid `pandas_code`:\n"
            "   - `df.groupby('city')['revenue'].sum().nlargest(5).reset_index()`\n"
            "   - `df.groupby('category')['profit'].mean().reset_index()`\n"
            "   - `df.sort_values(by='revenue', ascending=False).head(10)[['order_id', 'city', 'revenue']]`\n"
            "   - `df['revenue'].sum()`\n"
            "   Always use `.reset_index()` after groupby so the grouped column becomes a standard column for plotting.\n"
            "3. `chart_type` MUST be strictly one of: 'bar', 'line', 'area', or 'none'.\n"
            "4. `x_col` is the string name of the category/dimension column (or null if none).\n"
            "5. `y_col` is the string name of the numeric metric column (or null if none).\n"
            "6. `english_insight` MUST be a crisp 2-sentence executive summary explaining what this analytical query calculates and how a business executive should interpret the numbers.\n"
            "\n"
            "JSON Format Structure:\n"
            "{\n"
            '  "pandas_code": "df.groupby(\'city\')[\'revenue\'].sum().nlargest(5).reset_index()",\n'
            '  "chart_type": "bar",\n'
            '  "x_col": "city",\n'
            '  "y_col": "revenue",\n'
            '  "english_insight": "This analysis reveals the top 5 revenue-generating cities across all orders. Monitoring these high-velocity hubs is essential for targeted supply chain and inventory allocation."\n'
            "}"
        )

        user_content = (
            f"### DataFrame Schema:\n{schema_markdown}\n\n"
            f"### Column Data Types:\n{dtypes_info}\n\n"
            f"### Sample 3 Rows:\n{sample_rows_markdown}\n\n"
            f"### User Analytical Question:\n\"{user_query}\"\n\n"
            "Generate the JSON response now."
        )

        # Fallback simulation if no key is configured
        if not self.is_configured():
            return self._heuristic_fallback(user_query, schema_markdown)

        # Try models in order: sarvam-m, then sarvam-105b
        models_to_try = ["sarvam-m", "sarvam-105b"]

        for model_name in models_to_try:
            try:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }

                resp = requests.post(
                    "https://api.sarvam.ai/v1/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=45
                )

                if resp.status_code == 200:
                    resp_data = resp.json()
                    content = resp_data["choices"][0]["message"]["content"]
                    parsed = self._extract_json(content)
                    if parsed and "pandas_code" in parsed:
                        return {
                            "success": True,
                            "data": parsed,
                            "raw_content": content,
                            "model_used": model_name,
                            "error": None
                        }
            except Exception:
                continue

        # If API calls failed or rate limited, return fallback
        fallback = self._heuristic_fallback(user_query, schema_markdown)
        fallback["error"] = "Sarvam Chat Completions API unavailable or quota reached. Generated query using analytical heuristic engine."
        return fallback

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON even if surrounded by markdown code blocks."""
        try:
            text = text.strip()
            # If wrapped in markdown block
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # Try raw json search
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            return json.loads(text)
        except Exception:
            return None

    def _heuristic_fallback(self, query: str, schema_markdown: str) -> Dict[str, Any]:
        """
        Intelligent offline heuristic generator for standard retail queries.
        Ensures judges can test even if API keys have temporary network/quota issues.
        """
        q = query.lower()

        if any(w in q for w in ["city", "cities", "shahar", "region"]):
            metric = "revenue" if "revenue" in q or "sales" in q or "kamai" in q else "profit"
            return {
                "success": True,
                "data": {
                    "pandas_code": f"df.groupby('city')['{metric}'].sum().nlargest(5).reset_index()",
                    "chart_type": "bar",
                    "x_col": "city",
                    "y_col": metric,
                    "english_insight": f"This analysis aggregates the top 5 cities contributing the largest volume of {metric}. Identifying geographic hubs helps leadership double down on high-performing regional markets."
                },
                "raw_content": "Offline analytical fallback",
                "model_used": "heuristic-engine",
                "error": None
            }
        elif any(w in q for w in ["category", "categories", "segment", "vibhag"]):
            return {
                "success": True,
                "data": {
                    "pandas_code": "df.groupby('category')['profit'].sum().sort_values(ascending=False).reset_index()",
                    "chart_type": "bar",
                    "x_col": "category",
                    "y_col": "profit",
                    "english_insight": "This breakdown tracks the total net profit across distinct product categories. Higher-margin verticals should receive prioritized advertising budget and product line expansions."
                },
                "raw_content": "Offline analytical fallback",
                "model_used": "heuristic-engine",
                "error": None
            }
        elif any(w in q for w in ["month", "trend", "date", "samay", "mahina"]):
            return {
                "success": True,
                "data": {
                    "pandas_code": "df.assign(month=pd.to_datetime(df['date']).dt.to_period('M').astype(str)).groupby('month')['revenue'].sum().reset_index()",
                    "chart_type": "line",
                    "x_col": "month",
                    "y_col": "revenue",
                    "english_insight": "This time-series tracks monthly aggregated revenue performance throughout the fiscal cycle. Seasonal peaks reflect promotional surges and quarterly retail demand cycles."
                },
                "raw_content": "Offline analytical fallback",
                "model_used": "heuristic-engine",
                "error": None
            }
        elif any(w in q for w in ["payment", "upi", "credit card", "emi", "cod"]):
            return {
                "success": True,
                "data": {
                    "pandas_code": "df.groupby('payment_method')['revenue'].agg(['count', 'mean']).reset_index().rename(columns={'count': 'transactions', 'mean': 'avg_order_value'}).sort_values(by='transactions', ascending=False)",
                    "chart_type": "bar",
                    "x_col": "payment_method",
                    "y_col": "avg_order_value",
                    "english_insight": "This metric highlights transaction volume and average order values across different payment rails. Digital UPI and No-Cost EMI typically unlock higher basket sizes."
                },
                "raw_content": "Offline analytical fallback",
                "model_used": "heuristic-engine",
                "error": None
            }
        else:
            return {
                "success": True,
                "data": {
                    "pandas_code": "df.groupby('sub_category')['revenue'].sum().nlargest(8).reset_index()",
                    "chart_type": "bar",
                    "x_col": "sub_category",
                    "y_col": "revenue",
                    "english_insight": "This analysis surfaces the leading sub-categories by gross sales across the platform. Concentrating stock on top-performing items minimizes inventory holding overhead."
                },
                "raw_content": "Offline analytical fallback",
                "model_used": "heuristic-engine",
                "error": None
            }

    # =========================================================================
    # 3. TRANSLATION (Mayura v1)
    # =========================================================================
    def translate_text(
        self,
        text: str,
        target_language_code: str = "hi-IN",
        mode: str = "modern-colloquial"
    ) -> Dict[str, Any]:
        """
        Translates text from en-IN into target Indian language using Sarvam Mayura.
        Modes: 'modern-colloquial', 'formal', 'code-mixed'
        """
        clean_text = text.strip()[:950]  # Respect 1,000 char limit

        if not self.is_configured():
            indic_samples = {
                "hi-IN": f"यह विश्लेषण डेटा का कार्यकारी सारांश प्रस्तुत करता है। {clean_text}",
                "mr-IN": f"हे विश्लेषण डेटाचा मुख्य सारांश दर्शवते. {clean_text}",
                "ta-IN": f"இந்த பகுப்பாய்வு தரவின் முக்கிய சுருக்கத்தை அளிக்கிறது. {clean_text}",
                "te-IN": f"ఈ విశ్లేషణ డేటా యొక్క ముఖ్యమైన సారాంశాన్ని అందిస్తుంది. {clean_text}",
                "bn-IN": f"এই বিশ্লেষণটি তথ্যের মূল সারসংক্ষেপ উপস্থাপন করে। {clean_text}",
                "gu-IN": f"આ વિશ્લેષણ ડેટાનો મુખ્ય સારાંશ રજૂ કરે છે. {clean_text}",
                "kn-IN": f"ಈ ವಿಶ್ಲೇಷಣೆಯು ಡೇಟಾದ ಕಾರ್ಯನಿರ್ವಾಹಕ ಸಾರಾಂಶವನ್ನು ನೀಡುತ್ತದೆ. {clean_text}",
                "ml-IN": f"ഈ വിശകലനം ഡാറ്റയുടെ പ്രധാന സംഗ്രഹം അവതരിപ്പിക്കുന്നു. {clean_text}",
                "pa-IN": f"ਇਹ ਵਿਸ਼ਲੇਸ਼ਣ ਡੇਟਾ ਦਾ ਮੁੱਖ ਸਾਰਾਂਸ਼ ਪੇਸ਼ ਕਰਦਾ ਹੈ। {clean_text}",
                "od-IN": f"ଏହି ବିଶ୍ଳେଷଣ ତଥ୍ୟର ମୁଖ୍ୟ ସାରାଂଶ ଉପସ୍ଥାପନ କରେ। {clean_text}"
            }
            sample_trans = indic_samples.get(target_language_code, f"यह विश्लेषण डेटा का सारांश प्रस्तुत करता है: {clean_text}")
            return {
                "success": False,
                "translated_text": sample_trans,
                "error": "Sarvam API Key not configured. Displaying preview translation."
            }

        try:
            payload = {
                "input": clean_text,
                "source_language_code": "en-IN",
                "target_language_code": target_language_code,
                "model": "mayura:v1",
                "mode": mode
            }

            resp = requests.post(
                "https://api.sarvam.ai/translate",
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )

            if resp.status_code == 200:
                res_data = resp.json()
                translated = res_data.get("translated_text") or res_data.get("translation") or clean_text
                return {"success": True, "translated_text": translated, "error": None}
            else:
                return {
                    "success": False,
                    "translated_text": clean_text,
                    "error": f"Mayura Translation HTTP {resp.status_code}: {resp.text}"
                }
        except Exception as e:
            return {
                "success": False,
                "translated_text": clean_text,
                "error": f"Mayura connection error: {str(e)}"
            }

    # =========================================================================
    # 4. TEXT-TO-SPEECH (Bulbul v3 / v2)
    # =========================================================================
    def text_to_speech(
        self,
        text: str,
        target_language_code: str = "hi-IN",
        speaker: str = "meera",
        pace: float = 1.0,
        model: str = "bulbul:v3"
    ) -> Dict[str, Any]:
        """
        Synthesizes spoken audio from text using Sarvam Bulbul.
        Returns:
        {
            "success": bool,
            "audio_bytes": bytes or None,
            "format": "wav" or "mp3",
            "error": str or None
        }
        """
        clean_text = text.strip()[:2000]

        if not self.is_configured():
            return {
                "success": False,
                "audio_bytes": generate_fallback_chime(),
                "format": "wav",
                "is_fallback": True,
                "error": "Sarvam API Key not configured. Playing preview chime."
            }

        headers = self._get_headers()

        # Try schema 1: inputs array (standard Bulbul v3 / v2)
        models_to_try = [model, "bulbul:v3", "bulbul:v2", "bulbul:v4-flash"]
        seen_models = set()

        for m in models_to_try:
            if m in seen_models:
                continue
            seen_models.add(m)

            # Variant A: inputs array
            payload_a = {
                "inputs": [clean_text],
                "target_language_code": target_language_code,
                "speaker": speaker,
                "pitch": 0,
                "pace": float(pace),
                "loudness": 1.5,
                "speech_sample_rate": 22050,
                "enable_preprocessing": True,
                "model": m
            }

            try:
                resp = requests.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers=headers,
                    json=payload_a,
                    timeout=35
                )

                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    if "audio" in content_type:
                        fmt = "mp3" if "mpeg" in content_type else "wav"
                        return {"success": True, "audio_bytes": resp.content, "format": fmt, "error": None}

                    res_json = resp.json()
                    # Check audios array
                    if "audios" in res_json and len(res_json["audios"]) > 0:
                        b64_audio = res_json["audios"][0]
                        audio_data = base64.b64decode(b64_audio)
                        return {"success": True, "audio_bytes": audio_data, "format": "wav", "error": None}
                    elif "audio" in res_json and res_json["audio"]:
                        b64_audio = res_json["audio"]
                        audio_data = base64.b64decode(b64_audio)
                        return {"success": True, "audio_bytes": audio_data, "format": "wav", "error": None}
            except Exception:
                pass

            # Variant B: single text schema
            payload_b = {
                "text": clean_text,
                "language_code": target_language_code,
                "model": m,
                "speaker": speaker,
                "pace": float(pace)
            }

            try:
                resp = requests.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers=headers,
                    json=payload_b,
                    timeout=35
                )

                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    if "audio" in content_type:
                        fmt = "mp3" if "mpeg" in content_type else "wav"
                        return {"success": True, "audio_bytes": resp.content, "format": fmt, "error": None}

                    res_json = resp.json()
                    if "audio" in res_json and res_json["audio"]:
                        b64_audio = res_json["audio"]
                        audio_data = base64.b64decode(b64_audio)
                        return {"success": True, "audio_bytes": audio_data, "format": "wav", "error": None}
            except Exception:
                pass

        return {
            "success": False,
            "audio_bytes": generate_fallback_chime(),
            "format": "wav",
            "is_fallback": True,
            "error": "Bulbul TTS failed across all model versions or quota reached. Playing preview chime."
        }
