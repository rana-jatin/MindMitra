"""
MindMitra Psychology Workflow v2 — Modular Architecture
========================================================

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │                    UserContext (shared JSON)                  │
  │  Every module reads from and writes results back to this     │
  └────────┬──────────┬──────────┬──────────┬──────────┬─────────┘
           │          │          │          │          │
     ┌─────▼────┐ ┌──▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼──────────┐
     │  Memory   │ │ NLP  │ │Cultural│ │Psych   │ │  Technique   │
     │  System   │ │(Groq)│ │Context │ │Analysis│ │  Selector    │
     │ (kept as  │ │      │ │ Module │ │(GLM)   │ │  (GLM)       │
     │  is)      │ │      │ │        │ │        │ │              │
     └──────────┘ └──────┘ └────────┘ └────────┘ └──────────────┘
                                                        │
                                                  ┌─────▼──────┐
                                                  │  Response   │
                                                  │  Generator  │
                                                  │  (GLM)      │
                                                  └─────────────┘

External interface (process_user_chat / process_chat) is UNCHANGED.
"""

import openai
import os
import json
import time
import logging
import threading
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from copy import deepcopy
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from supabase import create_client, Client
from memory_architecture import UniversalMemorySystem

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


# ╔══════════════════════════════════════════════════════════════╗
# ║  1. SHARED USER-CONTEXT JSON SCHEMA                         ║
# ╚══════════════════════════════════════════════════════════════╝

def create_empty_user_context(
    user_id: str = "anonymous",
    session_id: str = None,
    user_message: str = "",
) -> Dict[str, Any]:
    """
    Canonical JSON envelope that every module reads from / writes to.
    Nothing leaves or enters the pipeline except through this structure.
    """
    return {
        # ── identity ──
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),

        # ── raw input ──
        "user_message": user_message,
        "voice_analysis": {},                       # optional voice data

        # ── session history (populated by caller / memory fetch) ──
        "session_context": {
            "recent_messages": [],
            "conversation_summary": {},
            "session_memories": {
                "procedural": [],
                "semantic": [],
                "episodic": [],
            },
            "user_activities": [],
            "user_patterns": {},
        },

        # ── NLP analysis  (written by Groq NLP module) ──
        "nlp_analysis": {
            "emotions": {},                         # {joy: 0.1, sadness: 0.7, …}
            "primary_emotion": "",
            "sentiment": {
                "score": 0.0,                       # -1 … +1
                "label": "neutral",                 # positive / negative / neutral / mixed
            },
            "intensity": 0.0,                       # 0 … 1
            "key_phrases": [],
            "language_detected": "en",
            "urgency_flag": False,
        },

        # ── cultural context (written by cultural module) ──
        "cultural_context": {
            "language_style": "casual",             # formal / casual / hindi-mixed
            "hindi_english_ratio": 0.0,             # 0 = pure English, 1 = pure Hindi
            "code_switching_detected": False,
            "cultural_sensitivity_flags": [],        # e.g. "parental_pressure", "exam_stress"
            "communication_pattern": "",
            "regional_context": "",
            "formality_level": "medium",            # low / medium / high
        },

        # ── psychological analysis  (written by GLM Agent 1) ──
        "psychological_analysis": {
            "emotional_state": "",
            "stress_categories": [],
            "risk_assessment": "low",
            "coping_assessment": "",
            "intervention_priority": "supportive",
            "psychological_insights": [],
            "cultural_pressures": "",
        },

        # ── technique selection  (written by GLM Agent 2) ──
        "technique_selection": {
            "primary_technique": "",                # CBT / ACT / MBCT / …
            "therapeutic_approach": "",
            "activity_recommendations": [],
            "rationale": "",
        },

        # ── final output ──
        "ai_response": "",
        "response_generated": False,
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. GROQ NLP — Emotion & Sentiment Analysis                 ║
# ╚══════════════════════════════════════════════════════════════╝

class GroqNLPModule:
    """
    Lightweight emotion / sentiment analysis via Groq (llama/mixtral).
    Handles token-limit errors with automatic truncation & retry.
    """

    # Groq free-tier context sizes by model
    _MODEL_TOKEN_LIMITS = {
        "qwen/qwen3-32b": 4_096,
        "moonshotai/kimi-k2-instruct-0905": 4096,
        "meta-llama/llama-4-scout-17b-16e-instruct": 8_192,
       # "mixtral-8x7b-32768": 32_768,
    }

    def __init__(self, api_key: str = None, model: str = "qwen/qwen3-32b"):
        self.api_key = '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'#api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ [GROQ-NLP] GROQ_API_KEY not set — NLP module disabled")
            self.client = None
            return

        try:
            
            self.client = Groq(api_key=self.api_key)
            self.model = model
            self._max_input_chars = self._MODEL_TOKEN_LIMITS.get(model, 8_192) * 3  # ~3 chars/token rough est
            logger.info(f"✅ [GROQ-NLP] Initialised with model={model}")
        except ImportError:
            logger.warning("⚠️ [GROQ-NLP] `groq` package not installed — NLP module disabled")
            self.client = None
        except Exception as e:
            logger.error(f"❌ [GROQ-NLP] Init failed: {e}")
            self.client = None

    # ── public entry ──────────────────────────────────────────
    def analyse(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run emotion + sentiment analysis; write results into user_context['nlp_analysis']."""
        if not self.client:
            logger.info("[GROQ-NLP] Skipped (client not available)")
            return user_context

        text = user_context.get("user_message", "")
        # Include last 3 messages for conversational context
        recent = user_context["session_context"].get("recent_messages", [])[-3:]
        history_snippet = " | ".join(
            f"{m.get('role','?')}: {m.get('content','')[:120]}" for m in recent
        )

        prompt = self._build_prompt(text, history_snippet)
        raw = self._call_groq(prompt)
        parsed = self._parse_response(raw)
        user_context["nlp_analysis"] = parsed
        logger.info(f"✅ [GROQ-NLP] Emotion={parsed.get('primary_emotion')}, Sentiment={parsed['sentiment']['label']}")
        return user_context

    # ── internals ─────────────────────────────────────────────
    def _build_prompt(self, text: str, history: str) -> str:
        return f"""Analyse the following user message for a mental-health chatbot.  
Return ONLY valid JSON (no markdown fences) with exactly these keys:

{{
  "emotions": {{"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "trust": 0.0, "anticipation": 0.0}},
  "primary_emotion": "<strongest emotion name>",
  "sentiment": {{"score": <float -1 to 1>, "label": "<positive|negative|neutral|mixed>"}},
  "intensity": <float 0 to 1>,
  "key_phrases": ["<phrase1>", "<phrase2>"],
  "language_detected": "<en|hi|hinglish>",
  "urgency_flag": <true if crisis/self-harm indicators else false>
}}

Recent conversation context: {history[:600]}

User message: \"{text[:1500]}\"

JSON:"""

    def _call_groq(self, prompt: str, _retry: int = 0) -> str:
        """Call Groq with automatic truncation on token-limit errors."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            err_str = str(e).lower()
            # Handle token limit exceeded — truncate and retry once
            if ("token" in err_str or "context_length" in err_str or "rate_limit" in err_str) and _retry < 2:
                logger.warning(f"⚠️ [GROQ-NLP] Token/rate limit hit (attempt {_retry+1}), truncating...")
                truncated = prompt[: len(prompt) // 2]
                return self._call_groq(truncated, _retry + 1)
            logger.error(f"❌ [GROQ-NLP] API call failed: {e}")
            return "{}"

    def _parse_response(self, raw: str) -> Dict:
        """Robust JSON parse with fallback defaults."""
        defaults = {
            "emotions": {},
            "primary_emotion": "unknown",
            "sentiment": {"score": 0.0, "label": "neutral"},
            "intensity": 0.0,
            "key_phrases": [],
            "language_detected": "en",
            "urgency_flag": False,
        }
        if not raw:
            return defaults
        try:
            # Strip markdown fences if the model wraps them
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            parsed = json.loads(cleaned)
            # Merge with defaults so no key is ever missing
            for k, v in defaults.items():
                if k not in parsed:
                    parsed[k] = v
            return parsed
        except json.JSONDecodeError:
            logger.warning("[GROQ-NLP] Failed to parse JSON, using defaults")
            return defaults


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. CULTURAL CONTEXT & LANGUAGE STYLE MODULE                 ║
# ╚══════════════════════════════════════════════════════════════╝

class CulturalContextModule:
    """
    Rule-based + lightweight LLM analysis for:
      • Hindi / Hinglish detection & code-switching level
      • Formality level
      • Cultural sensitivity flags (exam stress, parental pressure, …)
      • Communication pattern classification
    Uses the NLP analysis already present in user_context and optionally
    calls Groq for deeper classification (kept cheap — single short call).
    """

    # Common Hindi / Hinglish markers
    _HINDI_MARKERS = {
        "yaar", "bhai", "didi", "maa", "papa", "ghar", "padhai", "exam",
        "nahi", "kya", "hai", "mein", "toh", "acha", "theek", "kuch",
        "kaise", "kyun", "bohot", "bahut", "zyada", "bilkul", "sach",
        "samajh", "dukh", "tension", "pareshan", "darr", "chinta",
        "mann", "dil", "sapna", "zindagi", "rishta", "shaadi",
        "arre", "haan", "naa", "abhi", "bas", "matlab", "lekin",
        "accha", "suno", "bata", "bol", "rona", "akela", "thak",
    }

    _CULTURAL_KEYWORDS = {
        "parental_pressure": ["parents", "papa", "maa", "mom", "dad", "family", "ghar", "expect", "disappoint", "proud"],
        "exam_stress": ["exam", "jee", "neet", "boards", "cgpa", "marks", "rank", "topper", "padhai", "result", "semester"],
        "career_anxiety": ["career", "job", "placement", "package", "future", "engineer", "doctor", "startup", "salary"],
        "social_pressure": ["friends", "relationship", "breakup", "lonely", "akela", "judge", "log kya kahenge", "society"],
        "identity_struggle": ["identity", "confused", "who am i", "purpose", "meaning", "self", "worth"],
        "marriage_pressure": ["shaadi", "marriage", "rishta", "arrange", "partner", "settle"],
        "mental_health_stigma": ["pagal", "crazy", "weak", "therapy", "stigma", "shame", "hide"],
    }

    def __init__(self, groq_nlp: Optional[GroqNLPModule] = None):
        self.groq_nlp = groq_nlp  # reuse same Groq client for optional deep analysis
        logger.info("✅ [CULTURAL] Cultural context module initialised")

    def analyse(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run cultural analysis; write results into user_context['cultural_context']."""
        text = user_context.get("user_message", "").lower()
        history = user_context["session_context"].get("recent_messages", [])

        result = {
            "language_style": self._detect_language_style(text),
            "hindi_english_ratio": self._compute_hindi_ratio(text),
            "code_switching_detected": False,
            "cultural_sensitivity_flags": self._detect_cultural_flags(text),
            "communication_pattern": self._detect_communication_pattern(text, history),
            "regional_context": self._infer_regional_context(text, history),
            "formality_level": self._detect_formality(text),
        }
        result["code_switching_detected"] = result["hindi_english_ratio"] > 0.1

        # If session history exists, enrich from patterns across messages
        if history:
            result = self._enrich_from_history(result, history)

        user_context["cultural_context"] = result
        logger.info(
            f"✅ [CULTURAL] Style={result['language_style']}, "
            f"Hindi%={result['hindi_english_ratio']:.0%}, "
            f"Flags={result['cultural_sensitivity_flags']}"
        )
        return user_context

    # ── detection helpers ─────────────────────────────────────
    def _detect_language_style(self, text: str) -> str:
        words = set(text.split())
        hindi_count = len(words & self._HINDI_MARKERS)
        total = max(len(words), 1)
        ratio = hindi_count / total
        if ratio > 0.25:
            return "hindi-mixed"
        elif ratio > 0.08:
            return "hinglish"
        return "english"

    def _compute_hindi_ratio(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        hindi_count = sum(1 for w in words if w.lower() in self._HINDI_MARKERS)
        return round(hindi_count / len(words), 3)

    def _detect_cultural_flags(self, text: str) -> List[str]:
        flags = []
        text_lower = text.lower()
        for flag, keywords in self._CULTURAL_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                flags.append(flag)
        return flags

    def _detect_communication_pattern(self, text: str, history: List) -> str:
        if len(text.split()) < 5:
            return "terse"
        elif len(text.split()) > 80:
            return "verbose"
        elif text.endswith("?"):
            return "questioning"
        elif any(w in text.lower() for w in ["feel", "feeling", "felt", "lagta", "mehsoos"]):
            return "emotionally_expressive"
        return "conversational"

    def _detect_formality(self, text: str) -> str:
        informal_markers = {"lol", "haha", "omg", "wtf", "bruh", "yaar", "arre", "bc", "mc"}
        formal_markers = {"sir", "ma'am", "respected", "kindly", "please", "would you"}
        words = set(text.lower().split())
        if words & informal_markers:
            return "low"
        if words & formal_markers:
            return "high"
        return "medium"

    def _infer_regional_context(self, text: str, history: List) -> str:
        # Simple keyword-based; can be enhanced
        all_text = text + " ".join(m.get("content", "") for m in history[-5:])
        all_lower = all_text.lower()
        if any(w in all_lower for w in ["kota", "jee", "iit", "coaching"]):
            return "competitive_exam_belt"
        if any(w in all_lower for w in ["bangalore", "bengaluru", "hyderabad", "pune", "it job", "startup"]):
            return "tech_hub"
        if any(w in all_lower for w in ["village", "gaon", "rural"]):
            return "rural"
        return "urban_metro"

    def _enrich_from_history(self, result: Dict, history: List) -> Dict:
        """Aggregate patterns across session history."""
        all_text = " ".join(m.get("content", "") for m in history if m.get("role") == "user")
        # Accumulate cultural flags from entire session
        session_flags = set(result["cultural_sensitivity_flags"])
        for flag, keywords in self._CULTURAL_KEYWORDS.items():
            if any(kw in all_text.lower() for kw in keywords):
                session_flags.add(flag)
        result["cultural_sensitivity_flags"] = list(session_flags)

        # Detect overall session language style (may differ from single message)
        session_hindi = self._compute_hindi_ratio(all_text)
        if session_hindi > result["hindi_english_ratio"]:
            result["hindi_english_ratio"] = round(
                (result["hindi_english_ratio"] + session_hindi) / 2, 3
            )
            if session_hindi > 0.2:
                result["language_style"] = "hindi-mixed"
        return result


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. GLM CONCURRENCY CONTROLLER                              ║
# ╚══════════════════════════════════════════════════════════════╝

import openai
import os
import threading
import time
import logging

logger = logging.getLogger(__name__)

class GLMController:
    def __init__(
        self,
        api_key: str = None,
        model: str = "glm-4.5-flash",  # Change model name here
        max_concurrent: int = 2,
        max_retries: int = 3,
        base_backoff: float = 2.0,
    ):
        self.api_key = '0b7ae4bd0c9b45878e633fd8be74bd4a.yuhRrTenuKNVWhD4'#api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for GLM controller")

        self.model_name = model
        self._semaphore = threading.Semaphore(max_concurrent)
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._lock = threading.Lock()

        # Set OpenAI API key
        openai.api_key = self.api_key
        logger.info(f"✅ [GLM] Controller ready — model={model}, max_concurrent={max_concurrent}")

    def invoke(self, messages: List, **kwargs) -> Any:
        """
        Thread-safe invoke with semaphore gating and retry on rate limits.
        """
        for attempt in range(self._max_retries):
            self._semaphore.acquire()
            try:
                # Construct the OpenAI API request payload
                openai_messages = [{"role": "system", "content": "You are a helpful assistant."}]
                openai_messages.extend([{"role": "user", "content": msg["content"]} for msg in messages])
                
                # OpenAI API call to generate a response
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=openai_messages,
                    max_tokens=500,  # Set your preferred max tokens
                    temperature=0.3,  # Adjust based on desired creativity
                    top_p=0.8,
                    **kwargs
                )
                
                return response['choices'][0]['message']['content']
            
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "quota" in err or "resource_exhausted" in err:
                    wait = self._base_backoff * (2 ** attempt)
                    logger.warning(f"⚠️ [GLM] Rate limited (attempt {attempt+1}), backing off {wait:.1f}s")
                    time.sleep(wait)
                else:
                    logger.error(f"❌ [GLM] Non-retryable error: {e}")
                    raise
            finally:
                self._semaphore.release()

        raise RuntimeError(f"[GLM] Exhausted {self._max_retries} retries due to rate limiting")



# ╔══════════════════════════════════════════════════════════════╗
# ║  5. GLM AGENT 1 — Psychologist Analysis                     ║
# ╚══════════════════════════════════════════════════════════════╝

class PsychologistAnalysisAgent:
    def __init__(self, glm: GLMController):
        self.glm = glm
        logger.info("✅ [AGENT-1] Psychologist analysis agent ready")

    def run(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🧠 [AGENT-1] Starting psychological analysis...")

        prompt = self._build_prompt(user_context)
        resp = self.glm.invoke([{"role": "user", "content": prompt}])

        if not resp or not resp.content:
            raise ValueError("[AGENT-1] GLM returned empty response")

        parsed = self._parse_analysis(resp.content)
        user_context["psychological_analysis"] = parsed
        logger.info(
            f"✅ [AGENT-1] Done — state={parsed.get('emotional_state','?')}, "
            f"priority={parsed.get('intervention_priority','?')}"
        )
        return user_context


    def _build_prompt(self, ctx: Dict) -> str:
        nlp = ctx.get("nlp_analysis", {})
        cultural = ctx.get("cultural_context", {})
        session = ctx.get("session_context", {})
        activities = session.get("user_activities", [])
        memories = session.get("session_memories", {})

        # Format memories compactly
        mem_lines = []
        for mtype in ("procedural", "semantic", "episodic"):
            for m in memories.get(mtype, [])[:4]:
                content = m.get("memory_content", m.get("content", ""))
                mem_lines.append(f"  [{mtype}] {content[:120]}")
        mem_block = "\n".join(mem_lines) if mem_lines else "No prior memories."

        # Format activities compactly
        act_lines = []
        for a in activities[:5]:
            atype = a.get("activity_type", "unknown")
            score = a.get("score", "?")
            insights = a.get("insights_generated", {})
            patterns = insights.get("key_patterns", [])
            act_lines.append(f"  {atype}: score={score}, patterns={patterns[:2]}")
        act_block = "\n".join(act_lines) if act_lines else "No activities yet."

        # Recent messages (last 5)
        recent = session.get("recent_messages", [])[-5:]
        conv_lines = []
        for m in recent:
            role = "User" if m.get("role") == "user" else "AI"
            conv_lines.append(f"  {role}: {m.get('content','')[:100]}")
        conv_block = "\n".join(conv_lines) if conv_lines else "New conversation."

        return f"""You are a clinical psychologist specialising in Indian youth (16-25).
Analyse this user and return ONLY valid JSON (no markdown fences) matching this schema:

{{
  "emotional_state": "<descriptive string>",
  "stress_categories": ["<Academic|Family|Social|Emotional|Identity|Career|Miscellaneous>"],
  "risk_assessment": "<low|moderate|high|crisis>",
  "coping_assessment": "<description of coping mechanisms & resilience>",
  "intervention_priority": "<immediate|supportive|long-term>",
  "psychological_insights": ["<insight1>", "<insight2>", "<insight3>"],
  "cultural_pressures": "<relevant Indian cultural/family/academic pressures>"
}}

─── DATA ───

USER MESSAGE: "{ctx['user_message'][:800]}"

NLP ANALYSIS:
  Primary emotion: {nlp.get('primary_emotion','unknown')}
  Sentiment: {nlp.get('sentiment',{}).get('label','unknown')} ({nlp.get('sentiment',{}).get('score',0):.2f})
  Intensity: {nlp.get('intensity',0):.2f}
  Urgency: {nlp.get('urgency_flag', False)}
  Key phrases: {nlp.get('key_phrases',[])}

CULTURAL CONTEXT:
  Language style: {cultural.get('language_style','unknown')}
  Cultural flags: {cultural.get('cultural_sensitivity_flags',[])}
  Communication: {cultural.get('communication_pattern','unknown')}
  Formality: {cultural.get('formality_level','medium')}

SESSION MEMORIES:
{mem_block}

RECENT ACTIVITIES:
{act_block}

RECENT CONVERSATION:
{conv_block}

JSON:"""

    def _parse_analysis(self, raw: str) -> Dict:
        defaults = {
            "emotional_state": "needs assessment",
            "stress_categories": [],
            "risk_assessment": "low",
            "coping_assessment": "",
            "intervention_priority": "supportive",
            "psychological_insights": [],
            "cultural_pressures": "",
        }
        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            parsed = json.loads(cleaned)
            for k, v in defaults.items():
                if k not in parsed:
                    parsed[k] = v
            return parsed
        except json.JSONDecodeError:
            logger.warning("[AGENT-1] JSON parse failed, using LLM text as insight")
            defaults["psychological_insights"] = [raw[:300]]
            return defaults


# ╔══════════════════════════════════════════════════════════════╗
# ║  6. GLM AGENT 2 — Psychological Technique Selector           ║
# ╚══════════════════════════════════════════════════════════════╝

class TechniqueSelectorAgent:
    """
    Reads the psychological analysis + NLP + cultural context from
    UserContext and selects the optimal therapeutic technique(s).
    Writes into user_context['technique_selection'].
    """
    def __init__(self, glm: GLMController):
        self.glm = glm
        logger.info("✅ [AGENT-2] Technique selector agent ready")

    def run(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("💊 [AGENT-2] Selecting therapeutic technique...")

        prompt = self._build_prompt(user_context)
        resp = self.glm.invoke([{"role": "user", "content": prompt}])

        if not resp or not resp.content:
            raise ValueError("[AGENT-2] GLM returned empty response")

        parsed = self._parse_selection(resp.content)
        user_context["technique_selection"] = parsed
        logger.info(f"✅ [AGENT-2] Technique={parsed.get('primary_technique','?')}")
        return user_context


    def _build_prompt(self, ctx: Dict) -> str:
        psych = ctx.get("psychological_analysis", {})
        nlp = ctx.get("nlp_analysis", {})
        cultural = ctx.get("cultural_context", {})

        return f"""You are a therapeutic technique advisor for Indian youth (16-25).
Based on the psychological assessment below, select the best therapeutic approach.
Return ONLY valid JSON (no markdown fences):

{{
  "primary_technique": "<CBT|ACT|MBCT|DBT|MI|Solution-Focused|Person-Centered|Psychoeducation>",
  "therapeutic_approach": "<brief description of how to apply this technique>",
  "activity_recommendations": ["<activity1>", "<activity2>", "<activity3>"],
  "rationale": "<why this technique suits the current situation>"
}}

─── ASSESSMENT ───

Emotional state: {psych.get('emotional_state','')}
Stress categories: {psych.get('stress_categories',[])}
Risk: {psych.get('risk_assessment','low')}
Intervention priority: {psych.get('intervention_priority','supportive')}
Insights: {psych.get('psychological_insights',[])}
Cultural pressures: {psych.get('cultural_pressures','')}

Emotion intensity: {nlp.get('intensity',0):.2f}
Primary emotion: {nlp.get('primary_emotion','unknown')}
Urgency: {nlp.get('urgency_flag', False)}

Language style: {cultural.get('language_style','casual')}
Cultural flags: {cultural.get('cultural_sensitivity_flags',[])}
Formality: {cultural.get('formality_level','medium')}

Consider Indian cultural context: family dynamics, academic pressure, mental health stigma.
Prefer culturally appropriate, practical activities (yoga, journaling, grounding exercises).

JSON:"""

    def _parse_selection(self, raw: str) -> Dict:
        defaults = {
            "primary_technique": "Person-Centered",
            "therapeutic_approach": "Empathetic listening with gentle exploration",
            "activity_recommendations": [],
            "rationale": "",
        }
        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            parsed = json.loads(cleaned)
            for k, v in defaults.items():
                if k not in parsed:
                    parsed[k] = v
            return parsed
        except json.JSONDecodeError:
            logger.warning("[AGENT-2] JSON parse failed, using defaults")
            return defaults


# ╔══════════════════════════════════════════════════════════════╗
# ║  7. GLM RESPONSE GENERATOR                                  ║
# ╚══════════════════════════════════════════════════════════════╝

class ResponseGenerator:
    """
    Final stage: reads the full UserContext JSON and generates a natural,
    culturally-sensitive, therapeutically-informed companion response.
    """

    SYSTEM_PROMPT = """You are MindMitra, a culturally-aware AI therapeutic companion for Indian youth (16-25).

RESPONSE RULES:
• Combine psychology expertise with warm, companion-style delivery
• Match the user's language style (if they use Hindi/Hinglish, mirror appropriately)
• Apply the selected therapeutic technique naturally — do NOT label techniques
• Reference session memories when relevant to show continuity
• Be empathetic, non-judgmental, like a caring friend who understands psychology
• Validate cultural struggles without dismissing traditional values
• Keep responses conversational — concise for casual chat, deeper for heavy topics
• NEVER include numbered annotations, technique labels in parentheses, or meta-commentary
• Generate ONLY the natural conversation response"""

    def __init__(self, glm: GLMController):
        self.glm = glm
        logger.info("✅ [RESPONSE-GEN] Response generator ready")

    def generate(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("💬 [RESPONSE-GEN] Generating therapeutic response...")

        system_msg = {"role": "system", "content": self.SYSTEM_PROMPT}
        human_msg = {"role": "user", "content": self._build_context(user_context)}

        resp = self.glm.invoke([system_msg, human_msg])

        if not resp or not resp.content:
            raise ValueError("[RESPONSE-GEN] GLM returned empty response")

        cleaned = self._clean(resp.content)
        user_context["ai_response"] = cleaned
        user_context["response_generated"] = True
        logger.info(f"✅ [RESPONSE-GEN] Response ready ({len(cleaned)} chars)")
        return user_context

    def _build_context(self, ctx: Dict) -> str:
        psych = ctx.get("psychological_analysis", {})
        technique = ctx.get("technique_selection", {})
        nlp = ctx.get("nlp_analysis", {})
        cultural = ctx.get("cultural_context", {})
        voice = ctx.get("voice_analysis", {})
        session = ctx.get("session_context", {})

        # Format recent messages for conversation flow
        recent = session.get("recent_messages", [])[-3:]
        conv = "\n".join(
            f"{'User' if m.get('role')=='user' else 'MindMitra'}: {m.get('content','')[:150]}"
            for m in recent
        )

        # Format key memories
        memories = session.get("session_memories", {})
        mem_lines = []
        for mtype in ("procedural", "semantic", "episodic"):
            for m in memories.get(mtype, [])[:3]:
                c = m.get("memory_content", m.get("content", ""))
                mem_lines.append(f"[{mtype}] {c[:100]}")
        mem_block = "\n".join(mem_lines) if mem_lines else ""

        voice_block = ""
        if voice:
            voice_block = f"""
VOICE ANALYSIS:
  Emotional tone: {voice.get('emotional_tone','N/A')}
  Stress level: {voice.get('stress_level','N/A')}
  Speech pace: {voice.get('speech_pace','N/A')}"""

        return f"""PSYCHOLOGICAL ASSESSMENT:
  State: {psych.get('emotional_state','')}
  Stress: {psych.get('stress_categories',[])}
  Priority: {psych.get('intervention_priority','')}
  Insights: {psych.get('psychological_insights',[])}
  Cultural pressures: {psych.get('cultural_pressures','')}

TECHNIQUE:
  Approach: {technique.get('primary_technique','')} — {technique.get('therapeutic_approach','')}
  Activities: {technique.get('activity_recommendations',[])}

EMOTION: {nlp.get('primary_emotion','?')} (intensity {nlp.get('intensity',0):.1f}), sentiment={nlp.get('sentiment',{}).get('label','neutral')}
LANGUAGE STYLE: {cultural.get('language_style','casual')}, formality={cultural.get('formality_level','medium')}
CULTURAL FLAGS: {cultural.get('cultural_sensitivity_flags',[])}
{voice_block}

{f'MEMORIES:{chr(10)}{mem_block}' if mem_block else ''}

CONVERSATION:
{conv if conv else '(New conversation)'}

USER'S CURRENT MESSAGE: "{ctx['user_message']}"

Respond naturally as MindMitra:"""

    def _clean(self, text: str) -> str:
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("{") or text.startswith("["):
            try:
                p = json.loads(text)
                if isinstance(p, dict) and "content" in p:
                    return p["content"]
            except json.JSONDecodeError:
                pass
        return text.strip()


# ╔══════════════════════════════════════════════════════════════╗
# ║  8. MAIN WORKFLOW ORCHESTRATOR                               ║
# ║     (preserves identical external API)                       ║
# ╚══════════════════════════════════════════════════════════════╝

class MindMitraWorkflow:
    """
    Orchestrates the full pipeline:
      1. Build UserContext JSON
      2. Fetch memories → populate session_context
      3. Groq NLP → populate nlp_analysis
      4. Cultural context → populate cultural_context
      5. GLM Agent 1 (Psychologist) → populate psychological_analysis
      6. GLM Agent 2 (Technique Selector) → populate technique_selection
      7. GLM Response Generator → populate ai_response
      8. Return result in the SAME format as before
    """

    def __init__(self):
        logger.info("🧠 [WORKFLOW] Initialising MindMitra v2 (modular architecture)...")

        # ── Supabase ──
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if supabase_url and supabase_key:
            self.supabase: Client = create_client(supabase_url, supabase_key)
            logger.info("✅ [WORKFLOW] Supabase client ready")
        else:
            self.supabase = None
            logger.warning("⚠️ [WORKFLOW] Supabase not configured")

        # ── Memory system (UNCHANGED) ──
        google_api_key = os.getenv("GOOGLE_API_KEY")
        try:
            if google_api_key:
                self.memory_system = UniversalMemorySystem(api_key=google_api_key)
                logger.info("✅ [WORKFLOW] Memory system ready")
            else:
                self.memory_system = None
        except Exception as e:
            self.memory_system = None
            logger.error(f"❌ [WORKFLOW] Memory system init failed: {e}")

        # ── Modules ──
        self.groq_nlp = GroqNLPModule()
        self.cultural_module = CulturalContextModule(groq_nlp=self.groq_nlp)
        self.glm = GLMController(api_key=google_api_key)
        self.agent_psychologist = PsychologistAnalysisAgent(self.glm)
        self.agent_technique = TechniqueSelectorAgent(self.glm)
        self.response_gen = ResponseGenerator(self.glm)

        # ── Background summarisation cache (same as original) ──
        self._summarization_cache = {}
        self._last_summarization_count = {}

        logger.info("✅ [WORKFLOW] MindMitra v2 fully initialised\n")

    # ══════════════════════════════════════════════════════════
    #  MEMORY METHODS — KEPT IDENTICAL TO ORIGINAL
    # ══════════════════════════════════════════════════════════
    def save_user_context_to_file(self, user_context: Dict[str, Any], file_name: str) -> None:
        """Save the processed user context to a file (JSON format)."""
        try:
            # Define the local directory where the file will be saved
            save_dir = "user_contexts"  # Folder where JSON files will be saved
            os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist

            # Define the full path to the file
            file_path = os.path.join(save_dir, file_name)

            # Save the context data to the file in JSON format
            with open(file_path, "w") as file:
                json.dump(user_context, file, indent=4)  # Save with indentation for readability

            logger.info(f"✅ [FILE] UserContext saved to {file_path}")
        except Exception as e:
            logger.error(f"❌ [FILE] Failed to save user context: {e}")
    def fetch_session_memories(self, session_id: str) -> Dict[str, List[Dict]]:
        """Fetch all memories for a session from database (UNCHANGED from v1)."""
        logger.info(f"🔍 [FETCH_MEMORIES] Fetching for session: {session_id}")
        if not self.supabase or not session_id:
            return {"procedural": [], "semantic": [], "episodic": []}

        try:
            response = (
                self.supabase.table("memories")
                .select("*")
                .eq("session_id", session_id)
                .order("created_at", desc=True)
                .execute()
            )

            if not response.data:
                return {"procedural": [], "semantic": [], "episodic": []}

            memories: Dict[str, List] = {"procedural": [], "semantic": [], "episodic": []}

            for row in response.data:
                for memory_type in ("procedural", "semantic", "episodic"):
                    column_name = f"{memory_type}_memories"
                    jsonb_data = row.get(column_name, [])
                    if isinstance(jsonb_data, str):
                        try:
                            jsonb_data = json.loads(jsonb_data)
                        except Exception:
                            jsonb_data = []
                    if isinstance(jsonb_data, list):
                        for mem in jsonb_data:
                            memories[memory_type].append({
                                "memory_content": mem.get("memory_content", mem.get("content", str(mem))),
                                "confidence": mem.get("confidence", mem.get("confidence_level", 0.5)),
                                "created_at": row.get("created_at"),
                                "memory_id": row.get("id"),
                                "importance": mem.get("importance", "medium"),
                                "category": mem.get("category", "general"),
                            })

            total = sum(len(v) for v in memories.values())
            logger.info(f"✅ [FETCH_MEMORIES] {total} memories (P={len(memories['procedural'])}, S={len(memories['semantic'])}, E={len(memories['episodic'])})")
            return memories

        except Exception as e:
            logger.error(f"❌ [FETCH_MEMORIES] Error: {e}")
            return {"procedural": [], "semantic": [], "episodic": []}

    def fetch_last_n_messages(self, session_id: str, n: int = 15) -> List[Dict]:
        """Fetch last N unprocessed messages (UNCHANGED from v1)."""
        if not self.supabase or not session_id:
            return []
        try:
            response = (
                self.supabase.table("chat_messages")
                .select("id, role, content, created_at")
                .eq("session_id", session_id)
                .eq("processed_into_memory", False)
                .order("created_at", desc=False)
                .limit(n)
                .execute()
            )
            return [
                {"id": r["id"], "role": r["role"], "content": r["content"], "timestamp": r["created_at"]}
                for r in response.data
            ]
        except Exception as e:
            logger.error(f"❌ [WORKFLOW] fetch messages error: {e}")
            return []

    def trigger_memory_extraction(self, session_id: str, user_id: str):
        """Trigger memory extraction — UNCHANGED from v1."""
        try:
            logger.info("=" * 60)
            logger.info(f"🧠 [MEMORY EXTRACTION] session={session_id}, user={user_id}")
            messages = self.fetch_last_n_messages(session_id, n=15)
            if not messages or not self.memory_system:
                return

            chat_data = {
                "data_type": "chat",
                "user_id": user_id,
                "session_id": session_id,
                "chat_history": messages,
            }

            result = self.memory_system.process_data_to_memories(chat_data)

            memory_record = {
                "user_id": user_id,
                "session_id": session_id,
                "data_type": "chat",
                "procedural_memories": result["memories"].get("procedural", []),
                "semantic_memories": result["memories"].get("semantic", []),
                "episodic_memories": result["memories"].get("episodic", []),
                "memory_summary": {
                    "procedural_count": len(result["memories"].get("procedural", [])),
                    "semantic_count": len(result["memories"].get("semantic", [])),
                    "episodic_count": len(result["memories"].get("episodic", [])),
                    "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "source_message_ids": [msg["id"] for msg in messages],
                "metadata": {"message_count": len(messages), "extraction_method": "parallel_llm"},
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
            self.supabase.table("memories").insert(memory_record).execute()

            message_ids = [msg["id"] for msg in messages]
            if message_ids:
                self.supabase.table("chat_messages").update(
                    {"processed_into_memory": True}
                ).in_("id", message_ids).execute()

            logger.info(f"✅ [MEMORY EXTRACTION] Done")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ [MEMORY EXTRACTION] Failed: {e}")
    
    # ══════════════════════════════════════════════════════════
    #  CORE PIPELINE
    # ══════════════════════════════════════════════════════════

    def process_chat(
        self,
        user_message: str,
        recent_messages: Optional[List] = None,
        conversation_summary: Optional[Dict] = None,
        user_activities: Optional[List] = None,
        user_patterns: Optional[Dict] = None,
        voice_analysis: Optional[Dict] = None,
        user_id: str = "anonymous",
        session_id: str = None,
    ) -> Dict[str, Any]:
        """
        Main processing pipeline — SAME SIGNATURE & RETURN FORMAT as original.
        Internally uses the new modular architecture.
        """
        start_time = datetime.now()

        # ── 1. Build UserContext JSON ─────────────────────────
        ctx = create_empty_user_context(user_id, session_id, user_message.strip())
        ctx["voice_analysis"] = voice_analysis or {}
        ctx["session_context"]["recent_messages"] = recent_messages or []
        ctx["session_context"]["conversation_summary"] = conversation_summary or {}
        ctx["session_context"]["user_activities"] = user_activities or []
        ctx["session_context"]["user_patterns"] = user_patterns or {}

        # ── 2. Fetch session memories → into ctx ─────────────
        if session_id:
            ctx["session_context"]["session_memories"] = self.fetch_session_memories(session_id)

        # ── 3. Groq NLP emotion/sentiment ─────────────────────
        try:
            ctx = self.groq_nlp.analyse(ctx)
        except Exception as e:
            logger.error(f"❌ [PIPELINE] NLP module error (non-fatal): {e}")

        # ── 4. Cultural context analysis ──────────────────────
        try:
            ctx = self.cultural_module.analyse(ctx)
        except Exception as e:
            logger.error(f"❌ [PIPELINE] Cultural module error (non-fatal): {e}")

        # ── 5. GLM Agent 1: Psychologist analysis ─────────────
        ctx = self.agent_psychologist.run(ctx)

        # ── 6. GLM Agent 2: Technique selection ───────────────
        ctx = self.agent_technique.run(ctx)

        # ── 7. GLM Response generation ────────────────────────
        ctx = self.response_gen.generate(ctx)

        processing_time = (datetime.now() - start_time).total_seconds()

        # ── 8. Build output in ORIGINAL FORMAT ────────────────
        psych = ctx["psychological_analysis"]
        technique = ctx["technique_selection"]
        self.save_user_context_to_file(ctx, f"user_context_{ctx['user_id']}_{ctx['session_id']}.json")  # Save context to file


        return {
            "message": ctx["ai_response"],
            "modality": technique.get("primary_technique", "Person-Centered"),
            "confidence": 0.9,
            "processing_time": processing_time,
            "session_insights": {
                "emotional_state": psych.get("emotional_state", ""),
                "stress_categories": psych.get("stress_categories", []),
                "therapeutic_approach": technique.get("primary_technique", ""),
                "cultural_pressures": psych.get("cultural_pressures", ""),
                "language_style": ctx["cultural_context"].get("language_style", ""),
                "psychological_insights": psych.get("psychological_insights", []),
                "coping_assessment": psych.get("coping_assessment", ""),
                "intervention_priority": psych.get("intervention_priority", ""),
                "activity_recommendations": technique.get("activity_recommendations", []),
                # Extra data available in v2
                "nlp_analysis": ctx["nlp_analysis"],
                "cultural_context": ctx["cultural_context"],
                "technique_rationale": technique.get("rationale", ""),
                "performance_metrics": {
                    "context_messages": len(ctx["session_context"]["recent_messages"]),
                    "context_activities": len(ctx["session_context"]["user_activities"]),
                    "has_summary": bool(ctx["session_context"]["conversation_summary"]),
                    "memory_count": sum(
                        len(v) for v in ctx["session_context"]["session_memories"].values()
                    ),
                },
            },
        }


# ╔══════════════════════════════════════════════════════════════╗
# ║  9. GLOBAL INSTANCE & ENTRY POINT (UNCHANGED SIGNATURE)      ║
# ╚══════════════════════════════════════════════════════════════╝

_workflow_instance = None


def get_workflow_instance() -> MindMitraWorkflow:
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = MindMitraWorkflow()
    return _workflow_instance


def process_user_chat(
    user_message: str,
    recent_messages: Optional[List] = None,
    conversation_summary: Optional[Dict] = None,
    user_activities: Optional[List] = None,
    user_patterns: Optional[Dict] = None,
    voice_analysis: Optional[Dict] = None,
    user_id: str = "anonymous",
    session_id: str = None,
) -> Dict[str, Any]:
    """Main entry point — IDENTICAL SIGNATURE to original v1."""

    logger.info(f"🚀 [ENTRY] MindMitra v2 — user={user_id}, session={session_id}")
    start_time = time.time()

    try:
        workflow = get_workflow_instance()
        result = workflow.process_chat(
            user_message, recent_messages, conversation_summary,
            user_activities, user_patterns, voice_analysis, user_id, session_id,
        )
        result["processing_time"] = round(time.time() - start_time, 2)
        result["voice_aware"] = bool(voice_analysis)
        logger.info(f"✅ [ENTRY] Done in {result['processing_time']}s")
        return result

    except Exception as e:
        logger.error(f"❌ [ENTRY] Failed after {time.time()-start_time:.2f}s: {e}")
        raise
