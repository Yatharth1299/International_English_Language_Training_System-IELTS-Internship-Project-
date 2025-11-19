# agents/speaking_agent.py
import os
import re
import json
import tempfile
import logging
from typing import TypedDict, Dict, Any, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

# Use your project's config which already configured genai with the API key
# (your config.py contains: import google.generativeai as genai; genai.configure(api_key=...))
from config import genai  # type: ignore

# Use your ASR service (must exist in services/asr_service.py)
from services.asr_service import transcribe_audio

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")


# ---- State typing ----
class SpeakingState(TypedDict, total=False):
    test_id: str
    user_id: str
    responses: Dict[str, Any]    # e.g. {"part_1": "path_or_url", ...}
    transcripts: Dict[str, str]
    per_part: Dict[str, Dict[str, Any]]
    aggregated: Dict[str, Any]


# ---- Helpers ----
def _download_to_temp(url: str) -> str:
    import requests
    logger.info("Downloading remote audio: %s", url)
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    ext = os.path.splitext(url.split("?")[0])[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            tmp.write(chunk)
    tmp.flush()
    tmp.close()
    return tmp.name


def _safe_transcribe(src: Any) -> str:
    """
    Accepts local file path or dict with 'audio_url' or http(s) url string.
    """
    try:
        if isinstance(src, dict) and src.get("audio_url"):
            src = src.get("audio_url")
        if isinstance(src, str) and src.startswith(("http://", "https://")):
            path = _download_to_temp(src)
            try:
                return transcribe_audio(path)
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
        if isinstance(src, str):
            return transcribe_audio(src)
        raise ValueError("Unsupported audio source type for transcription")
    except Exception as e:
        logger.exception("Transcription failed for source %s: %s", str(src), e)
        return f"ERROR: {str(e)}"


def _extract_text_from_genai_response(resp: Any) -> str:
    # Many genai responses contain candidates[].content (string) — adapt if SDK differs
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            cand = resp.candidates[0]
            content = getattr(cand, "content", None)
            if isinstance(content, str):
                return content
            if hasattr(content, "text"):
                return content.text
            if hasattr(content, "parts") and content.parts:
                texts = [getattr(p, "text", "") for p in content.parts]
                return "\n".join([t for t in texts if t])
        return str(resp)
    except Exception:
        try:
            return json.dumps(resp)
        except Exception:
            return str(resp)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    # remove fences
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\})", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except Exception:
            return None


def _round_half(x: float) -> float:
    return round(x * 2) / 2.0


# ---- Prompt builder (few-shot) ----
def _build_evaluation_prompt(transcripts: Dict[str, str]) -> str:
    examples = [
        {
            "transcript": "I live in a small town. I like to read books and sometimes go cycling.",
            "json": {
                "fluency": 6,
                "coherence": 6,
                "lexical_resource": 6,
                "grammar": 6,
                "pronunciation": 6,
                "feedback": {
                    "fluency": "Generally fluent with occasional hesitation.",
                    "coherence": "Simple connected ideas.",
                    "lexical_resource": "Basic vocabulary but appropriate.",
                    "grammar": "Some grammatical errors in complex sentences.",
                    "pronunciation": "Mostly intelligible."
                },
                "band": 6.0
            }
        },
        {
            "transcript": "Travel broadened my horizons; I learned new cultures and realized how people live differently.",
            "json": {
                "fluency": 7,
                "coherence": 7,
                "lexical_resource": 7,
                "grammar": 7,
                "pronunciation": 7,
                "feedback": {
                    "fluency": "Mostly fluent with natural phrasing.",
                    "coherence": "Ideas are well connected.",
                    "lexical_resource": "Good range and collocations.",
                    "grammar": "Accurate grammar overall.",
                    "pronunciation": "Clear and easy to understand."
                },
                "band": 7.0
            }
        },
    ]

    parts_lines = [f"{p}: \"{t}\"" for p, t in transcripts.items()]
    prompt_parts = [
        "You are an experienced IELTS Speaking examiner.",
        "Evaluate the transcripts below and return EXACTLY one JSON object with keys: per_part, aggregated.",
        "per_part should map each part key (e.g. part_1) to an object with scores (fluency, coherence, lexical_resource, grammar, pronunciation), a 'feedback' object with one-sentence feedback per category, and a 'band' number.",
        "aggregated should contain averaged numeric scores and band. Band must be rounded to nearest 0.5.",
        "Return ONLY the JSON object (no extra text).",
        "",
        "FEW-SHOT EXAMPLES:"
    ]
    for ex in examples:
        prompt_parts.append(f"Transcript: \"{ex['transcript']}\"\nOutputJSON:\n{json.dumps(ex['json'], ensure_ascii=False)}")
        prompt_parts.append("---")

    prompt_parts.append("Now evaluate these transcripts. Return ONLY a single JSON object.")
    prompt_parts.append("Transcripts:")
    prompt_parts.append("\n".join(parts_lines))
    prompt_parts.append("\nInstructions: Scores must be integers 0-9. Band = average of five categories rounded to nearest 0.5. Feedback sentences should be short.")
    return "\n\n".join(prompt_parts)


# ---- LangGraph node: transcribe ----
def transcribe_node(state: SpeakingState) -> SpeakingState:
    logger.info("Node: transcribe")
    responses = state.get("responses", {}) or {}
    transcripts: Dict[str, str] = {}
    for part, src in responses.items():
        transcripts[part] = _safe_transcribe(src)
    state["transcripts"] = transcripts
    return state


# ---- LangGraph node: evaluate ----
def evaluate_node(state: SpeakingState) -> SpeakingState:
    logger.info("Node: evaluate")
    transcripts = state.get("transcripts", {}) or {}
    if not transcripts:
        raise ValueError("No transcripts available for evaluation.")

    prompt = _build_evaluation_prompt(transcripts)
    logger.info("Calling Gemini model for evaluation...")
    model = genai.GenerativeModel(LLM_MODEL)
    resp = model.generate_content(prompt)
    raw_text = _extract_text_from_genai_response(resp)
    logger.debug("Raw LLM response (truncated): %s", raw_text[:800])

    parsed = _extract_json(raw_text)
    per_part_eval: Dict[str, Dict[str, Any]] = {}
    aggregated: Dict[str, Any] = {}

    if parsed and "per_part" in parsed and "aggregated" in parsed:
        per_part_eval = parsed["per_part"]
        aggregated = parsed["aggregated"]
    else:
        # If parsed as aggregated only or failed to parse, evaluate per part individually
        if parsed and all(k in parsed for k in ("fluency", "coherence", "lexical_resource", "grammar", "pronunciation")):
            aggregated = parsed
        # per-part evaluation:
        for p, txt in transcripts.items():
            small_prompt = (
                "You are an IELTS Speaking examiner. Return ONLY JSON with keys: fluency, coherence, lexical_resource, grammar, pronunciation, feedback (object), band.\n"
                f"Transcript: \"{txt}\""
            )
            resp_p = model.generate_content(small_prompt)
            raw_p = _extract_text_from_genai_response(resp_p)
            parsed_p = _extract_json(raw_p) or {}
            per_part_eval[p] = parsed_p

        if not aggregated:
            aggregated = _aggregate_scores(per_part_eval)

    # Ensure aggregated is present
    if not aggregated and per_part_eval:
        aggregated = _aggregate_scores(per_part_eval)

    state["per_part"] = per_part_eval
    state["aggregated"] = aggregated
    return state


def _aggregate_scores(per_part_eval: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cats = ["fluency", "coherence", "lexical_resource", "grammar", "pronunciation"]
    sums = {c: 0.0 for c in cats}
    n = 0
    for part, obj in per_part_eval.items():
        n += 1
        for c in cats:
            try:
                sums[c] += float(obj.get(c, 0))
            except Exception:
                sums[c] += 0.0
    if n == 0:
        return {}
    avg = {c: round(sums[c] / n, 1) for c in cats}
    band = _round_half(sum(avg[c] for c in cats) / len(cats))
    return {**avg, "band": band}


# ---- Build LangGraph ----
graph = StateGraph(SpeakingState)
graph.add_node("transcribe", transcribe_node)
graph.add_node("evaluate", evaluate_node)
graph.set_entry_point("transcribe")
graph.add_edge("transcribe", "evaluate")
graph.add_edge("evaluate", END)
speaking_agent = graph.compile()


# ---- Output formatter ----
def format_output(state: SpeakingState) -> Dict[str, Any]:
    transcripts = state.get("transcripts", {})
    per_part = state.get("per_part", {})
    aggregated = state.get("aggregated", {})

    cats = ["fluency", "coherence", "lexical_resource", "grammar", "pronunciation"]
    feedback_out: Dict[str, str] = {}
    for c in cats:
        parts_texts = []
        for p, obj in per_part.items():
            fb = obj.get("feedback", {})
            if isinstance(fb, dict):
                t = fb.get(c, "")
            else:
                t = obj.get("feedback", "")
            if t:
                parts_texts.append(f"{p}: {t}")
        feedback_out[c] = " ".join(parts_texts) if parts_texts else ""

    score_obj = {
        "band": aggregated.get("band"),
        "fluency": aggregated.get("fluency"),
        "coherence": aggregated.get("coherence"),
        "lexical_resource": aggregated.get("lexical_resource"),
        "grammar": aggregated.get("grammar"),
        "pronunciation": aggregated.get("pronunciation"),
    }

    return {
        "test_id": state.get("test_id"),
        "user_id": state.get("user_id"),
        "transcripts": transcripts,
        "score": score_obj,
        "feedback": feedback_out,
        "per_part": per_part,
        "aggregated": aggregated
    }
