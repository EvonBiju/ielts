from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import httpx
import json
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IELTS Speaking v2")
app.mount("/static", StaticFiles(directory="static"), name="static")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

# ── Models ─────────────────────────────────────────────────────────────────────
class GeneratePromptReq(BaseModel):
    part: int
    topic: Optional[str] = ""
    previous_topic: Optional[str] = ""

class EvaluateReq(BaseModel):
    part: int
    prompt: str
    transcript: str
    conversation_history: Optional[list] = []

class FollowUpReq(BaseModel):
    part: int
    topic: str
    question: str
    transcript: str
    key_points: Optional[List[str]] = []
    conversation_history: Optional[list] = []

# ── Ollama ─────────────────────────────────────────────────────────────────────
async def ask_ollama(prompt: str, max_tokens: int = 1400) -> str:
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.45,
                "num_predict": max_tokens,
                "stop": ["```\n", "\nNote:", "\nPlease note", "\nI hope", "\nThis "]
            }
        })
        resp.raise_for_status()
        return resp.json()["response"].strip()

# ── JSON extraction with repair ────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    t = text.strip()

    # Strip markdown fences
    t = re.sub(r'```json\s*', '', t)
    t = re.sub(r'```\s*', '', t)
    t = t.strip()

    # 1. Direct parse
    try:
        return json.loads(t)
    except Exception:
        pass

    # 2. Find first { to last }
    start = t.find('{')
    end   = t.rfind('}')
    if start != -1 and end > start:
        candidate = t[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
        # 3. Try repairing truncated JSON
        repaired = repair_json(candidate)
        if repaired:
            return repaired

    raise ValueError(f"Cannot parse JSON from model output. First 400 chars:\n{text[:400]}")

def repair_json(s: str) -> dict | None:
    """Close unclosed brackets/braces and strings to fix truncated JSON."""
    try:
        s = s.rstrip().rstrip(',')
        # Fix unclosed string
        in_str = False
        escaped = False
        for ch in s:
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == '"':
                in_str = not in_str
        if in_str:
            s += '"'
        # Close brackets
        arr_opens = s.count('[') - s.count(']')
        obj_opens = s.count('{') - s.count('}')
        s += ']' * max(0, arr_opens)
        s += '}' * max(0, obj_opens)
        return json.loads(s)
    except Exception:
        return None

# ── Default fallback skeleton ──────────────────────────────────────────────────
def default_eval(wc: int = 0) -> dict:
    return {
        "band_scores": {
            "fluency_coherence": 5.0, "lexical_resource": 5.0,
            "grammatical_range": 5.0, "pronunciation": 5.0, "overall": 5.0
        },
        "pronunciation_issues": [],
        "grammar_errors": [],
        "vocabulary_upgrades": [],
        "fluency_analysis": {
            "total_words": wc, "speaking_pace": "moderate",
            "filler_words_found": [], "filler_count": 0, "repetitions": [],
            "long_pauses_indicated": "no", "discourse_markers_used": [],
            "missing_discourse_markers": ["however", "furthermore", "in addition"],
            "coherence_rating": 5,
            "coherence_feedback": "Partial evaluation — please try again."
        },
        "content_analysis": {
            "question_addressed": "partially", "key_points_covered": [],
            "missed_opportunities": [], "idea_development": "adequate",
            "specific_feedback": "Evaluation partially completed."
        },
        "strengths": ["Attempted to answer the question."],
        "overall_feedback": "Partial evaluation due to model output length. Please try again.",
        "alternate_best_reply": {
            "intro": "", "development": "", "example": "",
            "contrast_or_concession": "", "conclusion": "", "band_features": []
        }
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── GENERATE PROMPT ────────────────────────────────────────────────────────────
@app.post("/api/generate-prompt")
async def generate_prompt(req: GeneratePromptReq):
    topic_hint = f"Topic: {req.topic}" if req.topic else "Pick a fresh engaging IELTS topic"
    prev_hint  = f"Previous topic was '{req.previous_topic}' — pick a DIFFERENT one." if req.previous_topic else ""

    if req.part == 1:
        prompt = f"""You are an IELTS examiner. {topic_hint}. {prev_hint}
Generate 5 Part 1 personal everyday questions. Return ONLY this raw JSON with no extra text:
{{"part":1,"topic":"topic name","context":"Good morning. I'd like to ask you some questions about [topic].","questions":["Q1?","Q2?","Q3?","Q4?","Q5?"]}}"""

    elif req.part == 2:
        prompt = f"""You are an IELTS examiner. {topic_hint}. {prev_hint}
Generate a Part 2 cue card. Return ONLY this raw JSON with no extra text:
{{"part":2,"topic":"topic name","context":"I'm going to give you a topic. You have one minute to prepare.","cue_card":{{"title":"Describe a [X] that [Y]","points":["What it is and how you encountered it","When and where it happened","Who was involved","Why it was significant to you"]}},"follow_up_hint":"category"}}"""

    else:
        prev = req.previous_topic or "society"
        prompt = f"""You are an IELTS examiner. Previous Part 2 topic: '{prev}'.
Generate 4 Part 3 analytical questions extending that topic. Return ONLY this raw JSON:
{{"part":3,"topic":"topic name","context":"Now I would like to discuss some broader issues related to {prev}.","questions":["Analytical Q1?","Comparative Q2?","Future/prediction Q3?","Critical thinking Q4?"]}}"""

    try:
        raw  = await ask_ollama(prompt, max_tokens=500)
        logger.info(f"[generate-prompt p{req.part}] raw: {raw[:200]}")
        data = extract_json(raw)
        return {"success": True, "prompt_data": data}
    except Exception as e:
        logger.error(f"[generate-prompt] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── EVALUATE ── split into 3 focused Mistral calls ────────────────────────────
@app.post("/api/evaluate")
async def evaluate(req: EvaluateReq):
    tx  = req.transcript.strip()
    wc  = len(tx.split())
    q   = req.prompt
    result = default_eval(wc)

    # ────────────────────────────────────────────────────────────────────────────
    # CALL 1 — Band scores + pronunciation + grammar
    # ────────────────────────────────────────────────────────────────────────────
    p1 = f"""IELTS examiner task. Part {req.part} speaking evaluation.
Question asked: {q}
Candidate said: "{tx}"

Return ONLY this raw JSON. Fill values based on the actual transcript:
{{
  "band_scores": {{
    "fluency_coherence": 6.0,
    "lexical_resource": 6.0,
    "grammatical_range": 6.0,
    "pronunciation": 6.0,
    "overall": 6.0
  }},
  "pronunciation_issues": [
    {{
      "word_said": "word from transcript",
      "correct_word": "correct spelling",
      "common_mistake": "e.g. stress on wrong syllable or dropped consonant",
      "ipa_wrong": "/rong/",
      "ipa_correct": "/kəˈrekt/",
      "syllable_breakdown": "cor·rect",
      "stress_pattern": "cor·RECT",
      "tip": "one specific tip for this word",
      "audio_word": "correct"
    }}
  ],
  "grammar_errors": [
    {{
      "original_phrase": "exact phrase from transcript",
      "error_type": "Tense Error",
      "corrected_phrase": "corrected phrase",
      "explanation": "grammar rule explanation",
      "rule_name": "rule name",
      "native_example_wrong": "another learner mistake example",
      "native_example_right": "corrected version",
      "band_impact": "Band 5 error"
    }}
  ]
}}
RULES: pronunciation_issues = only words non-native speakers commonly mispronounce (empty array if none). grammar_errors = only real errors found in the transcript (empty array if none). band_scores = score 1.0 to 9.0 per IELTS descriptors."""

    try:
        raw1 = await ask_ollama(p1, max_tokens=1100)
        logger.info(f"[eval c1] raw: {raw1[:300]}")
        d1 = extract_json(raw1)
        if "band_scores" in d1:
            result["band_scores"] = d1["band_scores"]
        result["pronunciation_issues"] = d1.get("pronunciation_issues", [])
        result["grammar_errors"]       = d1.get("grammar_errors", [])
    except Exception as e:
        logger.error(f"[eval c1] failed: {e}")

    # ────────────────────────────────────────────────────────────────────────────
    # CALL 2 — Vocabulary + fluency + content + strengths + overall
    # ────────────────────────────────────────────────────────────────────────────
    p2 = f"""IELTS examiner task. Analyse vocabulary, fluency, and content.
Question: {q}
Transcript: "{tx}"

Return ONLY this raw JSON:
{{
  "vocabulary_upgrades": [
    {{
      "used_word": "basic word/phrase from transcript",
      "context_in_speech": "full sentence they used it in",
      "why_basic": "why this word limits band score",
      "band5_word": "basic word",
      "band7_alternatives": ["alt1", "alt2", "alt3"],
      "band8_phrase": "sophisticated collocation or phrase",
      "example_with_context": "Band 8 sentence for this same topic context"
    }}
  ],
  "fluency_analysis": {{
    "total_words": {wc},
    "speaking_pace": "moderate",
    "filler_words_found": ["um", "uh"],
    "filler_count": 2,
    "repetitions": ["phrase repeated if any"],
    "long_pauses_indicated": "no",
    "discourse_markers_used": ["marker1"],
    "missing_discourse_markers": ["furthermore", "in contrast"],
    "coherence_rating": 6,
    "coherence_feedback": "specific feedback referencing their answer structure"
  }},
  "content_analysis": {{
    "question_addressed": "yes",
    "key_points_covered": ["point 1 they made", "point 2"],
    "missed_opportunities": ["angle they could have explored"],
    "idea_development": "adequate",
    "specific_feedback": "specific content feedback referencing exact things they said"
  }},
  "strengths": [
    "specific strength with reference to their words",
    "second strength"
  ],
  "overall_feedback": "3-4 sentences referencing specific things the candidate said"
}}
RULES: vocabulary_upgrades = exactly 3 items, picking the most impactful basic words from the transcript. All text must reference the actual transcript."""

    try:
        raw2 = await ask_ollama(p2, max_tokens=1100)
        logger.info(f"[eval c2] raw: {raw2[:300]}")
        d2 = extract_json(raw2)
        result["vocabulary_upgrades"] = d2.get("vocabulary_upgrades", [])
        result["fluency_analysis"]    = d2.get("fluency_analysis", result["fluency_analysis"])
        result["content_analysis"]    = d2.get("content_analysis", result["content_analysis"])
        result["strengths"]           = d2.get("strengths", result["strengths"])
        result["overall_feedback"]    = d2.get("overall_feedback", result["overall_feedback"])
    except Exception as e:
        logger.error(f"[eval c2] failed: {e}")

    # ────────────────────────────────────────────────────────────────────────────
    # CALL 3 — Alternate Band 8 reply
    # ────────────────────────────────────────────────────────────────────────────
    p3 = f"""Write a Band 8 IELTS speaking answer to this exact question: "{q}"
Context — the candidate said: "{tx[:250]}"

Return ONLY this raw JSON:
{{
  "alternate_best_reply": {{
    "intro": "opening sentence that directly answers the question",
    "development": "2-3 sentences with specific detail, evidence or reasoning",
    "example": "a concrete real-world or personal example",
    "contrast_or_concession": "a nuance or opposing view showing critical thinking",
    "conclusion": "a thoughtful closing sentence",
    "band_features": ["feature1 e.g. sophisticated collocation", "feature2", "feature3"]
  }}
}}"""

    try:
        raw3 = await ask_ollama(p3, max_tokens=700)
        logger.info(f"[eval c3] raw: {raw3[:300]}")
        d3 = extract_json(raw3)
        result["alternate_best_reply"] = d3.get("alternate_best_reply", result["alternate_best_reply"])
    except Exception as e:
        logger.error(f"[eval c3] failed: {e}")

    return {"success": True, "evaluation": result}


# ── FOLLOW-UP ──────────────────────────────────────────────────────────────────
@app.post("/api/followup")
async def get_followup(req: FollowUpReq):
    key_pts = ", ".join(req.key_points) if req.key_points else ""
    kp_line = f"Key points they made: {key_pts}" if key_pts else ""

    prompt = f"""You are an IELTS examiner. Generate a content-based follow-up question.
Original question: "{req.question}"
Candidate answered: "{req.transcript}"
{kp_line}

The examiner_comment MUST quote or paraphrase something specific the candidate said.
Return ONLY this raw JSON:
{{
  "examiner_comment": "You mentioned [specific thing from their answer]. That's interesting.",
  "followup_question": "follow-up question building on their specific answer?",
  "what_to_probe": "what aspect is being explored",
  "ideal_answer_hint": "what a Band 8 answer would include"
}}"""

    try:
        raw  = await ask_ollama(prompt, max_tokens=350)
        logger.info(f"[followup] raw: {raw[:200]}")
        data = extract_json(raw)
        return {"success": True, "followup": data}
    except Exception as e:
        logger.error(f"[followup] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
