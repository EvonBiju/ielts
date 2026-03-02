from fastapi import FastAPI, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import json
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IELTS Listening Module")
templates = Jinja2Templates(directory="templates")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
MAX_RETRIES = 3


def build_prompt(section_number: int, topic: str) -> str:
    section_contexts = {
        1: "a realistic phone conversation between a customer and a service agent (e.g. booking a hotel, registering for a course, making an appointment). Include natural dialogue with names, dates, numbers, prices.",
        2: "a spoken monologue by a community speaker (e.g. a tour guide, a radio presenter, a facility manager). Include facts, descriptions, times, and specific details.",
        3: "an academic conversation between 2-3 university students and/or a tutor discussing a research project, assignment, or presentation. Include academic vocabulary, opinions, and decisions.",
        4: "an academic lecture by a professor on a complex topic such as climate science, neuroscience, urban planning, or economics. Include technical terms, statistics, and structured arguments."
    }
    topic_hint = f"The topic must be about: {topic}. " if topic else ""
    ctx = section_contexts[section_number]
    q_start = (section_number - 1) * 5 + 1

    return f"""You are a professional IELTS exam writer with 10 years of experience. Your task is to generate one complete section of an IELTS Listening test.

SECTION: {section_number} of 4
CONTEXT: {ctx}
{topic_hint}
DIFFICULTY: {"Easy" if section_number == 1 else "Medium" if section_number == 2 else "Hard" if section_number == 3 else "Very Hard"}

INSTRUCTIONS:
- Write a high-quality ttsScript of exactly 300-400 words. It must sound like real spoken English, not written text.
- Create exactly 5 questions that test specific details from the ttsScript.
- Questions must be genuinely challenging and based on real content in the script.
- fill_blank answers must appear word-for-word in the ttsScript.
- All acceptedAnswers must be lowercase.
- Do NOT use generic placeholder text. Every field must contain real, specific IELTS-quality content.

OUTPUT: Return ONLY a valid JSON object. No markdown. No explanation. No text outside the JSON.

JSON STRUCTURE (replace every value with real content):
{{
  "id": {section_number},
  "title": "Section {section_number}",
  "context": "<one specific sentence describing exactly what the listener will hear>",
  "ttsScript": "<full 300-400 word realistic spoken script — this is what gets read aloud>",
  "questions": [
    {{
      "id": {q_start},
      "type": "fill_blank",
      "prompt": "<sentence from script with one key word replaced by ___>",
      "wordLimit": "ONE WORD",
      "answer": "<the missing word, lowercase>",
      "acceptedAnswers": ["<answer>", "<alternate spelling if any>"]
    }},
    {{
      "id": {q_start + 1},
      "type": "fill_blank",
      "prompt": "<different sentence from script with a number or name replaced by ___>",
      "wordLimit": "ONE WORD OR NUMBER",
      "answer": "<the missing value, lowercase>",
      "acceptedAnswers": ["<answer>", "<word form if number>"]
    }},
    {{
      "id": {q_start + 2},
      "type": "multiple_choice",
      "prompt": "<a specific question about a detail or opinion in the script>",
      "options": ["A. <plausible wrong answer>", "B. <correct answer>", "C. <plausible wrong answer>", "D. <plausible wrong answer>"],
      "answer": "B",
      "acceptedAnswers": ["B"]
    }},
    {{
      "id": {q_start + 3},
      "type": "matching",
      "prompt": "Match each item to the correct description mentioned in the recording.",
      "items": ["<specific item from script>", "<specific item from script>", "<specific item from script>"],
      "options": ["<description from script>", "<description from script>", "<description from script>"],
      "answer": {{
        "<item1>": "<correct description>",
        "<item2>": "<correct description>",
        "<item3>": "<correct description>"
      }}
    }},
    {{
      "id": {q_start + 4},
      "type": "diagram_label",
      "prompt": "Label the key components described in the recording.",
      "diagramType": "triangle",
      "labels": ["Label A", "Label B", "Label C"],
      "options": ["<component from script>", "<component from script>", "<component from script>"],
      "answer": {{
        "Label A": "<component>",
        "Label B": "<component>",
        "Label C": "<component>"
      }}
    }}
  ]
}}

IMPORTANT: Every single value in the JSON must be original, specific, and derived from the ttsScript content. The script must be long enough and detailed enough that all 5 answers can be heard clearly in it."""


def extract_json(raw: str) -> dict:
    cleaned = re.sub(r'```json\s*', '', raw)
    cleaned = re.sub(r'```\s*', '', cleaned).strip()

    brace_match = re.search(r'\{[\s\S]*\}', cleaned)
    candidate = brace_match.group(0) if brace_match else cleaned

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Fix single quotes
    fixed2 = fixed.replace("'", '"')
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse JSON. Error: {e}. Raw (first 500 chars): {raw[:500]}")


def validate_section(data: dict, section_number: int) -> dict:
    data["id"] = section_number
    data.setdefault("title", f"Section {section_number}")
    data.setdefault("context", "")

    if "ttsScript" not in data:
        raise ValueError("Missing ttsScript")
    if len(data["ttsScript"]) < 100:
        raise ValueError(f"ttsScript too short: {len(data['ttsScript'])} chars")
    if "questions" not in data or not isinstance(data["questions"], list) or len(data["questions"]) == 0:
        raise ValueError("Missing or empty questions")

    # Reject if model returned placeholder text
    placeholder_phrases = [
        "the ticket costs 50 pounds",
        "the event starts at",
        "option one", "option two",
        "description a", "description b",
        "item 1", "item 2", "item 3",
        "part one", "part two", "part three"
    ]
    script_lower = data["ttsScript"].lower()
    for phrase in placeholder_phrases:
        if phrase in script_lower:
            raise ValueError(f"Model returned placeholder text: '{phrase}'")

    for q in data["questions"]:
        q.setdefault("type", "fill_blank")
        q.setdefault("prompt", "Answer the question.")
        # Reject placeholder prompts
        if "ticket costs 50" in q.get("prompt", "").lower():
            raise ValueError("Placeholder prompt detected")
        if q["type"] == "fill_blank":
            q.setdefault("answer", "")
            q.setdefault("acceptedAnswers", [q["answer"].lower()])
            q.setdefault("wordLimit", "ONE WORD")
        elif q["type"] == "multiple_choice":
            q.setdefault("options", ["A. A", "B. B", "C. C", "D. D"])
            q.setdefault("answer", "A")
            q.setdefault("acceptedAnswers", [q["answer"]])
            # Normalize options: add A/B/C/D prefix if model omitted it
            letters = ["A", "B", "C", "D"]
            import re as _re
            normalized = []
            for i, opt in enumerate(q["options"][:4]):
                if not _re.match(r'^[A-D][.)\s]', str(opt).strip()):
                    opt = f"{letters[i]}. {opt}"
                normalized.append(opt)
            q["options"] = normalized
            # Normalize answer: if model gave full text instead of letter, convert
            ans = q["answer"].strip()
            if len(ans) > 1:
                # Try to find which option matches
                for i, opt in enumerate(q["options"]):
                    if ans.lower() in opt.lower():
                        q["answer"] = letters[i]
                        q["acceptedAnswers"] = [letters[i]]
                        break
                else:
                    q["answer"] = "A"
                    q["acceptedAnswers"] = ["A"]
        elif q["type"] == "matching":
            q.setdefault("items", [])
            q.setdefault("options", [])
            q.setdefault("answer", {})
        elif q["type"] == "diagram_label":
            q.setdefault("labels", ["Label A", "Label B", "Label C"])
            q.setdefault("options", [])
            q.setdefault("answer", {})
            q.setdefault("diagramType", "triangle")

    return data


class GenerateRequest(BaseModel):
    section_number: int
    topic: str = ""

class ScoreRequest(BaseModel):
    sections: list
    answers: dict


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/generate-section")
async def generate_section(req: GenerateRequest):
    if req.section_number < 1 or req.section_number > 4:
        raise HTTPException(status_code=400, detail="Section must be 1-4")

    prompt = build_prompt(req.section_number, req.topic)
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Section {req.section_number} — attempt {attempt}/{MAX_RETRIES}")
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(OLLAMA_URL, json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5 if attempt > 1 else 0.75,
                        "top_p": 0.95,
                        "num_predict": 3500
                    }
                })
                result = response.json()
                raw = result.get("response", "").strip()
                logger.info(f"Raw response (first 400): {raw[:400]}")

            data = extract_json(raw)
            data = validate_section(data, req.section_number)
            logger.info(f"Section {req.section_number} generated successfully on attempt {attempt}")
            return JSONResponse(content={"success": True, "section": data})

        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Run: ollama serve")
        except (ValueError, KeyError) as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt} failed: {last_error}")
        except Exception as e:
            last_error = str(e)
            logger.error(f"Unexpected error attempt {attempt}: {last_error}")

    raise HTTPException(
        status_code=500,
        detail=f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


@app.post("/api/score")
async def score_test(req: ScoreRequest):
    total = 0
    results = []
    for section in req.sections:
        sec_correct = 0
        sec_results = []
        for q in section["questions"]:
            qid = str(q["id"])
            user_ans = req.answers.get(qid, "")
            correct = False
            if q["type"] == "fill_blank":
                accepted = [a.lower().strip() for a in q.get("acceptedAnswers", [])]
                correct = str(user_ans).strip().lower() in accepted
            elif q["type"] == "multiple_choice":
                correct = user_ans == q["answer"]
            elif q["type"] in ("matching", "diagram_label"):
                keys = q.get("items") or q.get("labels", [])
                user_obj = user_ans if isinstance(user_ans, dict) else {}
                correct = all(user_obj.get(k) == q["answer"].get(k) for k in keys)
            if correct:
                sec_correct += 1
                total += 1
            sec_results.append({
                "id": q["id"], "correct": correct,
                "user": user_ans, "expected": q["answer"], "prompt": q["prompt"]
            })
        results.append({
            "section_id": section["id"],
            "title": section.get("title", f"Section {section['id']}"),
            "correct": sec_correct,
            "total": len(section["questions"]),
            "questions": sec_results
        })

    return JSONResponse(content={
        "total": total,
        "max": sum(len(s["questions"]) for s in req.sections),
        "band": get_band(total),
        "sections": results
    })


def get_band(score):
    if score >= 39: return "9.0"
    if score >= 37: return "8.5"
    if score >= 35: return "8.0"
    if score >= 32: return "7.5"
    if score >= 30: return "7.0"
    if score >= 26: return "6.5"
    if score >= 23: return "6.0"
    if score >= 18: return "5.5"
    if score >= 16: return "5.0"
    if score >= 13: return "4.5"
    return "4.0"
