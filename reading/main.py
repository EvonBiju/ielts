from fastapi import FastAPI, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx, json, re, logging, random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IELTS Reading Module")
templates = Jinja2Templates(directory="templates")

OLLAMA_URL = "http://localhost:11434"
MAX_RETRIES = 3

PASSAGE_CONFIGS = {
    1: {"difficulty": "Easy",   "topic": "general interest (travel, animals, food, sports)",        "q_count": 5, "q_start": 1},
    2: {"difficulty": "Medium", "topic": "social topic (education, urban life, environment, work)",  "q_count": 5, "q_start": 6},
    3: {"difficulty": "Hard",   "topic": "academic topic (science, climate, archaeology, medicine)", "q_count": 5, "q_start": 11},
}


async def get_model():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in res.json().get("models", [])]
            logger.info(f"Available models: {models}")
            for p in ["llama3.2:3b","llama3.2","llama3.2:latest","mistral","mistral:latest","llama3","llama3:latest"]:
                if p in models:
                    return p
            if models: return models[0]
            raise HTTPException(503, "No models. Run: ollama pull mistral")
    except httpx.ConnectError:
        raise HTTPException(503, "Ollama not running. Run: ollama serve")


async def ollama_call(model, prompt, tokens=2000, temp=0.6):
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": temp, "top_p": 0.95, "num_predict": tokens}
        })
        if r.status_code == 404: raise HTTPException(503, f"Model not found: {model}")
        if r.status_code >= 500: raise ValueError(f"Ollama HTTP {r.status_code}")
        return r.json().get("response", "").strip()


def _label_paragraphs(text):
    paras = re.split(r'\n\s*\n', text.strip())
    paras = [p.strip() for p in paras if len(p.strip()) > 20]
    labels = ['A','B','C','D','E']
    labeled = []
    used = set()
    for p in paras[:5]:
        m = re.match(r'^([A-E])[\s.\)\-:]+', p)
        if m and m.group(1) not in used:
            labeled.append(f"{m.group(1)}  {p[m.end():].strip()}")
            used.add(m.group(1))
        else:
            for lbl in labels:
                if lbl not in used:
                    labeled.append(f"{lbl}  {p}")
                    used.add(lbl)
                    break
    if len(labeled) < 3:
        raise ValueError(f"Too few paragraphs: {len(labeled)}")
    return '\n\n'.join(labeled)


def extract_passage(raw):
    raw = raw.strip()
    
    # Attempt to extract json object if markdown code blocks are present or extra text
    clean = re.sub(r'```json\s*|```\s*', '', raw).strip()
    match = re.search(r'\{[\s\S]*\}', clean)
    candidate = match.group(0) if match else clean

    try:
        data = json.loads(candidate)
        title = data.get("title", "")
        paragraphs = data.get("paragraphs", [])
        if not title:
            title = f"Passage: {' '.join(str(paragraphs[0]).split()[:6])}..." if paragraphs else "Passage"
        
        # Ensure paragraphs are joined by double newlines for formatting
        if isinstance(paragraphs, list):
            return title, '\n\n'.join(str(p).strip() for p in paragraphs if str(p).strip())
        elif isinstance(paragraphs, str):
            return title, paragraphs.strip()
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse passage JSON: {e}")
        pass

    raise ValueError(f"Cannot extract passage. Raw: {raw[:300]}")


def fix_json_newlines(raw):
    result = []
    in_string = False
    escape = False
    for ch in raw:
        if escape:
            result.append(ch); escape = False
        elif ch == '\\':
            result.append(ch); escape = True
        elif ch == '"':
            result.append(ch); in_string = not in_string
        elif in_string and ch == '\n': result.append('\\n')
        elif in_string and ch == '\t': result.append('\\t')
        elif in_string and ch == '\r': result.append('\\r')
        else: result.append(ch)
    return ''.join(result)


def extract_json_list(raw):
    if not raw: raise ValueError("Empty response")
    
    # Strip markdown
    clean = re.sub(r'```(?:json)?\s*', '', raw)
    clean = re.sub(r'```\s*', '', clean).strip()
    
    # Try to find array brackets
    match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', clean)
    candidate = match.group(0) if match else clean

    for fn in [
        lambda s: s,
        lambda s: fix_json_newlines(s),
        lambda s: re.sub(r',\s*([}\]])', r'\1', fix_json_newlines(s)),
        lambda s: re.sub(r',\s*([}\]])', r'\1', s),
    ]:
        try:
            r = json.loads(fn(candidate))
            return r if isinstance(r, list) else [r]
        except Exception: pass
        
    # Recovery: close truncated JSON
    fixed = fix_json_newlines(candidate)
    s = re.sub(r',\s*$', '', fixed.rstrip())
    s += ']' * max(s.count('[') - s.count(']'), 0)
    s += '}' * max(s.count('{') - s.count('}'), 0)
    try:
        r = json.loads(s)
        return r if isinstance(r, list) else [r]
    except Exception: pass
    
    raise ValueError(f"Cannot parse question list. Raw: {raw[:300]}")


def passage_prompt(num, cfg, topic_hint):
    return f"""Write an IELTS Academic Reading passage on the subject: {cfg['topic']}. {topic_hint}

Format as exactly 5 paragraphs. You MUST return ONLY a valid JSON object in the following format:
{{
  "title": "A descriptive title for the passage",
  "paragraphs": [
    "A  First paragraph text here, about sixty words of academic content on the topic.",
    "B  Second paragraph continues here.",
    "C  Third paragraph here.",
    "D  Fourth paragraph here.",
    "E  Fifth paragraph here."
  ]
}}

Rules:
- Start each paragraph text with its uppercase letter and two spaces (e.g. "A  ").
- Each paragraph 55-70 words
- Difficulty: {cfg['difficulty']}
- Academic English, factual, informative
- Output ONLY the JSON object. No markdown. No explanations. Nothing else."""


import random

def questions_prompt(num, cfg, passage_text):
    qs = cfg['q_start']
    qe = qs + cfg['q_count'] - 1
    return f"""Read this IELTS passage and write exactly {cfg['q_count']} highly challenging exam questions about it.

PASSAGE:
{passage_text}

Write exactly {cfg['q_count']} questions with IDs {qs} to {qe}.
Use exactly a mix of these types: multiple_choice, true_false_ng, fill_blank, matching_headings, short_answer.

Rules for High Difficulty:
- All answers must be factually derived from the passage above, requiring deep inference, not just word matching.
- multiple_choice: 4 specific options. The 3 wrong options MUST be highly confusing, plausible distractors that deliberately misinterpret details or use deceptive synonyms from the passage. Exactly A, B, C, or D as the correct answer.
- true_false_ng: answer is exactly "True", "False", or "Not Given". The claim should be a complex sentence testing subtle distinctions.
- fill_blank: substitute one exact word from the passage with ___, answer = that exact lowercase word. The sentence should test understanding of complex grammar or references.
- matching_headings: write exactly 3 REAL descriptive headings for paragraphs A, B, C. Add TWO extra highly plausible distractor headings that sound like they fit but are subtly wrong.
- short_answer: 1-3 exact words from the passage. answer must be lowercase. Ask about a deeply embedded sub-point.

You MUST output ONLY a valid JSON array of objects. Do NOT use markdown. Do NOT provide an explanation.
[
  {{"id": {qs}, "type": "multiple_choice", "prompt": "What does the passage imply about X?", "options": ["A. confusing distractor", "B. confusing distractor", "C. real answer", "D. confusing distractor"], "answer": "C", "acceptedAnswers": ["c"]}},
  {{"id": {qs+1}, "type": "true_false_ng", "prompt": "A complex verifiable claim.", "answer": "False", "acceptedAnswers": ["false"]}},
  {{"id": {qs+2}, "type": "fill_blank", "prompt": "Sentence with one ___ removed.", "wordLimit": "ONE WORD", "answer": "theword", "acceptedAnswers": ["theword"]}},
  {{"id": {qs+3}, "type": "matching_headings", "prompt": "Match the headings.", "paragraphs": ["A", "B", "C"], "headings": ["Heading 1", "Heading 2", "Heading 3", "Distractor 1", "Distractor 2"], "answer": {{"A": "Heading 1", "B": "Heading 2", "C": "Heading 3"}}}},
  {{"id": {qs+4}, "type": "short_answer", "prompt": "What specific subtle thing about Y?", "wordLimit": "NO MORE THAN THREE WORDS", "answer": "real answer", "acceptedAnswers": ["real answer"]}}
]
Replace ALL example values with real content from the passage. Output ONLY the unformatted JSON array."""


def validate_questions(questions, cfg):
    letters = ["A","B","C","D"]
    bad_mc = ["real answer from passage","real option","wrong answer","correct answer",
              "specific option","opt1","opt2","opt3","opt4","option here",
              "first real","second real","third real","fourth real", "confusing distractor"]
    bad_hd = ["heading1","heading2","heading3","heading4","real heading",
              "real heading describing","descriptive heading","heading about paragraph",
              "a plausible heading","extra heading","distractor","example heading",
              "heading here","heading for"]
    cleaned = []
    for q in questions:
        if not isinstance(q, dict): continue
        q.setdefault("type","multiple_choice")
        q.setdefault("prompt","")
        t = q["type"]
        if t == "multiple_choice":
            q.setdefault("options",["A. a","B. b","C. c","D. d"])
            q.setdefault("answer","A")
            
            # Find the actual text of the correct answer before shuffling
            ans_letter = str(q["answer"]).strip().upper()
            correct_text = ""
            raw_options = []
            
            for i, opt in enumerate(q["options"][:4]):
                opt_str = str(opt).strip()
                # strip A. / B. prefix for clean shuffling
                clean_opt = re.sub(r'^[A-D][.)\s]+', '', opt_str).strip()
                raw_options.append(clean_opt)
                if letters[i] == ans_letter or (len(ans_letter) > 1 and ans_letter.lower() in opt_str.lower()):
                    correct_text = clean_opt
            
            # If we couldn't find the correct text and options aren't empty, guess the first one
            if not correct_text and raw_options:
                correct_text = raw_options[0]

            random.shuffle(raw_options)

            new_options = []
            new_ans_letter = "A"
            for i, clean_opt in enumerate(raw_options):
                for phrase in bad_mc:
                    if phrase in clean_opt.lower(): raise ValueError(f"Placeholder MC: '{clean_opt}'")
                new_options.append(f"{letters[i]}. {clean_opt}")
                if clean_opt == correct_text:
                    new_ans_letter = letters[i]

            q["options"] = new_options
            q["answer"] = new_ans_letter
            q["acceptedAnswers"] = [new_ans_letter.lower()]
            
        elif t == "true_false_ng":
            a = str(q.get("answer","")).strip().lower()
            q["answer"] = "True" if a in ["true","t","yes"] else "False" if a in ["false","f","no"] else "Not Given"
            q["acceptedAnswers"] = [q["answer"].lower()]
            
        elif t in ("fill_blank","short_answer"):
            ans = str(q.get("answer","")).strip().lower()
            if not ans: raise ValueError(f"Empty answer Q{q.get('id')}")
            q["answer"] = ans
            q["acceptedAnswers"] = list({ans}|{str(a).lower().strip() for a in q.get("acceptedAnswers",[]) if str(a).strip()})
            q.setdefault("wordLimit","ONE WORD")
            
        elif t == "matching_headings":
            q.setdefault("paragraphs",["A","B","C"])
            
            raw_headings = q.get("headings",[])
            for h in raw_headings:
                for phrase in bad_hd:
                    if phrase in str(h).lower(): raise ValueError(f"Placeholder heading: '{h}'")
                    
            random.shuffle(raw_headings)
            q["headings"] = raw_headings
            q.setdefault("answer",{})
            
        cleaned.append(q)
    if len(cleaned)<3: raise ValueError(f"Too few questions: {len(cleaned)}")
    return cleaned


class GenerateRequest(BaseModel):
    passage_number: int
    topic: str = ""

class ScoreRequest(BaseModel):
    passages: list
    answers: dict


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{OLLAMA_URL}/api/tags")
            return {"models": [m["name"] for m in res.json().get("models",[])]}
    except Exception as e:
        return {"models":[],"error":str(e)}


@app.post("/api/generate-passage")
async def generate_passage(req: GenerateRequest):
    if not 1 <= req.passage_number <= 3:
        raise HTTPException(400, "Passage must be 1-3")
    model = await get_model()
    cfg = PASSAGE_CONFIGS[req.passage_number]
    topic_hint = f"Topic must be: {req.topic}." if req.topic else ""
    last_error = ""
    logger.info(f"Using model: {model}")

    for attempt in range(1, MAX_RETRIES+1):
        temp = 0.35 if attempt > 1 else 0.65
        logger.info(f"Passage {req.passage_number} attempt {attempt}/{MAX_RETRIES}")
        try:
            logger.info("Step 1: passage...")
            p_raw = await ollama_call(model, passage_prompt(req.passage_number, cfg, topic_hint), tokens=1200, temp=temp)
            logger.info(f"Passage raw (200): {p_raw[:200]}")
            title, passage_text = extract_passage(p_raw)
            if not title:
                title = f"Passage {req.passage_number}: {' '.join(passage_text.split()[:6])}..."
            logger.info(f"Passage OK: {len(passage_text)} chars")

            logger.info("Step 2: questions...")
            q_raw = await ollama_call(model, questions_prompt(req.passage_number, cfg, passage_text), tokens=3000, temp=temp)
            logger.info(f"Questions raw (300): {q_raw[:300]}")
            questions = extract_json_list(q_raw)
            questions = validate_questions(questions, cfg)

            return JSONResponse({"success":True,"passage":{"id":req.passage_number,"title":title,"topic":"","passage":passage_text,"questions":questions}})

        except HTTPException: raise
        except httpx.ConnectError: raise HTTPException(503, "Ollama not running.")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt} failed: {last_error}")

    raise HTTPException(500, f"Failed after {MAX_RETRIES} attempts: {last_error}")


@app.post("/api/score")
async def score_test(req: ScoreRequest):
    total, results = 0, []
    for passage in req.passages:
        p_correct, p_results = 0, []
        for q in passage.get("questions", []):
            user_ans = req.answers.get(str(q.get("id")), "")
            correct = False
            q_type = q.get("type", "")
            
            if q_type == "multiple_choice": 
                correct = str(user_ans).strip().upper() == str(q.get("answer", "")).strip().upper()
            elif q_type == "true_false_ng": 
                correct = str(user_ans).strip().lower() == str(q.get("answer", "")).strip().lower()
            elif q_type in ("fill_blank", "short_answer"): 
                accepted = [str(a).lower().strip() for a in q.get("acceptedAnswers", [])]
                correct = str(user_ans).strip().lower() in accepted
            elif q_type == "matching_headings":
                user_obj = user_ans if isinstance(user_ans, dict) else {}
                expected_obj = q.get("answer", {})
                correct = True
                for p in q.get("paragraphs", []):
                    u_val = str(user_obj.get(p, "")).strip().lower()
                    e_val = str(expected_obj.get(p, "")).strip().lower()
                    if u_val != e_val:
                        correct = False
                        break
                        
            if correct: 
                p_correct += 1
                total += 1
                
            p_results.append({
                "id": q.get("id"),
                "correct": correct,
                "user": user_ans,
                "expected": q.get("answer"),
                "prompt": q.get("prompt"),
                "type": q_type
            })
            
        results.append({
            "passage_id": passage.get("id", "Unknown"),
            "title": passage.get("title", f"Passage {passage.get('id', '')}"),
            "correct": p_correct,
            "total": len(passage.get("questions", [])),
            "questions": p_results
        })
        
    max_q = sum(len(p.get("questions", [])) for p in req.passages)
    pct = total / max_q if max_q else 0
    band = ("9.0" if pct >= .97 else "8.5" if pct >= .93 else "8.0" if pct >= .87 else "7.5" if pct >= .80 else "7.0" if pct >= .73 else "6.5" if pct >= .65 else "6.0" if pct >= .55 else "5.5" if pct >= .45 else "5.0" if pct >= .35 else "4.5" if pct >= .28 else "4.0")
    
    return JSONResponse({
        "total": total,
        "max": max_q,
        "band": band,
        "passages": results
    })
