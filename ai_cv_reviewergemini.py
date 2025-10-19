import os
import json
import textwrap
import time
from pathlib import Path

import requests



GEMINI_API_KEY = "YOUR_API_KEY" 
if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("AIzaSy_"):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    raise RuntimeError("Lūdzu iestatiet GEMINI_API_KEY vides mainīgo ar jūsu API atslēgu.")


API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.2 

BASE_DIR = Path.cwd()
SAMPLE_DIR = BASE_DIR / "sample_inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
PROMPT_DIR = BASE_DIR / "prompts"
OUTPUT_DIR.mkdir(exist_ok=True)
PROMPT_DIR.mkdir(exist_ok=True)

JD_PATH = SAMPLE_DIR / "jd.txt" 
CV_PATHS = [SAMPLE_DIR / f"cv{i}.txt" for i in (1, 2, 3)] 


REQUIRED_KEYS = ["match_score", "summary", "strengths", "missing_requirements", "verdict"]


def read_text(path: Path) -> str:
    """Nolasa teksta failu."""
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def build_prompt(jd_text: str, cv_text: str, candidate_label: str = "Candidate") -> str:
    """Sagatavo oriģinālu Gemini promptu ar stingrām instrukcijām."""
    instruct = textwrap.dedent("""
    You are a hiring-focused assistant. Compare a Job Description (JD) with a candidate CV and
    produce a single JSON object (no extra text, no commentary) that assesses fit.

    Requirements for the JSON output (HR focus):
    - Respond ONLY with a single JSON object encoded in UTF-8 and valid JSON.
    - Do not include any explanatory text, markdown, or trailing commas.
    - IMPORTANT: The JSON must strictly adhere to the provided schema below.

    JSON schema (exact keys):
    {{
      "match_score": integer between 0 and 100,
      "summary": string (short, 1-3 sentences describing overall fit),
      "strengths": array of strings (key skills/experience from CV that match JD),
      "missing_requirements": array of strings (important JD requirements not visible in CV),
      "verdict": one of ["strong match", "possible match", "not a match"]
    }}

    Rules to score and produce fields:
    1) Evaluate technical skills, years of experience, domain knowledge, tools, certifications,
       and soft skills required in the JD. Give more weight to mandatory requirements.
    2) match_score: produce an integer 0-100. Use 0-39 = not a match, 40-69 = possible match, 70-100 = strong match.
    3) strengths: up to 6 bullet items pulled verbatim or paraphrased from the CV.
    4) missing_requirements: up to 6 items from the JD not found in the CV; prefer exact phrasing.
    5) verdict: map from match_score using the ranges above.

    Now compare the Job Description and the CV below.

    JOB DESCRIPTION:
    ------------------------------------------------------------
    {jd}
    ------------------------------------------------------------

    {candidate_label} CV:
    ------------------------------------------------------------
    {cv}
    ------------------------------------------------------------

    Produce the JSON now.
    """).format(jd=jd_text, cv=cv_text, candidate_label=candidate_label)

    return instruct


def save_prompt_md(text: str, path: Path):
    """Saglabā prompt tekstu .md failā."""
    path.write_text(text, encoding="utf-8")


def call_gemini(prompt: str) -> dict:
    """Izsauc Gemini Flash 2.5 REST API, izmantojot stingru JSON izvades shēmu."""
    url_with_key = f"{API_URL}?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json",
    }

   
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": TEMPERATURE,
            
            "responseMimeType": "application/json",
            
            "responseSchema": {
                "type": "object",
                "properties": {
                    "match_score": {"type": "integer"},
                    "summary": {"type": "string"},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "missing_requirements": {"type": "array", "items": {"type": "string"}},
                    "verdict": {"type": "string", "enum": ["strong match", "possible match", "not a match"]}
                },
                "required": REQUIRED_KEYS
            }
        }
    }

    resp = requests.post(url_with_key, headers=headers, json=payload, timeout=60)
    
    try:
        resp.raise_for_status()
    except Exception as e:
        error_msg = f"API request failed: {e}"
        try:
            error_data = resp.json()
            if 'error' in error_data:
                error_msg += f" - API Error: {error_data['error']['message']}"
        except json.JSONDecodeError:
            error_msg += f" - response: {resp.text}"
        raise RuntimeError(error_msg)

    data = resp.json()

    
    try:
        json_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
    except (KeyError, IndexError):
        raise RuntimeError("Nevarēja atrast teksta atbildi Gemini API atbildē. Pilns API atbildes saturs:\n" + json.dumps(data, ensure_ascii=False, indent=2))


    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Nevarēja parsēt JSON atbildi: {e}. Teksts: {json_text}")
        
    return parsed


def validate_hr_json(obj: dict) -> bool:
    """Validē saņemto JSON atbilstoši uzdevuma prasībām."""
    if not isinstance(obj, dict):
        return False
    for k in REQUIRED_KEYS:
        if k not in obj:
            return False

    if not isinstance(obj.get("match_score"), int) or not (0 <= obj.get("match_score") <= 100):
        return False
    if not isinstance(obj.get("summary"), str):
        return False
    if not isinstance(obj.get("strengths"), list):
        return False
    if not isinstance(obj.get("missing_requirements"), list):
        return False
    if obj.get("verdict") not in ["strong match", "possible match", "not a match"]:
        return False
    return True


def generate_report_md(json_obj: dict, candidate_label: str, out_path: Path):
    """Ģenerē īsu pārskatu (Markdown) no JSON atbildes."""
    md = []
    md.append(f"# CV Review — {candidate_label} (Gemini Flash 2.5)\n")
    md.append(f"**Match score:** {json_obj.get('match_score')} / 100  ")
    md.append(f"**Verdict:** {json_obj.get('verdict')}\n")
    md.append("## Summary\n")
    md.append(json_obj.get('summary') + "\n")
    md.append("## Strengths (from CV) / Galvenās prasmes un pieredze\n")
    for s in json_obj.get('strengths', []):
        md.append(f"- {s}")
    md.append("\n## Missing / Not Evident Requirements (from JD) / Trūkstošās prasības\n")
    for m in json_obj.get('missing_requirements', []):
        md.append(f"- {m}")

    out_text = "\n".join(md)
    out_path.write_text(out_text, encoding='utf-8')


def main():
    """Galvenā izpildes loģika, atkārtojot 2.-5. soli visiem CV[cite: 12]."""
    try:
        jd_text = read_text(JD_PATH) 
    except FileNotFoundError:
        print(f"Kļūda: Darba apraksta fails '{JD_PATH}' (jd.txt) nav atrasts. Lūdzu pārliecinieties, ka 'sample_inputs/jd.txt' pastāv.")
        return

    for i, cv_path in enumerate(CV_PATHS, start=1):
        try:
            cv_text = read_text(cv_path) 
        except FileNotFoundError:
            print(f"Kļūda: CV fails '{cv_path}' nav atrasts. Izlaižam šo kandidātu.")
            continue
            
        label = f"Candidate {i}"

       
        prompt_text = build_prompt(jd_text, cv_text, candidate_label=label)
        prompt_file = PROMPT_DIR / f"prompt_cv{i}.md"
        save_prompt_md(prompt_text, prompt_file)
        print(f"Saved prompt for {label} -> {prompt_file}")

     
        print(f"Calling model {MODEL_NAME} for {label}... (this may take a few seconds)")
        try:
            model_json = call_gemini(prompt_text)
        except RuntimeError as e:
            print(f"Error while calling model for {label}: {e}")
            continue

    
        if not validate_hr_json(model_json):
            print(f"Warning: model JSON for {label} failed validation. Saving raw response for inspection.")
            (OUTPUT_DIR / f"cv{i}_raw.json").write_text(json.dumps(model_json, ensure_ascii=False, indent=2), encoding='utf-8')
            continue

        out_json_path = OUTPUT_DIR / f"cv{i}.json"
        out_json_path.write_text(json.dumps(model_json, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Saved JSON result for {label} -> {out_json_path}")

   
        report_path = OUTPUT_DIR / f"cv{i}_report.md"
        generate_report_md(model_json, label, report_path)
        print(f"Saved report for {label} -> {report_path}")

       
        time.sleep(1)


if __name__ == '__main__':

    main()

