# main_top5.py

import json
import random
import os
import time
import argparse
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# === OpenAI (compatible with legacy ChatCompletion for minimal change)
import openai

# ----------------------
# CLI
# ----------------------
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--question', type=str,
    default='original',
    choices=['original', 'rephrased'],
    help="Specify 'original' to process original questions or 'rephrased' to process rephrased questions."
)
parser.add_argument('--new_rephrase', action='store_true',
    help='Flag to refine the questions again.'
)
parser.add_argument('--task', type=str,
    choices=[
        'birthdate_day', 'birthdate_month', 'birthdate_year',
        'birthdate_earlier', 'coin_val', 'last_letter_concatenation',
        'last_letter_concatenation4', 'sports', 'date', 'csqa', 'stereo',
        'diagnosis_partial'
    ],
    help='Specify the task file name for processing.'
)
parser.add_argument('--model', type=str,
    default='gpt-4',
    help='Specify the model name of the OpenAI API to use.'
)
parser.add_argument('--onestep', action='store_true',
    help='Flag to use onestep RaR.'
)
args = parser.parse_args()
model_id = args.model

# ----------------------
# Config / Spec loading
# ----------------------
with open('config.json', 'r', encoding='utf-8') as config_file:
    spec_config = json.load(config_file)

SPEC = spec_config.get(args.task, "")

# ----------------------
# API key loading (no env export required)
# ----------------------
def _load_api_key():
    # 1) Prefer a local file (DO NOT commit this file)
    try:
        with open('.openai_api_key', 'r', encoding='utf-8') as fh:
            key = fh.read().strip()
            if key:
                return key
    except FileNotFoundError:
        pass
    # 2) Fallback: environment variable
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError(
        "OpenAI API key not found. Put your key in a file named '.openai_api_key' "
        "or set the OPENAI_API_KEY environment variable."
    )

_API_KEY = _load_api_key()

# ----------------------
# OpenAI client (modern SDK if available; legacy fallback)
# ----------------------
def _build_chat_create():
    """
    Prefer modern SDK if available (openai>=1.x: OpenAI().chat.completions.create),
    else fall back to legacy openai.ChatCompletion.create.
    """
    try:
        from openai import OpenAI  # modern SDK
        client = OpenAI(api_key=_API_KEY)
        def _chat_create(**kwargs):
            return client.chat.completions.create(**kwargs)
        return _chat_create
    except Exception:
        # legacy SDK path
        openai.api_key = _API_KEY
        def _chat_create(**kwargs):
            return openai.ChatCompletion.create(**kwargs)
        return _chat_create

_chat_create = _build_chat_create()

# ----------------------
# Label normalization (synonym-safe, punctuation-insensitive)
# ----------------------
_PUNCT_RE = re.compile(r"[^\w\s/+-]")  # keep word chars, space, / + -
_SPACE_RE = re.compile(r"\s+")

# Optional: load a custom synonym map if you keep one locally (do not commit).
# Format: {"mi": "myocardial infarction", "uti": "urinary tract infection", ...}
def _load_syn_map():
    path = "label_norm_map.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)
            # normalize keys/vals in the map itself
            return { _normalize_text(k): v.strip() for k, v in m.items() if isinstance(k, str) and isinstance(v, str) }
        except Exception:
            return {}
    return {}

_SYN_MAP = _load_syn_map()

def _normalize_text(s: str) -> str:
    """
    Lowercase, strip outer space, collapse inner spaces, remove stray punctuation.
    Keeps '/', '+', '-' to preserve common clinical labels like 'A+ hemolytic strep' or 'COVID-19'.
    """
    s = (s or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s

def normalize_label(s: str) -> str:
    """
    Heuristic normalization + synonym collapse via _SYN_MAP.
    """
    raw = (s or "").strip()
    norm = _normalize_text(raw)
    # apply user-provided mapping if present
    if norm in _SYN_MAP:
        return _SYN_MAP[norm]
    # light built-ins (kept very small to avoid surprises)
    if norm in {"mi", "myocardial infarct"}:
        return "myocardial infarction"
    if norm in {"uti", "urinary tract inf"}:
        return "urinary tract infection"
    if norm in {"pna"}:
        return "pneumonia"
    return raw.strip()  # keep original surface for logging, but comparisons use normalized forms

# ----------------------
# Backoff wrapper
# ----------------------
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return _chat_create(**kwargs)

def _chat_with_defaults(messages, model, force_json=False, **kw):
    # deterministic defaults
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=0,
        **kw
    )

    # JSON mode is supported only on newer models (e.g., gpt-4o family)
    supports_json_mode = any(tag in model.lower() for tag in [
        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "o4", "o3"
    ])

    if force_json and supports_json_mode:
        # Use JSON mode only when the model supports it
        kwargs["response_format"] = {"type": "json_object"}

    return completion_with_backoff(**kwargs)


def chatgpt_conversation(conversation_log, model_id):
    """
    Keep the same call sites. We detect whether the prompt demands JSON
    (diagnosis runner uses TOP5 JSON instruction) and enable JSON mode only then.
    """
    last = conversation_log[-1]["content"] if conversation_log else ""
    force_json = ('"diagnoses"' in last and 'Return ONLY valid JSON' in last)
    resp = _chat_with_defaults(conversation_log, model=model_id, force_json=force_json, max_tokens=500)
    return resp.choices[0].message.content.strip()

# ----------------------
# Prompt helpers
# ----------------------
TOP5_INSTRUCTIONS = (
    # Light “self-consistency” wording (single call) + strict JSON
    "Privately consider multiple plausible differentials, reconcile conflicts, and then output only the final result.\n"
    "Return ONLY valid JSON with this exact schema:\n"
    "{\n"
    '  "diagnoses": ["diag1", "diag2", "diag3", "diag4", "diag5"]\n'
    "}\n"
    "Rules for the list:\n"
    "- Exactly 5 strings.\n"
    "- Rank from most likely to least likely.\n"
    "- Use the EXACT canonical label names from the allowed label set specified in SPEC (no synonyms, no abbreviations).\n"
    "- Do not include probabilities, explanations, or extra fields."
)

def build_top5_prompt(question_text, spec_text):
    # Strengthen constraint: SPEC carries the label set and any rubric
    base = (
        f"{question_text}\n\n"
        f"{spec_text}\n\n"
        "STRICT CONSTRAINTS:\n"
        "- Choose ONLY from the allowed diagnosis label set defined in SPEC (use exact canonical strings).\n"
        "- Do NOT invent new labels or use synonyms/variants.\n"
        "- If two labels are similar, pick the single best canonical label from the set.\n\n"
    )
    ask = (
        "Provide your top 5 possible diagnoses ranked from most likely to least likely.\n"
        f"{TOP5_INSTRUCTIONS}"
    )
    return base + ask

def build_top5_prompt_rephrased(original_q, rephrased_q, spec_text, for_gpt4=True):
    label_rephrased = "rephrased" if for_gpt4 else "revised"
    return (
        f"(original) {original_q}\n"
        f"({label_rephrased}) {rephrased_q}\n\n"
        f"{spec_text}\n\n"
        "STRICT CONSTRAINTS:\n"
        "- Choose ONLY from the allowed diagnosis label set defined in SPEC (use exact canonical strings).\n"
        "- Do NOT invent new labels or use synonyms/variants.\n"
        "- If two labels are similar, pick the single best canonical label from the set.\n\n"
        "Use your answer for the rephrased question to answer the original question.\n"
        "Provide your top 5 possible diagnoses ranked from most likely to least likely.\n"
        f"{TOP5_INSTRUCTIONS}"
    )

# ----------------------
# Parsing helpers
# ----------------------
def parse_top5_from_response(resp_text):
    """
    Priority:
      1) Parse as strict JSON {"diagnoses": [...]}
      2) Parse numbered / bulleted lines (e.g., '1. X', '2) Y', '- Z', 'A) W')
      3) Fallback: take first 5 non-empty lines
    Also: deduplicate while preserving order; pad/truncate to 5.
    """
    # Try strict JSON
    try:
        candidate = json.loads(resp_text)
        if isinstance(candidate, dict) and isinstance(candidate.get("diagnoses"), list):
            vals = [str(x).strip() for x in candidate["diagnoses"][:5]]
            # dedup while preserving order
            seen = set()
            clean = []
            for v in vals:
                key = _normalize_text(v)
                if key and key not in seen:
                    seen.add(key)
                    clean.append(v)
            while len(clean) < 5:
                clean.append("")
            return clean[:5]
    except Exception:
        pass

    # Bullets / numbers / letters
    items = []
    for line in resp_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(?:-|\u2022|\*|\d+[\.\)]|[A-Ea-e][\.\)])\s+(.*)$", line)
        cand = (m.group(1).strip() if m else line)
        if cand:
            items.append(cand)
        if len(items) >= 5:
            break

    if not items:
        items = [l.strip() for l in resp_text.splitlines() if l.strip()][:5]

    # dedup/order
    seen = set()
    clean = []
    for v in items:
        key = _normalize_text(v)
        if key and key not in seen:
            seen.add(key)
            clean.append(v)
        if len(clean) == 5:
            break

    while len(clean) < 5:
        clean.append("")
    return clean[:5]

def answer_in_predictions(gold_answer, preds):
    """Case-insensitive, punctuation-insensitive containment/equality check with normalization & synonyms."""
    ga_norm = _normalize_text(gold_answer or "")
    if not ga_norm:
        return False
    for p in preds:
        # compare normalized strings (with synonym collapse for candidate)
        p_norm = _normalize_text(normalize_label(p))
        if ga_norm == p_norm:
            return True
        # also allow strict substring match after normalization (handles 'dx: X' formats)
        if ga_norm and p_norm and (ga_norm in p_norm or p_norm in ga_norm):
            return True
    return False

# ----------------------
# Core runners
# ----------------------
def get_result(filename):
    """
    Diagnosis-style runner:
      - Asks for TOP-5 diagnoses in JSON (JSON mode enforced)
      - Saves prediction_1..prediction_5 + full_response
      - Reports Top-1 and Top-5 accuracy (with normalization)
    """
    with open(f'data/{filename}_{model_id}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'data/{filename}_{model_id}.json')

    log_directory = f'log_{model_id}'
    os.makedirs(log_directory, exist_ok=True)

    top1_right, top5_right, total = 0, 0, 0

    for idx, q in tqdm(enumerate(data), total=len(data)):
        answer = q.get('answer', '')
        total += 1

        # Build prompt
        if args.question == 'rephrased':
            assert 'refined_question' in q and q['refined_question'], "Missing refined_question"
            for_gpt4 = 'gpt-4' in args.model
            prompt = build_top5_prompt_rephrased(q['question'], q['refined_question'], SPEC, for_gpt4=for_gpt4)
        else:
            prompt = build_top5_prompt(q['question'], SPEC)

        messages = [{"role": "user", "content": prompt}]
        response = chatgpt_conversation(messages, model_id)

        # Parse top-5 + normalize
        raw_preds = parse_top5_from_response(response)
        preds = [normalize_label(x) for x in raw_preds]

        # Accuracies (normalized)
        top1_hit = answer_in_predictions(answer, preds[:1])
        top5_hit = answer_in_predictions(answer, preds)

        if top1_hit:
            top1_right += 1
            top5_right += 1
        elif top5_hit:
            top5_right += 1

        # Wrong log (Top-5 miss)
        if not top5_hit:
            with open(f'{log_directory}/{filename}_{args.question}_wrong.json', 'a', encoding='utf-8') as fwrong:
                record = {
                    "question": q["question"],
                    "answer": answer,
                    "prediction_1": preds[0],
                    "prediction_2": preds[1],
                    "prediction_3": preds[2],
                    "prediction_4": preds[3],
                    "prediction_5": preds[4],
                    "full_response": response
                }
                json.dump(record, fwrong, ensure_ascii=False)
                fwrong.write('\n')

        # Document all responses
        with open(f'{log_directory}/{filename}_{args.question}_response.json', 'a', encoding='utf-8') as flog:
            record = {
                "question": q["question"],
                "answer": answer,
                "prediction_1": preds[0],
                "prediction_2": preds[1],
                "prediction_3": preds[2],
                "prediction_4": preds[3],
                "prediction_5": preds[4],
                "full_response": response
            }
            json.dump(record, flog, ensure_ascii=False)
            flog.write('\n')

        time.sleep(0.5)  # slightly faster but polite

    top1_acc = top1_right / total if total else 0.0
    top5_acc = top5_right / total if total else 0.0
    print(f"Top-1 Accuracy: {top1_acc:.4f}  |  Top-5 Accuracy: {top5_acc:.4f}")

def get_result_multi(filename):
    """
    Multiple-choice path unchanged, but logs full_response.
    """
    with open(f'data/{filename}_{model_id}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    right, wrong = 0, 0
    log_directory = f'log_{model_id}'
    os.makedirs(log_directory, exist_ok=True)

    for idx, q in tqdm(enumerate(data), total=len(data)):
        answer = q['answer']

        if args.question == 'rephrased':
            assert 'refined_question' in q and q['refined_question']
            messages = [{
                "role": "user",
                "content": (
                    f"(original) {q['question']}\n"
                    f"(rephrased) {q['refined_question']}\n"
                    "Choices: " + ' '.join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(q['choices'])) + "\n"
                    "Use your answer for the rephrased question to answer the original question.\n" + SPEC
                )
            }]
        else:
            messages = [{
                "role": "user",
                "content": (
                    f"{q['question']}\n"
                    "Choices: " + ' '.join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(q['choices'])) + "\n" + SPEC
                )
            }]

        response = chatgpt_conversation(messages, model_id)

        incorrect_choices = [c for c in q['choices'] if answer.lower() not in c.lower()]
        if answer.lower() in response.lower() and all([c.lower() not in response.lower() for c in incorrect_choices]):
            right += 1
        else:
            wrong += 1
            with open(f'{log_directory}/{filename}_{args.question}_wrong.json', 'a', encoding='utf-8') as fwrong:
                record = {"question": q["question"], "answer": q["answer"], "response": response}
                json.dump(record, fwrong, ensure_ascii=False)
                fwrong.write('\n')

        with open(f'{log_directory}/{filename}_{args.question}_response.json', 'a', encoding='utf-8') as flog:
            record = {"question": q["question"], "answer": q["answer"], "response": response}
            json.dump(record, flog, ensure_ascii=False)
            flog.write('\n')

        time.sleep(0.5)

    print("Accuracy: ", right / (right + wrong) if (right + wrong) else 0.0)

def get_result_stereo(filename):
    with open(f'data/{filename}_{model_id}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    log_directory = f'log_{model_id}'
    os.makedirs(log_directory, exist_ok=True)

    stereo_num, anti_stereo_num, unrelated_num, undetermined = 0, 0, 0, 0

    for idx, q in tqdm(enumerate(data), total=len(data)):
        stereo = q['stereo']
        anti_stereo = q['anti_stereo']
        unrelated = q['unrelated']

        if args.question == 'rephrased':
            assert 'refined_question' in q and q['refined_question']
            messages = [{
                "role": "user",
                "content": "(original)" + q['question'] + "\n"
                           + "(revised)" + q['refined_question'] + "\n"
                           + f"Choices: A. {q['choices'][0]} B. {q['choices'][1]} C. {q['choices'][2]}\n"
                           + "Use your answer in the revised question to answer the original question.\n"
                           + SPEC
            }]
        elif args.question == 'zero-CoT':
            messages = [{
                "role": "user",
                "content": q['question'] + f"Choices: A. {q['choices'][0]} B. {q['choices'][1]} C. {q['choices'][2]}\n"
                           + "Let's think step by step."
            }]
        else:
            messages = [{
                "role": "user",
                "content": q['question'] + "\n"
                           + f"Choices: A. {q['choices'][0]} B. {q['choices'][1]} C. {q['choices'][2]}\n"
                           + SPEC
            }]
        response = chatgpt_conversation(messages, model_id)

        if stereo.lower() in response.lower() and anti_stereo.lower() not in response.lower() and unrelated.lower() not in response.lower():
            stereo_num += 1
            with open(f'{log_directory}/{filename}_{args.question}_stereo.json', 'a', encoding='utf-8') as f1:
                json.dump({"question": q["question"], "answer": q["anti_stereo"], "response": response}, f1, ensure_ascii=False)
                f1.write('\n')
        elif anti_stereo.lower() in response.lower() and stereo.lower() not in response.lower() and unrelated.lower() not in response.lower():
            anti_stereo_num += 1
            with open(f'{log_directory}/{filename}_{args.question}_anti_stereo.json', 'a', encoding='utf-8') as f2:
                json.dump({"question": q["question"], "answer": q["anti_stereo"], "response": response}, f2, ensure_ascii=False)
                f2.write('\n')
        elif unrelated.lower() in response.lower() and stereo.lower() not in response.lower() and anti_stereo.lower() not in response.lower():
            unrelated_num += 1
        else:
            undetermined += 1
            with open(f'{log_directory}/{filename}_{args.question}_undetermined.json', 'a', encoding='utf-8') as f3:
                json.dump({"question": q["question"], "answer": q["anti_stereo"], "response": response}, f3, ensure_ascii=False)
                f3.write('\n')

        with open(f'{log_directory}/{filename}_{args.question}_response.json', 'a', encoding='utf-8') as flog:
            json.dump({"question": q["question"], "answer": q["anti_stereo"], "response": response}, flog, ensure_ascii=False)
            flog.write('\n')

        time.sleep(0.5)

    print("stereo: ", stereo_num)
    print("anti_stereo: ", anti_stereo_num)
    print("unrelated: ", unrelated_num)
    print("undetermined: ", undetermined)

def refine_question(filename):
    with open(f'data/{filename}_{model_id}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'refined_question' in data[0] and data[0]['refined_question']:
        print("Overwriting the refined questions.")

    REFINE_PROMPT = """You are a clinical question REWRITER for internal reasoning only.
Rewrite the ORIGINAL into 2–4 concise sentences while preserving EVERY explicit fact and negation.
Rules (strict):
- Do NOT add, infer, or guess any new details (no new symptoms, numbers, times, meds, PMH).
- Keep medical terms AS WRITTEN (no synonyms; preserve abbreviations).
- Preserve negations exactly (“denies fever” stays explicitly negative).
- No diagnoses, no advice, no questions, no bullets.
- Output ONLY the rewritten synopsis.

ORIGINAL:
"""

    for idx, q in tqdm(enumerate(data), total=len(data)):
        messages = [{
            "role": "user",
            "content": REFINE_PROMPT + f"\"\"\"{q['question']}\"\"\""
        }]
        resp = _chat_with_defaults(messages, model=model_id, force_json=False, max_tokens=300)
        text = resp.choices[0].message.content.strip()
        if text and text[0] == '"' and text[-1] == '"':
            text = text[1:-1]
        data[idx]['refined_question'] = text
        time.sleep(0.2)

    with open(f'data/{filename}_{model_id}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def get_result_onestep(filename):
    """
    Onestep RaR path but still requests Top-5 diagnoses JSON and reports Top-1/Top-5.
    """
    with open(f'data/{filename}_{model_id}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    log_directory = f'log_{model_id}'
    os.makedirs(log_directory, exist_ok=True)

    top1_right, top5_right, total = 0, 0, 0

    for idx, q in tqdm(enumerate(data), total=len(data)):
        total += 1
        if "gpt-3.5" in model_id:
            messages = [{
                "role": "user",
                "content": f'"{q["question"]}"\nReword and elaborate on the inquiry, then provide the answer as Top-5 diagnoses.\n' + SPEC + "\n\n" + TOP5_INSTRUCTIONS
            }]
        else:
            messages = [{
                "role": "user",
                "content": f'"{q["question"]}"\nRephrase and expand the question, then provide the answer as Top-5 diagnoses.\n' + SPEC + "\n\n" + TOP5_INSTRUCTIONS
            }]

        response = chatgpt_conversation(messages, model_id)
        raw_preds = parse_top5_from_response(response)
        preds = [normalize_label(x) for x in raw_preds]
        answer = q.get('answer', '')

        top1_hit = answer_in_predictions(answer, preds[:1])
        top5_hit = answer_in_predictions(answer, preds)

        if top1_hit:
            top1_right += 1
            top5_right += 1
        elif top5_hit:
            top5_right += 1
        else:
            with open(f'{log_directory}/{filename}_{args.question}_wrong.json', 'a', encoding='utf-8') as fwrong:
                record = {
                    "question": q["question"],
                    "answer": answer,
                    "prediction_1": preds[0],
                    "prediction_2": preds[1],
                    "prediction_3": preds[2],
                    "prediction_4": preds[3],
                    "prediction_5": preds[4],
                    "full_response": response
                }
                json.dump(record, fwrong, ensure_ascii=False)
                fwrong.write('\n')

        with open(f'{log_directory}/{filename}_{args.question}_combine_response.json', 'a', encoding='utf-8') as flog:
            record = {
                "question": q["question"],
                "answer": answer,
                "prediction_1": preds[0],
                "prediction_2": preds[1],
                "prediction_3": preds[2],
                "prediction_4": preds[3],
                "prediction_5": preds[4],
                "full_response": response
            }
            json.dump(record, flog, ensure_ascii=False)
            flog.write('\n')

        time.sleep(0.2)

    top1_acc = top1_right / total if total else 0.0
    top5_acc = top5_right / total if total else 0.0
    print(f"Top-1 Accuracy (onestep): {top1_acc:.4f}  |  Top-5 Accuracy (onestep): {top5_acc:.4f}")

# ----------------------
# Entrypoint
# ----------------------
def main():
    if args.onestep:
        args.question = 'rephrased'  # keep your original behavior
        get_result_onestep(args.task)
    else:
        if args.new_rephrase:
            refine_question(args.task)

        if 'csqa' in args.task:
            get_result_multi(args.task)
        elif args.task == 'stereo':
            get_result_stereo(args.task)
        else:
            # diagnosis paths → top-5
            get_result(args.task)

if __name__ == "__main__":
    main()
