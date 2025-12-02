#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import os
import time
import argparse
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# OpenAI (supports both modern and legacy clients)
import openai

# ----------------------
# CLI
# ----------------------
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--question', type=str,
    default='original',
    choices=['original', 'rephrased'],
    help="Use 'original' to ask original questions or 'rephrased' to use Two-step RaR."
)
parser.add_argument('--new_rephrase', action='store_true',
    help='Recompute and overwrite rephrased questions into the data file.'
)
parser.add_argument('--task', type=str,
    choices=[
        'birthdate_day', 'birthdate_month', 'birthdate_year',
        'birthdate_earlier', 'coin_val', 'last_letter_concatenation',
        'last_letter_concatenation4', 'sports', 'date', 'csqa', 'stereo',
        'diagnosis_partial'
    ],
    required=True,
    help='Name (prefix) of the dataset JSON in data/.'
)
parser.add_argument('--model', type=str,
    default='gpt-4',
    help='OpenAI model name (e.g., gpt-4, gpt-4o, gpt-3.5-turbo).'
)
parser.add_argument('--onestep', action='store_true',
    help='Use One-step RaR (rephrase and respond in a single prompt).'
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
# API key loading
# ----------------------
def _load_api_key():
    # Prefer local file
    try:
        with open('.openai_api_key', 'r', encoding='utf-8') as fh:
            key = fh.read().strip()
            if key:
                return key
    except FileNotFoundError:
        pass
    # Fallback env var
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError(
        "OpenAI API key not found. Put your key in '.openai_api_key' "
        "or set OPENAI_API_KEY environment variable."
    )

_API_KEY = _load_api_key()

# ----------------------
# OpenAI client (modern SDK if available; legacy fallback)
# ----------------------
def _build_chat_create():
    """
    Prefer modern SDK (openai>=1.x: OpenAI().chat.completions.create),
    else fall back to legacy openai.ChatCompletion.create.
    """
    try:
        from openai import OpenAI  # modern SDK
        client = OpenAI(api_key=_API_KEY)
        def _chat_create(**kwargs):
            return client.chat.completions.create(**kwargs)
        return _chat_create
    except Exception:
        openai.api_key = _API_KEY
        def _chat_create(**kwargs):
            return openai.ChatCompletion.create(**kwargs)
        return _chat_create

_chat_create = _build_chat_create()

# Deterministic generation defaults
GEN_KW = dict(temperature=0, top_p=1, max_tokens=512)

# ----------------------
# Backoff wrapper
# ----------------------
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return _chat_create(**kwargs)

def chatgpt_conversation(conversation_log, model_id):
    response = completion_with_backoff(
        model=model_id,
        messages=conversation_log,
        **GEN_KW
    )
    return response.choices[0].message.content.strip()

# ----------------------
# Prompt helpers
# ----------------------
TOP5_INSTRUCTIONS = (
    "Return ONLY valid JSON with this exact schema:\n"
    "{\n"
    '  "diagnoses": ["diag1", "diag2", "diag3", "diag4", "diag5"]\n'
    "}\n"
    "Make sure there are exactly 5 strings, ranked from most likely to least likely."
)

def build_top5_prompt(question_text, spec_text):
    base = f"{question_text}\n{spec_text}\n\n"
    ask = (
        "Provide your top 5 possible diagnoses ranked from most likely to least likely.\n"
        f"{TOP5_INSTRUCTIONS}"
    )
    return base + ask

def build_top5_prompt_rephrased(original_q, rephrased_q, spec_text):
    return (
        f"(original) {original_q}\n"
        f"(rephrased) {rephrased_q}\n"
        "Use your answer for the rephrased question to answer the original question.\n"
        "Provide your top 5 possible diagnoses ranked from most likely to least likely.\n"
        f"{spec_text}\n\n{TOP5_INSTRUCTIONS}"
    )

# ----------------------
# Parsing helpers
# ----------------------
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.I | re.M)

def _strip_code_fences(s: str) -> str:
    return FENCE_RE.sub("", (s or "").strip())

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def parse_top5_from_response(resp_text):
    """
    Priority:
      1) Strict JSON {"diagnoses": [...]}, after stripping code fences.
      2) Parse numbered/bulleted lines
      3) Fallback: first 5 non-empty lines
    Always returns list of length 5 (pads with "")
    """
    resp_text = _strip_code_fences(resp_text)

    # Try strict JSON
    try:
        candidate = json.loads(resp_text)
        if isinstance(candidate, dict) and isinstance(candidate.get("diagnoses"), list):
            vals = [str(x).strip() for x in candidate["diagnoses"][:5]]
            while len(vals) < 5:
                vals.append("")
            return vals
    except Exception:
        pass

    # Bullets / numbers / letters
    items = []
    for line in resp_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(?:-|\u2022|\*|\d+[\.\)]|[A-Ea-e][\.\)])\s+(.*)$", line)
        items.append((m.group(1).strip() if m else line))
        if len(items) >= 5:
            break

    if not items:
        items = [l.strip() for l in resp_text.splitlines() if l.strip()][:5]

    while len(items) < 5:
        items.append("")
    return items[:5]

def answer_in_predictions(gold_answer, preds):
    """
    Prefer normalized exact match; fallback to containment.
    """
    ga = _norm_text(gold_answer)
    if not ga:
        return False
    for p in preds:
        pn = _norm_text(p)
        if pn == ga:
            return True
    for p in preds:
        if ga and ga in _norm_text(p):
            return True
    return False

# ----------------------
# Core runners
# ----------------------
def _open_data(filename):
    with open(f'data/{filename}_{model_id}.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def _ensure_logs():
    log_directory = f'log_{model_id}'
    os.makedirs(log_directory, exist_ok=True)
    return log_directory

def get_result(filename):
    """
    Diagnosis-style runner (Top-5 JSON):
      - Prompts model for TOP-5 diagnoses in JSON
      - Logs prediction_1..prediction_5 + full_response + prompt
      - Reports Top-1 and Top-5 accuracy
    """
    data = _open_data(filename)
    print(f'data/{filename}_{model_id}.json')

    log_directory = _ensure_logs()
    top1_right, top5_right, total = 0, 0, 0

    for idx, q in tqdm(enumerate(data), total=len(data)):
        answer = q.get('answer', '')
        total += 1

        # Build prompt
        if args.question == 'rephrased':
            if not q.get('refined_question'):
                raise ValueError("Missing refined_question; run with --new_rephrase first.")
            prompt = build_top5_prompt_rephrased(q['question'], q['refined_question'], SPEC)
        else:
            prompt = build_top5_prompt(q['question'], SPEC)

        messages = [{"role": "user", "content": prompt}]
        response = chatgpt_conversation(messages, model_id)

        # Parse Top-5
        preds = parse_top5_from_response(response)

        # Accuracies
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
                    "full_response": response,
                    "prompt": prompt
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
                "full_response": response,
                "prompt": prompt
            }
            json.dump(record, flog, ensure_ascii=False)
            flog.write('\n')

        time.sleep(1)

    top1_acc = top1_right / total if total else 0.0
    top5_acc = top5_right / total if total else 0.0
    print(f"Top-1 Accuracy: {top1_acc:.4f}  |  Top-5 Accuracy: {top5_acc:.4f}")

LETTER_RE = re.compile(r"\b([A-E])\b", flags=re.I)

def get_result_multi(filename):
    """
    Multiple-choice path with full_response logging and prompt capture.
    If SPEC doesn’t already enforce letter-only, we append a line to request letter output.
    """
    data = _open_data(filename)
    log_directory = _ensure_logs()
    right, wrong = 0, 0

    for idx, q in tqdm(enumerate(data), total=len(data)):
        # Build prompt
        choices_str = ' '.join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(q['choices']))
        base = (
            f"{q['question']}\n"
            f"Choices: {choices_str}\n"
            f"{SPEC}\n"
        )
        enforce_letter = "Answer with the single letter" not in SPEC
        if enforce_letter:
            base += "Answer with the single letter (A, B, C, ...). Do not explain.\n"

        if args.question == 'rephrased':
            if not q.get('refined_question'):
                raise ValueError("Missing refined_question; run with --new_rephrase first.")
            prompt = (
                f"(original) {q['question']}\n"
                f"(rephrased) {q['refined_question']}\n"
                f"Choices: {choices_str}\n"
                "Use your answer for the rephrased question to answer the original question.\n"
                f"{SPEC}\n"
            )
            if enforce_letter:
                prompt += "Answer with the single letter (A, B, C, ...). Do not explain.\n"
        else:
            prompt = base

        messages = [{"role": "user", "content": prompt}]
        response = chatgpt_conversation(messages, model_id)

        # Parse letter first
        m = LETTER_RE.search(response)
        if m:
            chosen_idx = ord(m.group(1).upper()) - ord('A')
        else:
            # fallback: try exact choice containment
            chosen_idx = next((i for i, c in enumerate(q['choices']) if c.lower() in response.lower()), -1)

        gold_idx = next((i for i, c in enumerate(q['choices']) if q['answer'].lower() == c.lower()), -1)
        is_correct = (chosen_idx == gold_idx)

        if is_correct:
            right += 1
        else:
            wrong += 1
            with open(f'{log_directory}/{filename}_{args.question}_wrong.json', 'a', encoding='utf-8') as fwrong:
                record = {"question": q["question"], "answer": q["answer"], "response": response, "prompt": prompt}
                json.dump(record, fwrong, ensure_ascii=False)
                fwrong.write('\n')

        with open(f'{log_directory}/{filename}_{args.question}_response.json', 'a', encoding='utf-8') as flog:
            record = {"question": q["question"], "answer": q["answer"], "response": response, "prompt": prompt}
            json.dump(record, flog, ensure_ascii=False)
            flog.write('\n')

        time.sleep(1)

    acc = right / (right + wrong) if (right + wrong) else 0.0
    print("Accuracy: ", f"{acc:.4f}")

def get_result_stereo(filename):
    """
    StereoSet runner (simple tallies). If you need LM/Fair scores later,
    change the prompt to allow NONE and compute both metrics.
    """
    data = _open_data(filename)
    log_directory = _ensure_logs()

    stereo_num, anti_stereo_num, unrelated_num, undetermined = 0, 0, 0, 0

    for idx, q in tqdm(enumerate(data), total=len(data)):
        choices = q['choices']

        # Build lines safely (no backslashes inside f-string expressions)
        if args.question == 'rephrased':
            if not q.get('refined_question'):
                raise ValueError("Missing refined_question; run with --new_rephrase first.")
            orig_line = f"(original) {q['question']}\n"
            reph_line = f"(rephrased) {q['refined_question']}\n"
        else:
            orig_line = f"{q['question']}\n"
            reph_line = ""

        prompt = (
            orig_line
            + reph_line
            + f"Choices: A. {choices[0]} B. {choices[1]} C. {choices[2]}\n"
            + f"{SPEC}\n"
            + "Answer with a single letter (A, B, or C). Do not explain.\n"
        )

        messages = [{"role": "user", "content": prompt}]
        response = chatgpt_conversation(messages, model_id)

        # Parse letter
        m = LETTER_RE.search(response)
        chosen = m.group(1).upper() if m else None

        stereo, anti_stereo, unrelated = q['stereo'], q['anti_stereo'], q['unrelated']

        # Map letter to text
        letter_to_choice = {'A': choices[0], 'B': choices[1], 'C': choices[2]}
        picked = letter_to_choice.get(chosen, "")

        if picked and _norm_text(stereo) == _norm_text(picked):
            stereo_num += 1
            out_path = f'{log_directory}/{filename}_{args.question}_stereo.json'
        elif picked and _norm_text(anti_stereo) == _norm_text(picked):
            anti_stereo_num += 1
            out_path = f'{log_directory}/{filename}_{args.question}_anti_stereo.json'
        elif picked and _norm_text(unrelated) == _norm_text(picked):
            unrelated_num += 1
            out_path = None  # optional log
        else:
            undetermined += 1
            out_path = f'{log_directory}/{filename}_{args.question}_undetermined.json'

        if out_path:
            with open(out_path, 'a', encoding='utf-8') as fout:
                json.dump({"question": q["question"], "picked": picked, "response": response, "prompt": prompt}, fout, ensure_ascii=False)
                fout.write('\n')

        with open(f'{log_directory}/{filename}_{args.question}_response.json', 'a', encoding='utf-8') as flog:
            json.dump({"question": q["question"], "response": response, "prompt": prompt}, flog, ensure_ascii=False)
            flog.write('\n')

        time.sleep(1)

    print("stereo: ", stereo_num)
    print("anti_stereo: ", anti_stereo_num)
    print("unrelated: ", unrelated_num)
    print("undetermined: ", undetermined)

def refine_question(filename):
    """
    Compute (or recompute) the rephrased question for each item and save under 'refined_question'.
    Also removes GPT-3.5's occasional 'rephrase...' echo lines.
    """
    data = _open_data(filename)
    if 'refined_question' in data[0] and data[0]['refined_question']:
        print("Overwriting the refined questions.")

    for idx, q in tqdm(enumerate(data), total=len(data)):
        prompt = (
            f"\"{q['question']}\"\n"
            "Given the above question, rephrase and expand it to help you do better answering. "
            "Maintain all information in the original question."
        )
        messages = [{"role": "user", "content": prompt}]
        response = chatgpt_conversation(messages, model_id)
        # Trim quotes and remove 'rephrase' echo lines
        resp = response.strip()
        if resp and resp[0] == '"' and resp[-1] == '"':
            resp = resp[1:-1]
        resp = "\n".join(line for line in resp.splitlines() if "rephrase" not in line.lower())
        q['refined_question'] = resp.strip()
        time.sleep(1)

    with open(f'data/{filename}_{model_id}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def get_result_onestep(filename):
    """
    One-step RaR (rephrase + respond in one prompt) for Top-5 diagnoses JSON.
    """
    data = _open_data(filename)
    log_directory = _ensure_logs()
    top1_right, top5_right, total = 0, 0, 0

    for idx, q in tqdm(enumerate(data), total=len(data)):
        total += 1
        prompt = (
            f"\"{q['question']}\"\n"
            "Rephrase and expand the question, and respond.\n"
            "Provide your top 5 possible diagnoses ranked from most likely to least likely.\n"
            f"{SPEC}\n\n{TOP5_INSTRUCTIONS}"
        )

        messages = [{"role": "user", "content": prompt}]
        response = chatgpt_conversation(messages, model_id)
        preds = parse_top5_from_response(response)
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
                    "full_response": response,
                    "prompt": prompt
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
                "full_response": response,
                "prompt": prompt
            }
            json.dump(record, flog, ensure_ascii=False)
            flog.write('\n')

        time.sleep(1)

    top1_acc = top1_right / total if total else 0.0
    top5_acc = top5_right / total if total else 0.0
    print(f"Top-1 Accuracy (onestep): {top1_acc:.4f}  |  Top-5 Accuracy (onestep): {top5_acc:.4f}")

# ----------------------
# Entrypoint
# ----------------------
def main():
    if args.onestep:
        # One-step ignores args.question value; it internally rephrases+answers
        get_result_onestep(args.task)
    else:
        if args.new_rephrase:
            refine_question(args.task)

        if 'csqa' in args.task:
            get_result_multi(args.task)
        elif args.task == 'stereo':
            get_result_stereo(args.task)
        else:
            # diagnosis & similar → Top-5 JSON path
            get_result(args.task)

if __name__ == "__main__":
    main()
