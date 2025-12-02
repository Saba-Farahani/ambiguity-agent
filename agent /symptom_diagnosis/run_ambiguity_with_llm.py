# examples/ambiguity/run_ambiguity_with_llm.py

import os
import json
import ast
import math
from typing import List, Dict, Any

import pandas as pd
from openai import OpenAI

from openCHA.tasks.ambiguity.diagnostic_ambiguity_task import (
    DiagnosticAmbiguityTask,
    _normalize_symptom_text,  # use the same normalization as the KG agent
)

# Requires OPENAI_API_KEY in the environment.
client = OpenAI()


# ---------- helpers for CSV ----------
def safe_list_parse(x):
    """Parse a stringified Python list safely (or return [])."""
    if not isinstance(x, str):
        return []
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list):
            return [str(t).strip() for t in v]
    except Exception:
        pass
    return []


def join_lower(lst: List[Any]) -> str:
    """Join a list into a single lowercase string, separated by |."""
    return " | ".join([str(s).lower() for s in lst])


# ---------- KG triage loop ----------
def run_kg_ambiguity_for_patient(
    agent: DiagnosticAmbiguityTask,
    current_symptoms: List[str],
    full_symptoms: List[str],
    max_turns: int = 20,
) -> Dict[str, Any]:
    """
    1) triage_start with CURRENT_SYMPTOMS
    2) loop over next_questions, auto-answer using FULL_SYMPTOMS
    3) return final KG agent state (JSON dict)

    This simulates a “perfect” patient by answering questions
    using FULL_SYMPTOMS. In a real interactive system, you would replace
    this with real user answers.
    """

    # Normalize FULL_SYMPTOMS using the SAME function as the KG task.
    # This ensures that things like "runny/stuffy nose" match "nasal congestion".
    normalized_full = {
        _normalize_symptom_text(s)
        for s in (full_symptoms or [])
        if isinstance(s, str)
    }

    def has(symptom_name: str) -> bool:
        """
        Decide if a KG symptom is present in the “true” symptom set.
        We normalize the KG symptom name and check membership in normalized_full.
        """
        if not isinstance(symptom_name, str):
            return False
        canon = _normalize_symptom_text(symptom_name)
        return canon in normalized_full

    # ---- triage_start ----
    start_payload = {
        "query_type": "triage_start",
        "symptoms_text": current_symptoms,
    }
    state = json.loads(agent._execute([json.dumps(start_payload)]))

    # ---- clarification loop ----
    for _ in range(max_turns):
        if not state.get("ok", False):
            break

        result = state.get("result", {}) or {}
        if result.get("done"):
            break

        next_qs = result.get("next_questions", []) or []
        if not next_qs:
            break

        # take the first next question
        q = next_qs[0]
        scui = q["cui"]
        sname = q["name"]

        present_flag = has(sname)

        ans_payload = {
            "query_type": "triage_answer",
            "cui": scui,
            "present": present_flag,
        }
        state = json.loads(agent._execute([json.dumps(ans_payload)]))

    return state


# ---------- LLM prompt + call ----------
def build_llm_prompt(
    conversation: str,
    clarified_symptoms: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    question_log: List[Dict[str, Any]],
):
    """
    Build a tightly specified system+user prompt for diagnostic ranking.

    Design: LLM behaves like a clinician.
      - KG provides clarified PRESENT/ABSENT symptoms and candidate diagnoses
        with prior probability hints.
      - LLM can go beyond KG and propose its own diagnoses.
      - LLM must always return exactly 5 diagnoses.
    """
    present = [s for s in clarified_symptoms if s.get("status") == "present"]
    absent = [s for s in clarified_symptoms if s.get("status") == "absent"]
    unknown = [s for s in clarified_symptoms if s.get("status") not in ("present", "absent")]

    # Softmax-normalize KG scores into prior hints
    scores = [c["score"] for c in candidates[:10]] if candidates else []
    if scores:
        max_s = max(scores)
        exp_scores = [math.exp(s - max_s) for s in scores]
        total = sum(exp_scores) or 1.0
        priors = [e / total for e in exp_scores]
    else:
        priors = []

    kg_candidates_payload = []
    for idx, c in enumerate(candidates[:10]):
        prior = priors[idx] if idx < len(priors) else None
        kg_candidates_payload.append(
            {
                "name": c["name"],
                "cui": c["cui"],
                "score": c["score"],
                "match": c["match"],
                "prior_prob_hint": prior,
            }
        )

    payload = {
        "conversation": conversation,
        # Final clarified symptom set (original + KG-questioning)
        "symptoms": {
            "present": [{"name": s["name"], "cui": s["cui"]} for s in present],
            "absent": [{"name": s["name"], "cui": s["cui"]} for s in absent],
            "unknown": [{"name": s["name"], "cui": s["cui"]} for s in unknown],
        },
        "kg_candidates": kg_candidates_payload,
        "clarification_questions": question_log,
    }

    json_block = json.dumps(payload, indent=2)

    system_msg = (
        "You are a diagnostic reasoning assistant used in an offline RESEARCH experiment.\n"
        "You are NOT providing medical advice and your output will never be used for real patients.\n\n"
        "You are given structured case information:\n"
        "1) A synthetic patient conversation.\n"
        "2) A list of PRESENT / ABSENT / UNKNOWN symptoms that have been clarified by a knowledge-graph agent.\n"
        "3) A list of candidate diagnoses generated by the knowledge graph, each with a score and a prior probability hint.\n"
        "4) A log of clarification questions that were asked (which symptom was queried, and if it was present/absent).\n\n"
        "Interpretation rules:\n"
        "- Treat PRESENT and ABSENT symptoms as the main evidence, like a clinician.\n"
        "- Treat KG candidates as helpful prior hints, NOT hard constraints.\n"
        "- The field 'prior_prob_hint' is a softmax-normalized hint across candidates: if symptoms are ambiguous,\n"
        "  you may lean toward candidates with higher prior_prob_hint, but you are free to override them when the\n"
        "  clinical picture points elsewhere.\n"
        "- You may (and often should) consider diagnoses that are not in the KG candidate list when clinically plausible.\n"
        "- If you add a diagnosis that is not in kg_candidates, mark its 'source' as 'llm_extra' and explain why it fits\n"
        "  the symptom pattern.\n"
        "- When a symptom is ABSENT, down-weight diagnoses where that symptom is typically important.\n"
        "- When a symptom is PRESENT, up-weight diagnoses strongly linked to it.\n\n"
        "Output format (IMPORTANT):\n"
        "- Respond with a SINGLE valid JSON object and NOTHING else (no comments, no prose outside JSON).\n"
        "{\n"
        '  \"top_diagnoses\": [\n'
        '    {\n'
        '      \"name\": \"...\",                  // diagnosis label\n'
        '      \"probability\": 0.35,             // between 0 and 1; all probs sum to ~1.0\n'
        '      \"source\": \"kg\" | \"llm_extra\", // \"kg\" if it appears in kg_candidates, otherwise \"llm_extra\"\n'
        '      \"supporting_symptoms\": [\"...\"], // PRESENT symptoms supporting this dx\n'
        '      \"contradicting_symptoms\": [\"...\"], // ABSENT symptoms that argue against this dx\n'
        '      \"rationale\": \"Concise explanation using symptoms and KG prior.\"\n'
        '    },\n'
        "    ... exactly 5 total.\n"
        "  ],\n"
        "  \"global_explanation\": \"Short explanation of how the symptoms and KG candidates led to this ranking.\"\n"
        "}\n"
        "- You MUST return exactly 5 entries in 'top_diagnoses'. If there are only 1–2 strong candidates,\n"
        "  still add additional, less likely diagnoses (with smaller probabilities) so that there are 5 in total.\n"
        "- Probabilities must sum to approximately 1.0 across all returned diagnoses.\n"
        "- Do NOT include any disclaimers inside the JSON.\n"
        "- Do NOT give medical advice; this is for algorithm research only."
    )

    user_msg = (
        "Here is the structured case information to analyze:\n\n"
        f"{json_block}\n\n"
        "Use ONLY this information. Now produce the diagnostic JSON exactly in the specified format."
    )

    return system_msg, user_msg


def call_llm(system_msg: str, user_msg: str, model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    """
    Call the LLM and try to parse a JSON response.
    If parsing fails, keep the raw text so you can debug.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        data = {"raw_text": content}
    return data


# ---------- pretty printer for per-patient log ----------
def format_case_block(
    pid,
    true_dx,
    current_symptoms,
    full_symptoms,
    kg_expl,
    clarified,
    candidates,
    llm_out,
    kg_true_in_top5,
    llm_true_in_top5,
):
    lines = []
    lines.append("======================================")
    lines.append(f"PATIENT: {pid}")
    lines.append(f"True diagnosis: {true_dx}")
    lines.append(f"Current symptoms: {current_symptoms}")
    lines.append(f"Full symptoms   : {full_symptoms}")
    lines.append("")
    lines.append(f"KG explanation: {kg_expl}")
    lines.append("Clarified symptoms:")
    if clarified:
        for s in clarified:
            lines.append(f"  - {s['name']} [{s['status']}]")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("KG Top candidates:")
    if candidates:
        for c in candidates[:5]:
            lines.append(f"  - {c['name']} (score={c['score']}, match={c['match']})")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("LLM output:")
    lines.append(json.dumps(llm_out, indent=2))
    lines.append("")
    lines.append(f"True diagnosis in KG Top-5?: {kg_true_in_top5}")
    lines.append(f"True diagnosis in LLM Top-5?: {llm_true_in_top5}")
    lines.append("")
    return "\n".join(lines)


# ---------- main ----------
def main():
    csv_path = "data/test_patient_dx_symptoms.csv"
    df = pd.read_csv(csv_path)

    # Parse symptom lists
    df["FULL_SYMPTOMS_LIST"] = df["FULL_SYMPTOMS"].apply(safe_list_parse)
    df["CURRENT_SYMPTOMS_LIST"] = df["CURRENT_SYMPTOMS"].apply(safe_list_parse)

    has_conversation = "CONVERSATION" in df.columns

    agent = DiagnosticAmbiguityTask()

    summary_rows = []
    os.makedirs("data", exist_ok=True)
    log_path = "data/ambiguity_case_logs.txt"
    log_f = open(log_path, "w", encoding="utf-8")

    for _, row in df.iterrows():
        pid = row["PATIENT_ID"]
        true_dx = row["PATHOLOGY"]
        curr = row["CURRENT_SYMPTOMS_LIST"]
        full = row["FULL_SYMPTOMS_LIST"]
        conversation = row["CONVERSATION"] if has_conversation else ""

        if not curr:
            # if no current symptoms, skip this patient
            continue

        print("\n======================================")
        print(f"PATIENT: {pid}")
        print("True diagnosis:", true_dx)
        print("Current symptoms:", curr)
        print("Full symptoms   :", full)

        # ---- 1) KG ambiguity agent ----
        state = run_kg_ambiguity_for_patient(agent, curr, full, max_turns=20)
        if not state.get("ok", False):
            print("KG agent error:", state.get("error"))
            row_dict = row.to_dict()
            row_dict.update(
                {
                    "KG_ERROR": state.get("error"),
                    "KG_EXPLANATION": "",
                    "KG_CLARIFIED_SYMPTOMS_JSON": "[]",
                    "KG_CANDIDATES_JSON": "[]",
                    "KG_QUESTION_LOG_JSON": "[]",
                    "KG_TRUE_IN_TOP5": False,
                    "LLM_OUTPUT_JSON": "{}",
                    "LLM_TRUE_IN_TOP5": False,
                }
            )
            summary_rows.append(row_dict)
            continue

        result = state.get("result", {}) or {}
        clarified = result.get("clarified_symptoms", []) or []
        candidates = result.get("candidates", []) or []
        qlog = result.get("question_log", []) or []
        kg_expl = result.get("explanation", "")

        print("\nKG explanation:", kg_expl)
        print("Clarified symptoms:")
        for s in clarified:
            print(f"  - {s['name']} [{s['status']}]")

        print("\nKG Top candidates:")
        for c in candidates[:5]:
            print(f"  - {c['name']} (score={c['score']}, match={c['match']})")

        # KG Top-5 info + flag
        kg_top5_names = [c["name"] for c in candidates[:5]]
        kg_true_in_top5 = any(
            isinstance(true_dx, str)
            and isinstance(n, str)
            and true_dx.lower() in n.lower()
            for n in kg_top5_names
        )

        # ---- 2) LLM call ----
        system_msg, user_msg = build_llm_prompt(
            conversation=conversation,
            clarified_symptoms=clarified,
            candidates=candidates,
            question_log=qlog,
        )

        try:
            llm_out = call_llm(system_msg, user_msg)
        except Exception as e:
            print("LLM error:", e)
            llm_out = {"error": str(e)}

        print("\nLLM output:")
        print(json.dumps(llm_out, indent=2))

        # --------- USE ONLY LLM DIAGNOSES (KG is just context) ---------
        top_diag_objs: List[Dict[str, Any]] = []
        if isinstance(llm_out, dict) and "top_diagnoses" in llm_out:
            top_diag_objs = [
                d for d in llm_out.get("top_diagnoses", [])
                if isinstance(d, dict)
            ]

        # keep at most 5 from the LLM (it is instructed to ALWAYS output 5)
        top_diag_objs = top_diag_objs[:5]

        # fix probabilities: if missing or sum <= 0, assign simple distribution
        probs = []
        for d in top_diag_objs:
            p = d.get("probability", None)
            try:
                p = float(p)
            except Exception:
                p = None
            probs.append(p)

        if not top_diag_objs:
            # no usable diagnoses from the LLM → dummy entry, mostly for debugging
            top_diag_objs = [
                {
                    "name": "",
                    "probability": 1.0,
                    "source": "llm_extra",
                    "supporting_symptoms": [],
                    "contradicting_symptoms": [],
                    "rationale": "No valid diagnoses returned by LLM.",
                }
            ]
            probs = [1.0]

        if any(p is None for p in probs) or sum(p for p in probs if p is not None) <= 0:
            # fallback: primary dx gets 0.6, rest share 0.4
            n = len(top_diag_objs)
            if n > 1:
                first_p = 0.6
                rest_p = 0.4 / (n - 1)
                probs = [first_p] + [rest_p] * (n - 1)
            else:
                probs = [1.0]
        else:
            total = sum(probs)
            probs = [p / total for p in probs]

        for d, p in zip(top_diag_objs, probs):
            d["probability"] = float(p)

        # LLM Top-5 names for accuracy flag
        llm_top5_names = [str(d.get("name", "")) for d in top_diag_objs]
        llm_true_in_top5 = any(
            isinstance(true_dx, str)
            and isinstance(n, str)
            and true_dx.lower() in n.lower()
            for n in llm_top5_names
        )

        print("\nTrue diagnosis in KG Top-5?:", kg_true_in_top5)
        print("True diagnosis in LLM Top-5?:", llm_true_in_top5)

        # ---- build prediction fields (PRED1..PRED5) ----
        pred_rows = []
        for d in top_diag_objs:
            pred_rows.append(
                {
                    "name": str(d.get("name", "")),
                    "prob": float(d.get("probability", 0.0) or 0.0),
                    "source": str(d.get("source", "")),
                }
            )

        # pad to exactly 5 predictions in CSV (names may be empty if LLM misbehaves)
        while len(pred_rows) < 5:
            pred_rows.append({"name": "", "prob": 0.0, "source": ""})

        # ---- write pretty block to text log ----
        block = format_case_block(
            pid=pid,
            true_dx=true_dx,
            current_symptoms=curr,
            full_symptoms=full,
            kg_expl=kg_expl,
            clarified=clarified,
            candidates=candidates,
            llm_out=llm_out,
            kg_true_in_top5=kg_true_in_top5,
            llm_true_in_top5=llm_true_in_top5,
        )
        log_f.write(block + "\n")

        # ---- build CSV row with EVERYTHING ----
        row_dict = row.to_dict()  # all original CSV columns

        row_dict.update(
            {
                # KG
                "KG_ERROR": "",
                "KG_EXPLANATION": kg_expl,
                "KG_CLARIFIED_SYMPTOMS_JSON": json.dumps(clarified),
                "KG_CANDIDATES_JSON": json.dumps(candidates),
                "KG_QUESTION_LOG_JSON": json.dumps(qlog),
                "KG_TRUE_IN_TOP5": kg_true_in_top5,
                # LLM
                "LLM_OUTPUT_JSON": json.dumps(llm_out),
                "LLM_TRUE_IN_TOP5": llm_true_in_top5,
                # Final top-5 predictions (LLM over KG)
                "PRED1_NAME": pred_rows[0]["name"],
                "PRED1_PROB": pred_rows[0]["prob"],
                "PRED1_SOURCE": pred_rows[0]["source"],
                "PRED2_NAME": pred_rows[1]["name"],
                "PRED2_PROB": pred_rows[1]["prob"],
                "PRED2_SOURCE": pred_rows[1]["source"],
                "PRED3_NAME": pred_rows[2]["name"],
                "PRED3_PROB": pred_rows[2]["prob"],
                "PRED3_SOURCE": pred_rows[2]["source"],
                "PRED4_NAME": pred_rows[3]["name"],
                "PRED4_PROB": pred_rows[3]["prob"],
                "PRED4_SOURCE": pred_rows[3]["source"],
                "PRED5_NAME": pred_rows[4]["name"],
                "PRED5_PROB": pred_rows[4]["prob"],
                "PRED5_SOURCE": pred_rows[4]["source"],
            }
        )

        summary_rows.append(row_dict)

    log_f.close()
    print(f"\nWrote detailed text log to: {log_path}")

    # ---- write CSV with full info ----
    out_csv = "data/ambiguity_kg_llm_full.csv"
    if summary_rows:
        out_df = pd.DataFrame(summary_rows)
        out_df.to_csv(out_csv, index=False)
        print(f"Wrote full CSV to: {out_csv}")
    else:
        print("No rows to write – check your input / filters.")


if __name__ == "__main__":
    main()