"""
Prepare fine-tuning data from tot_synthetic.jsonl.

Key constraint: challenge inference runs with max_tokens=7680, max_model_len=8192.
The model can never generate more than 7680 tokens at inference time, so training
on longer traces is counterproductive — the model learns a behaviour it can never
reproduce.

Strategy:
  1. TRUNCATED traces (finish_reason=length):
       Keep the first TRAIN_TOKEN_BUDGET chars of reasoning (= the branch
       exploration, which is the valuable part), then hard-append:
           "\n\n### Final Answer\n$\\boxed{GT}$"
       The model learns: explore → conclude → answer, all within budget.

  2. COMPLETE traces (finish_reason=stop):
       Replace any wrong $\\boxed{...}$ with the ground truth answer.
       Keep as-is if already correct.

  3. Both paths produce training examples that:
       - Are always complete (boxed answer always present)
       - Always fit inside the 7680-token inference window
       - Teach the reasoning process, not just the answer

Output:
    data/finetune_sft.jsonl   — corrected messages[] for SFT
    data/finetune_rl.jsonl    — original trace + reward=1/0 for RL training
"""

import json
import re
from pathlib import Path
from collections import Counter

INPUT_FILE = Path("data/tot_synthetic.jsonl")
SFT_FILE   = Path("data/finetune_sft.jsonl")
RL_FILE    = Path("data/finetune_rl.jsonl")

# Leave room for the prompt tokens (~300-500 tokens) and the appended conclusion.
# 7000 tokens * 4 chars/token = 28 000 chars.  We use 24 000 to be conservative.
TRAIN_CHAR_BUDGET = 24_000   # ~6000 tokens for reasoning body
CONCLUSION_TEMPLATE = "\n\n### Final Answer\n$\\boxed{{{answer}}}$"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    return m.group(1).strip() if m else None


def replace_last_boxed(text: str, correct: str) -> tuple[str, bool]:
    """Replace the last \\boxed{...} with the correct answer. Returns (text, changed)."""
    replacement = f"$\\boxed{{{correct}}}$"
    matches = list(re.finditer(r'\$?\\boxed\{[^}]*\}\$?', text))
    if not matches:
        return text.rstrip() + CONCLUSION_TEMPLATE.format(answer=correct), True
    last = matches[-1]
    changed = last.group(0) != replacement
    return text[:last.start()] + replacement + text[last.end():], changed


def trim_and_cap(reasoning: str, gt: str) -> str:
    """
    For truncated traces: keep the branch-exploration portion up to the
    character budget, then append the correct final answer.
    The cut is made at the last clean paragraph boundary before the limit,
    so we don't split mid-sentence.
    """
    if len(reasoning) <= TRAIN_CHAR_BUDGET:
        # Already short enough — just append conclusion
        return reasoning.rstrip() + CONCLUSION_TEMPLATE.format(answer=gt)

    # Truncate at last double-newline before the budget
    chunk = reasoning[:TRAIN_CHAR_BUDGET]
    cut   = chunk.rfind("\n\n")
    if cut == -1:
        cut = TRAIN_CHAR_BUDGET
    trimmed = reasoning[:cut].rstrip()

    # Remove any dangling partial boxed expression at the cut boundary
    trimmed = re.sub(r'\$?\\boxed\{[^}]*$', '', trimmed).rstrip()

    return trimmed + CONCLUSION_TEMPLATE.format(answer=gt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run generate_tot_data.py first.")
        return

    with open(INPUT_FILE) as f:
        records = [json.loads(l) for l in f if l.strip()]

    total = len(records)

    # Counters
    exact_orig      = 0   # already correct before any fix
    boxed_replaced  = 0   # wrong answer corrected
    trimmed_capped  = 0   # truncated trace → trim-and-cap applied
    no_boxed_fixed  = 0   # force_finish failed → appended GT anyway

    sft_out = open(SFT_FILE, "w")
    rl_out  = open(RL_FILE,  "w")

    for r in records:
        gt            = r["answer"].strip()
        reasoning     = r["tot_reasoning"]
        finish_reason = r.get("finish_reason", "stop")
        predicted     = extract_boxed(reasoning) or ""
        is_correct    = (predicted == gt)
        user_msg      = r["messages"][0]["content"]

        if is_correct:
            exact_orig += 1

        # ------------------------------------------------------------------
        # Build the corrected assistant message
        # ------------------------------------------------------------------
        if finish_reason == "length":
            # Trace was cut off — trim to budget and append correct answer
            final_reasoning = trim_and_cap(reasoning, gt)
            trimmed_capped += 1
            was_changed = True
        else:
            # Trace completed — fix the boxed answer if wrong
            final_reasoning, was_changed = replace_last_boxed(reasoning, gt)
            if was_changed:
                if predicted:
                    boxed_replaced += 1
                else:
                    no_boxed_fixed += 1

        # Sanity check: final_reasoning must always contain the GT boxed answer
        assert gt in final_reasoning, f"BUG: GT not found in final for id={r['id']}"

        # ------------------------------------------------------------------
        # SFT record
        # ------------------------------------------------------------------
        sft_out.write(json.dumps({
            "id":       r["id"],
            "category": r["category"],
            "messages": [
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": final_reasoning},
            ],
            "answer":           gt,
            "was_corrected":    was_changed,
            "originally_correct": is_correct,
            "trim_applied":     finish_reason == "length",
            "approx_tokens":    len(final_reasoning) // 4,
        }) + "\n")

        # ------------------------------------------------------------------
        # RL record — raw original trace with reward signal
        # ------------------------------------------------------------------
        rl_out.write(json.dumps({
            "id":        r["id"],
            "category":  r["category"],
            "prompt":    user_msg,
            "response":  reasoning,         # ORIGINAL, uncorrected
            "answer":    gt,
            "predicted": predicted,
            "correct":   is_correct,
            "reward":    1.0 if is_correct else 0.0,
        }) + "\n")

    sft_out.close()
    rl_out.close()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"Input records       : {total}")
    print(f"Already correct     : {exact_orig}  ({exact_orig/total*100:.1f}%)")
    print(f"Boxed answer fixed  : {boxed_replaced}")
    print(f"Trim-and-cap applied: {trimmed_capped}  (truncated traces)")
    print(f"No-boxed appended   : {no_boxed_fixed}")
    print()

    # Token length stats of final SFT data
    with open(SFT_FILE) as f:
        sft_recs = [json.loads(l) for l in f if l.strip()]
    tok_lens = sorted(r["approx_tokens"] for r in sft_recs)
    n = len(tok_lens)
    print("Final SFT token lengths (approx):")
    print(f"  min={tok_lens[0]}  p25={tok_lens[n//4]}  median={tok_lens[n//2]}"
          f"  p75={tok_lens[3*n//4]}  p95={tok_lens[int(n*0.95)]}  max={tok_lens[-1]}")
    over_budget = sum(1 for t in tok_lens if t > 7680)
    print(f"  Exceeds 7680-token inference limit: {over_budget}/{n}")
    print()
    print(f"SFT  → {SFT_FILE}  ({n} records)")
    print(f"RL   → {RL_FILE}   ({total} records)")
    print()

    # Per-category accuracy of original ToT traces
    cat_total   = Counter(r["category"] for r in records)
    cat_correct = Counter(
        r["category"] for r in records
        if (extract_boxed(r["tot_reasoning"]) or "") == r["answer"].strip()
    )
    print("Category accuracy (original ToT, before correction):")
    for cat in sorted(cat_total):
        n, c = cat_total[cat], cat_correct[cat]
        print(f"  {cat:<22} {c:>4}/{n:<4}  ({c/n*100:5.1f}%)")


if __name__ == "__main__":
    main()
