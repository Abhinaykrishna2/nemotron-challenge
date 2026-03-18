"""
Tree of Thought (ToT) synthetic data generation pipeline.

Primary provider  : Cerebras  (qwen-3-235b-a22b-instruct-2507)
Fallback provider : OpenRouter (qwen/qwen3-235b-a22b-2507) — same model,
                    auto-activates when Cerebras daily quota is exhausted.

Key features:
  - Dual-provider with automatic failover (Cerebras → OpenRouter)
  - Adaptive rate limiter per provider
  - PoE branch pruning post-hoc when trace > 8000 tokens
  - Resume support: already-processed IDs are skipped on restart
  - Force-finish pass if model gets cut off before $\\boxed{}$
"""

import csv
import json
import math
import os
import random
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI, RateLimitError, APIError
from poe_pruner import prune_if_needed

# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------
PROVIDERS = {
    "cerebras": {
        "api_key":  os.environ["CEREBRAS_API_KEY"],
        "base_url": "https://api.cerebras.ai/v1",
        "model":    "qwen-3-235b-a22b-instruct-2507",
        "rpm":      15.0,
        "logprobs": True,     # Cerebras supports logprobs → real PoE scoring
    },
    # "openrouter": {
    #     "api_key":  os.environ["OPENROUTER_API_KEY"],
    #     "base_url": "https://openrouter.ai/api/v1",
    #     "model":    "qwen/qwen3-235b-a22b-2507",
    #     "rpm":      20.0,
    #     "logprobs": False,    # OpenRouter/WandB doesn't → heuristic PoE fallback
    # },
}

# Daily quota threshold — if retry-after >= this, treat as daily cap exhausted
DAILY_QUOTA_THRESHOLD = 3600   # 1 hour+

DATA_DIR    = Path("data")
OUTPUT_FILE = Path("data/tot_synthetic.jsonl")
FAILED_FILE = Path("data/tot_failed.jsonl")

MAX_TOKENS  = 16000   # let model fully complete; PoE prunes if > 8000 tokens
MAX_RETRIES = 6
BASE_DELAY  = 5.0


# ---------------------------------------------------------------------------
# Provider manager — Cerebras primary, OpenRouter fallback
# ---------------------------------------------------------------------------
class ProviderManager:
    """
    Manages dual-provider failover.
    Starts on Cerebras; if a retry-after >= DAILY_QUOTA_THRESHOLD is received,
    permanently switches all workers to OpenRouter for the rest of the run.
    """
    def __init__(self):
        self._lock    = threading.Lock()
        self._active  = "cerebras"
        self._clients = {}
        self._limiters = {}
        for name, cfg in PROVIDERS.items():
            self._clients[name]  = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
            self._limiters[name] = AdaptiveRateLimiter(rpm=cfg["rpm"])

    @property
    def active(self):
        with self._lock:
            return self._active

    @property
    def model(self):
        return PROVIDERS[self.active]["model"]

    def client(self):
        return self._clients[self.active]

    def limiter(self):
        return self._limiters[self.active]

    def on_daily_quota(self):
        with self._lock:
            if "openrouter" in PROVIDERS and self._active == "cerebras":
                self._active = "openrouter"
                print("\n  [provider] Cerebras daily quota exhausted → switching to OpenRouter",
                      flush=True)
            else:
                print("\n  [provider] Cerebras daily quota exhausted — no fallback, will retry after reset",
                      flush=True)

    def on_rate_limit(self, retry_after=None):
        if retry_after and retry_after >= DAILY_QUOTA_THRESHOLD:
            self.on_daily_quota()
        else:
            self.limiter().on_rate_limit(retry_after)

    @property
    def supports_logprobs(self):
        return PROVIDERS[self.active].get("logprobs", False)

    def on_success(self):
        self.limiter().on_success()

    def acquire(self):
        self.limiter().acquire()


_provider: ProviderManager | None = None


# ---------------------------------------------------------------------------
# Adaptive rate limiter
# ---------------------------------------------------------------------------
class AdaptiveRateLimiter:
    """
    Token-bucket rate limiter shared across all worker threads.

    - Enforces a minimum interval between requests (60 / rpm seconds).
    - On a 429, all workers pause for an exponentially growing backoff.
    - Backoff resets after a run of consecutive successes.
    """

    def __init__(self, rpm: float = 15.0):
        self._interval   = 60.0 / rpm   # minimum seconds between requests
        self._last_req   = 0.0
        self._lock       = threading.Lock()
        self._backoff_until = 0.0
        self._fail_streak   = 0

    def acquire(self):
        """Block until a request slot is available."""
        while True:
            with self._lock:
                now  = time.time()
                wait = max(
                    self._backoff_until - now,           # global backoff
                    self._interval - (now - self._last_req),  # pacing
                )
                if wait <= 0:
                    self._last_req = time.time()
                    return
            time.sleep(min(wait, 1.0))   # wake up frequently to re-check

    def on_rate_limit(self, retry_after: float | None = None):
        with self._lock:
            self._fail_streak += 1
            if retry_after:
                delay = retry_after + random.uniform(0, 2)
            else:
                delay = BASE_DELAY * (2 ** self._fail_streak) + random.uniform(0, 3)
            self._backoff_until = time.time() + delay
            print(
                f"  [rate-limit] backing off {delay:.1f}s "
                f"(streak={self._fail_streak})",
                flush=True,
            )

    def on_success(self):
        with self._lock:
            if self._fail_streak > 0:
                self._fail_streak = max(0, self._fail_streak - 1)




# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "bit_manipulation": ["bit manipulation", "8-bit", "binary"],
    "cipher":           ["encryption", "decrypt", "cipher"],
    "numeral_system":   ["numeral system", "base ", "hexadecimal", "octal"],
    "unit_conversion":  ["unit conversion", "measurement"],
    "physics":          ["gravitational", "velocity", "acceleration", "force"],
    "math_rule":        ["transformation rules", "equations", "arithmetic", "polynomial"],
}

def detect_category(prompt: str) -> str:
    p = prompt.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in p for kw in kws):
            return cat
    return "general"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a concise mathematical reasoning assistant. Solve puzzles using a Tree of Thought approach. Your entire response MUST stay under 4000 tokens.

## Required Structure

### Branch 1: <Hypothesis Name>
- Hypothesis: one sentence
- Quick test (2–3 examples): show input → computed → expected [✓/✗]
- Verdict: VALID ✓ or INVALID ✗

### Branch 2: <Hypothesis Name>
(same format)

### Branch 3: <Hypothesis Name>  (only if branches 1–2 both failed)
(same format)

### Selected Rule
One sentence: describe the validated rule.

### Answer Derivation
Apply the rule step-by-step to the target. Be explicit but brief (≤15 lines).

### Final Answer
$\\boxed{<answer>}$

## Hard Rules
- Test each branch on at most 3 examples; stop testing a branch the moment it fails.
- Do NOT explore more than 3 branches.
- The boxed answer must exactly match the expected output format.
- Total response ≤ 4000 tokens — be direct, skip filler text.
"""

CATEGORY_HINTS = {
    "bit_manipulation": (
        "Operations to try (in order of likelihood): "
        "circular left/right rotation, XOR with constant, bitwise NOT, "
        "bit reversal, left/right shift, OR/AND mask, swap nibbles."
    ),
    "cipher": (
        "Build a word-to-word or letter-to-letter substitution table "
        "from the examples. Look for the key that maps every encrypted "
        "token to a unique plaintext token."
    ),
    "numeral_system": (
        "Check base conversions: binary, octal, decimal, hex, Roman "
        "numerals, or non-standard bases. Identify the source and target base."
    ),
    "unit_conversion": (
        "Compute the conversion factor from two examples, verify on a "
        "third, then apply to the target."
    ),
    "physics": (
        "Identify the modified physical constant or formula from the "
        "examples (e.g., g, c, or a scaling factor) and apply it."
    ),
    "math_rule": (
        "Look for a hidden arithmetic or algebraic rule: polynomial, "
        "modular arithmetic, digit manipulation, or composed operations."
    ),
    "general": "",
}

FORCE_FINISH_PROMPT = (
    "You stopped before giving the final answer. "
    "In ≤ 100 words, state your best answer and wrap it: $\\boxed{<answer>}$"
)


def build_user_prompt(puzzle_prompt: str, category: str) -> str:
    hint = CATEGORY_HINTS.get(category, "")
    hint_block = f"[Hint: {hint}]\n\n" if hint else ""
    return (
        f"{hint_block}"
        f"{puzzle_prompt}\n\n"
        "Solve using Tree of Thought (≤ 4000 tokens total). "
        "End with $\\boxed{<answer>}$."
    )


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _call_api(messages: list, temperature: float = 0.5) -> tuple[str, str]:
    """Make one API call via active provider with retry + failover. Returns (content, finish_reason)."""
    for attempt in range(1, MAX_RETRIES + 1):
        _provider.acquire()
        try:
            resp = _provider.client().chat.completions.create(
                model       = _provider.model,
                messages    = messages,
                temperature = temperature,
                max_tokens  = MAX_TOKENS,
                top_p       = 0.95,
            )
            _provider.on_success()
            return resp.choices[0].message.content.strip(), resp.choices[0].finish_reason

        except RateLimitError as e:
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after", 0)) or None
            except Exception:
                pass
            _provider.on_rate_limit(retry_after)
            if attempt == MAX_RETRIES:
                raise

        except APIError as e:
            wait = BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
            print(f"    [api-error] attempt {attempt}/{MAX_RETRIES}: {e} — sleeping {wait:.1f}s")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                raise


def _has_boxed(text: str) -> bool:
    return "\\boxed{" in text


def generate_tot(row: dict) -> dict:
    """Generate a ToT trace for one puzzle row. Returns a record dict."""
    category = detect_category(row["prompt"])
    user_msg = build_user_prompt(row["prompt"], category)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    content, finish_reason = _call_api(messages)

    force_finish_used = False
    if not _has_boxed(content):
        follow_msgs = messages + [
            {"role": "assistant", "content": content},
            {"role": "user",      "content": FORCE_FINISH_PROMPT},
        ]
        extra, _ = _call_api(follow_msgs, temperature=0.0)
        content = content + "\n\n---\n\n" + extra
        force_finish_used = True

    # PoE pruning — only if trace exceeds 8000-token inference budget
    answer = row.get("answer", "")
    orig_len = len(content)
    content, poe_pruned = prune_if_needed(
        client        = _provider.client(),
        model         = _provider.model,
        trace         = content,
        puzzle_prompt = row["prompt"],
        answer        = answer,
        use_logprobs  = _provider.supports_logprobs,
    )

    if poe_pruned:
        method = "PoE logprobs" if _provider.supports_logprobs else "structural heuristic"
        print(f"  [poe] {row['id']}: pruned {orig_len//4}→{len(content)//4} tokens via {method}",
              flush=True)

    approx_tokens = len(content) // 4

    return {
        "id":               row["id"],
        "category":         category,
        "prompt":           row["prompt"],
        "answer":           answer,
        "tot_reasoning":    content,
        "has_boxed_answer": _has_boxed(content),
        "force_finish_used": force_finish_used,
        "poe_pruned":       poe_pruned,
        "finish_reason":    finish_reason,
        "approx_tokens":    approx_tokens,
        "messages": [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": content},
        ],
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()


def load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def append_record(path: Path, record: dict):
    with _write_lock:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def append_failed(path: Path, row_id: str, error: str):
    with _write_lock:
        with open(path, "a") as f:
            f.write(json.dumps({"id": row_id, "error": error}) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _provider

    parser = argparse.ArgumentParser(
        description="Generate ToT data — Cerebras primary, OpenRouter fallback"
    )
    parser.add_argument("--input",    default=str(DATA_DIR / "train.csv"))
    parser.add_argument("--output",   default=str(OUTPUT_FILE))
    parser.add_argument("--limit",    type=int,   default=None)
    parser.add_argument("--workers",  type=int,   default=2)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    _provider = ProviderManager()

    input_file  = Path(args.input)
    output_file = Path(args.output)

    # Load data
    with open(input_file) as f:
        rows = list(csv.DictReader(f))

    if args.category:
        rows = [r for r in rows if detect_category(r["prompt"]) == args.category]
        print(f"Category filter '{args.category}': {len(rows)} rows")

    # Dedup: skip already-done IDs
    done_ids = load_done_ids(output_file)
    pending  = [r for r in rows if r["id"] not in done_ids]
    if args.limit:
        pending = pending[: args.limit]

    total = len(pending)
    rpm_primary = PROVIDERS["cerebras"]["rpm"]
    eta_hours   = total / rpm_primary / 60.0
    print(f"Primary    : Cerebras   ({PROVIDERS['cerebras']['model']})")
    if "openrouter" in PROVIDERS:
        print(f"Fallback   : OpenRouter ({PROVIDERS['openrouter']['model']})")
    else:
        print(f"Fallback   : None (Cerebras only)")
    print(f"Input      : {input_file}  ({len(rows)} rows)")
    print(f"Output     : {output_file}")
    print(f"Already    : {len(done_ids)} done")
    print(f"Pending    : {total}")
    print(f"Workers    : {args.workers}  |  Max tokens: {MAX_TOKENS}")
    print(f"Est. time  : {eta_hours:.1f} h  ({eta_hours*60:.0f} min)")
    print()

    if total == 0:
        print("Nothing to do.")
        return

    completed = 0
    failed    = 0
    token_sum = 0
    start     = time.time()

    def process_row(row):
        return generate_tot(row)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_row, r): r for r in pending}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                record = fut.result()
                append_record(output_file, record)
                completed += 1
                token_sum += record["approx_tokens"]
            except Exception as e:
                failed += 1
                append_failed(FAILED_FILE, row["id"], str(e))
                print(f"  [failed] {row['id']}: {e}", flush=True)

            done_n  = completed + failed
            elapsed = time.time() - start
            rate    = done_n / elapsed if elapsed > 0 else 0
            eta_s   = (total - done_n) / rate if rate > 0 else 0
            avg_tok = token_sum // completed if completed else 0

            if done_n % 10 == 0 or done_n == total:
                print(
                    f"  [{done_n}/{total}] "
                    f"ok={completed} fail={failed} "
                    f"rate={rate:.2f}/s  "
                    f"avg_tokens≈{avg_tok}  "
                    f"eta={eta_s/3600:.2f}h",
                    flush=True,
                )

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed/3600:.2f}h  —  {completed}/{total} records written to {output_file}")
    if failed:
        print(f"  {failed} failures logged in {FAILED_FILE}  (re-run to retry them)")


if __name__ == "__main__":
    main()
