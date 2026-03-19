"""
Tree of Thought (ToT) synthetic data generation pipeline.

Runs both Cerebras API keys in parallel for 2× throughput.
Each key gets its own workers, rate limiter, and client.
A shared ID lock prevents duplicate processing.

Key features:
  - Parallel dual-key Cerebras (2× throughput)
  - Per-key adaptive rate limiter
  - Thread-safe ID locking (no duplicates across keys)
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
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI, RateLimitError, APIError
from poe_pruner import normalize_trace, prune_if_needed

# ---------------------------------------------------------------------------
# Provider config — both Cerebras keys run in parallel
# ---------------------------------------------------------------------------
PROVIDERS = {
    "cerebras_1": {
        "api_key":  os.environ.get("CEREBRAS_API_KEY_1", ""),
        "base_url": "https://api.cerebras.ai/v1",
        "model":    "qwen-3-235b-a22b-instruct-2507",
        "rpm":      15.0,
        "logprobs": True,
    },
    "cerebras_2": {
        "api_key":  os.environ.get("CEREBRAS_API_KEY_2", ""),
        "base_url": "https://api.cerebras.ai/v1",
        "model":    "qwen-3-235b-a22b-instruct-2507",
        "rpm":      15.0,
        "logprobs": True,
    },
    # "openrouter": {
    #     "api_key":  os.environ.get("OPENROUTER_API_KEY", ""),
    #     "base_url": "https://openrouter.ai/api/v1",
    #     "model":    "qwen/qwen3-235b-a22b-2507",
    #     "rpm":      20.0,
    #     "logprobs": False,
    # },
}

DAILY_QUOTA_THRESHOLD = 3600

DATA_DIR    = Path("data")
OUTPUT_FILE = Path("data/tot_synthetic.jsonl")
FAILED_FILE = Path("data/tot_failed.jsonl")

MAX_TOKENS  = 16000
MAX_RETRIES = 6
BASE_DELAY  = 5.0


# ---------------------------------------------------------------------------
# Adaptive rate limiter (one per key)
# ---------------------------------------------------------------------------
class AdaptiveRateLimiter:
    def __init__(self, rpm: float = 15.0):
        self._interval      = 60.0 / rpm
        self._last_req      = 0.0
        self._lock          = threading.Lock()
        self._backoff_until = 0.0
        self._fail_streak   = 0

    def acquire(self):
        while True:
            with self._lock:
                now  = time.time()
                wait = max(
                    self._backoff_until - now,
                    self._interval - (now - self._last_req),
                )
                if wait <= 0:
                    self._last_req = time.time()
                    return
            time.sleep(min(wait, 1.0))

    def on_rate_limit(self, retry_after: float | None = None):
        with self._lock:
            self._fail_streak += 1
            if retry_after:
                delay = retry_after + random.uniform(0, 2)
            else:
                delay = BASE_DELAY * (2 ** self._fail_streak) + random.uniform(0, 3)
            self._backoff_until = time.time() + delay
            print(f"  [rate-limit] backing off {delay:.1f}s (streak={self._fail_streak})",
                  flush=True)

    def on_success(self):
        with self._lock:
            if self._fail_streak > 0:
                self._fail_streak = max(0, self._fail_streak - 1)


# ---------------------------------------------------------------------------
# Provider handle — one per API key, used by workers directly
# ---------------------------------------------------------------------------
class ProviderHandle:
    """Lightweight handle for a single API key."""
    def __init__(self, name: str, cfg: dict):
        self.name     = name
        self.client   = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        self.model    = cfg["model"]
        self.logprobs = cfg.get("logprobs", False)
        self.limiter  = AdaptiveRateLimiter(rpm=cfg["rpm"])
        self.exhausted = False   # set True when daily quota hits

    def acquire(self):
        self.limiter.acquire()

    def on_rate_limit(self, retry_after=None):
        if retry_after and retry_after >= DAILY_QUOTA_THRESHOLD:
            self.exhausted = True
            print(f"\n  [provider] {self.name} daily quota exhausted", flush=True)
        else:
            self.limiter.on_rate_limit(retry_after)

    def on_success(self):
        self.limiter.on_success()


# ---------------------------------------------------------------------------
# Shared ID lock — prevents two keys from processing the same puzzle
# ---------------------------------------------------------------------------
class IDLock:
    """Thread-safe set of claimed IDs. Prevents duplicate work across keys."""
    def __init__(self, done_ids: set):
        self._lock = threading.Lock()
        self._ids  = set(done_ids)

    def try_claim(self, puzzle_id: str) -> bool:
        """Atomically claim an ID. Returns True if claimed, False if already taken."""
        with self._lock:
            if puzzle_id in self._ids:
                return False
            self._ids.add(puzzle_id)
            return True

    def release(self, puzzle_id: str):
        """Release a claim (on failure, so another key can retry)."""
        with self._lock:
            self._ids.discard(puzzle_id)




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
def _call_api(ph: ProviderHandle, messages: list, temperature: float = 0.5) -> tuple[str, str]:
    """Make one API call via the given provider handle. Returns (content, finish_reason)."""
    for attempt in range(1, MAX_RETRIES + 1):
        ph.acquire()
        try:
            resp = ph.client.chat.completions.create(
                model       = ph.model,
                messages    = messages,
                temperature = temperature,
                max_tokens  = MAX_TOKENS,
                top_p       = 0.95,
            )
            ph.on_success()
            return resp.choices[0].message.content.strip(), resp.choices[0].finish_reason

        except RateLimitError as e:
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after", 0)) or None
            except Exception:
                pass
            ph.on_rate_limit(retry_after)
            if attempt == MAX_RETRIES:
                raise

        except APIError as e:
            wait = BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
            print(f"    [{ph.name}] api-error attempt {attempt}/{MAX_RETRIES}: {e} — sleeping {wait:.1f}s")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                raise


def _has_boxed(text: str) -> bool:
    return "\\boxed{" in text


def generate_tot(row: dict, ph: ProviderHandle) -> dict:
    """Generate a ToT trace for one puzzle row using the given provider handle."""
    category = detect_category(row["prompt"])
    user_msg = build_user_prompt(row["prompt"], category)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    content, finish_reason = _call_api(ph, messages)

    force_finish_used = False
    if not _has_boxed(content):
        follow_msgs = messages + [
            {"role": "assistant", "content": content},
            {"role": "user",      "content": FORCE_FINISH_PROMPT},
        ]
        extra, _ = _call_api(ph, follow_msgs, temperature=0.0)
        content = content + "\n\n---\n\n" + extra
        force_finish_used = True

    # Normalize section structure before pruning so footer-aware logic has stable anchors.
    answer = row.get("answer", "")
    content = normalize_trace(content, answer=answer)

    # PoE pruning — only if trace exceeds 8000-token inference budget
    orig_len = len(content)
    content, poe_pruned, prune_method = prune_if_needed(
        client        = ph.client,
        model         = ph.model,
        trace         = content,
        puzzle_prompt = row["prompt"],
        answer        = answer,
        use_logprobs  = ph.logprobs,
    )

    if poe_pruned:
        print(f"  [{ph.name}] [poe] {row['id']}: pruned {orig_len//4}→{len(content)//4} tokens via {prune_method}",
              flush=True)

    # Run one final normalization pass so non-pruned traces also keep the required footer shape.
    content = normalize_trace(content, answer=answer)
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
        "provider":         ph.name,
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
# Main — parallel dual-key execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate ToT data — parallel Cerebras keys"
    )
    parser.add_argument("--input",    default=str(DATA_DIR / "train.csv"))
    parser.add_argument("--output",   default=str(OUTPUT_FILE))
    parser.add_argument("--limit",    type=int,   default=None)
    parser.add_argument("--workers-per-key", type=int, default=2,
                        help="Worker threads per API key (total = keys × this)")
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    input_file  = Path(args.input)
    output_file = Path(args.output)

    # Build provider handles
    handles = []
    for name, cfg in PROVIDERS.items():
        if cfg["api_key"]:
            handles.append(ProviderHandle(name, cfg))
        else:
            print(f"  [skip] {name}: no API key set")

    if not handles:
        print("ERROR: No valid API keys found. Set CEREBRAS_API_KEY_1 / _2 in .env")
        return

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

    # Shared ID lock — prevents two keys from working on the same puzzle
    id_lock = IDLock(done_ids)

    total = len(pending)
    rpm_total = sum(PROVIDERS[h.name]["rpm"] for h in handles)
    eta_hours = total / rpm_total / 60.0
    total_workers = len(handles) * args.workers_per_key

    print(f"Keys       : {len(handles)} ({', '.join(h.name for h in handles)})")
    print(f"Combined   : {rpm_total} rpm  |  {total_workers} workers")
    print(f"Input      : {input_file}  ({len(rows)} rows)")
    print(f"Output     : {output_file}")
    print(f"Already    : {len(done_ids)} done")
    print(f"Pending    : {total}")
    print(f"Max tokens : {MAX_TOKENS}")
    print(f"Est. time  : {eta_hours:.1f} h  ({eta_hours*60:.0f} min)")
    print()

    if total == 0:
        print("Nothing to do.")
        return

    # Shared counters
    _stats_lock = threading.Lock()
    stats = {"completed": 0, "failed": 0, "token_sum": 0}
    start = time.time()

    def worker_loop(ph: ProviderHandle, work_queue: list):
        """Each worker drains from a shared queue using the ID lock."""
        for row in work_queue:
            if ph.exhausted:
                break
            # Atomically claim this ID
            if not id_lock.try_claim(row["id"]):
                continue  # another key already processing/completed it
            try:
                record = generate_tot(row, ph)
                append_record(output_file, record)
                with _stats_lock:
                    stats["completed"] += 1
                    stats["token_sum"] += record["approx_tokens"]
                    done_n = stats["completed"] + stats["failed"]
                    if done_n % 10 == 0:
                        elapsed = time.time() - start
                        rate = done_n / elapsed if elapsed > 0 else 0
                        eta_s = (total - done_n) / rate if rate > 0 else 0
                        avg_tok = stats["token_sum"] // stats["completed"] if stats["completed"] else 0
                        print(
                            f"  [{done_n}/{total}] "
                            f"ok={stats['completed']} fail={stats['failed']} "
                            f"rate={rate:.2f}/s  avg_tok≈{avg_tok}  "
                            f"eta={eta_s/3600:.2f}h",
                            flush=True,
                        )
            except Exception as e:
                id_lock.release(row["id"])  # release so other key can retry
                with _stats_lock:
                    stats["failed"] += 1
                append_failed(FAILED_FILE, row["id"], f"[{ph.name}] {e}")
                print(f"  [{ph.name}] [failed] {row['id']}: {e}", flush=True)

    # Shuffle pending so both keys work on different puzzles
    random.shuffle(pending)

    # Launch worker threads — each key gets its own pool, all share the queue
    all_threads = []
    for ph in handles:
        for i in range(args.workers_per_key):
            t = threading.Thread(
                target=worker_loop,
                args=(ph, pending),
                name=f"{ph.name}-w{i}",
                daemon=True,
            )
            t.start()
            all_threads.append(t)
            print(f"  Started {t.name}", flush=True)

    # Wait for all workers to finish
    for t in all_threads:
        t.join()

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed/3600:.2f}h  —  {stats['completed']}/{total} records written to {output_file}")
    if stats["failed"]:
        print(f"  {stats['failed']} failures logged in {FAILED_FILE}  (re-run to retry them)")


if __name__ == "__main__":
    main()
