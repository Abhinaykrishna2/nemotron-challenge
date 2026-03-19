"""
PoE-based branch pruner for Tree of Thought reasoning traces.

Only invoked when a generated trace exceeds the 8000-token inference budget.

Scoring strategy (in order of preference):
  1. Fixed-answer echo scoring via the legacy completions endpoint when supported.
     This measures log P(answer | prompt + branch) directly.
  2. Chat-native answer generation scoring when only chat logprobs are available.
     This asks the model to emit the final boxed answer from the branch and scores
     the generated answer tokens if they match the known answer exactly.
  3. Structure-aware heuristic fallback (when logprobs are unavailable):
       a. INVALID branches → score -inf  (remove first, they're explicitly wrong)
       b. VALID branches not referenced in footer → score -1  (remove next)
       c. VALID branch referenced in footer (the selected/winning branch) → score 0 (keep last)
       d. If still over budget → truncate derivation section, never branches or answer

From poe_architecture.txt:
  log P(s|p) = (1/m) * Σ_j log P̂_j(s|p) - log Z
  Each branch is a "view"; we keep the views that most support the final answer.
"""

import re
from dataclasses import dataclass
from openai import OpenAI

TOKEN_BUDGET  = 8000
CHARS_PER_TOK = 4
BUDGET_CHARS  = TOKEN_BUDGET * CHARS_PER_TOK   # 32 000 chars

DERIVATION_MARKERS = re.compile(
    r'(#{2,3}\s+Answer Derivation|#{2,3}\s+Derivation)', re.IGNORECASE
)
SELECTED_RULE_RE = re.compile(r'#{2,3}\s+Selected Rule', re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(r'#{2,3}\s+Final Answer', re.IGNORECASE)
BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')
CHAT_SCORE_SYSTEM_PROMPT = (
    "You are scoring a candidate reasoning branch. "
    "Return only the final answer implied by the branch in the form "
    "$\\boxed{answer}$. Do not explain."
)
NUMERIC_REL_TOL = 0.01


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Branch:
    header:   str
    body:     str
    is_valid: bool
    score:    float = 0.0


@dataclass
class ParsedTrace:
    preamble: str
    branches: list[Branch]
    footer:   str          # Selected Rule + Derivation + Final Answer


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

BRANCH_PATTERN = re.compile(r'(#{2,3}\s+Branch\s+\d+[^\n]*\n)', re.IGNORECASE)

FOOTER_MARKERS = [
    r'#{2,3}\s+Selected Rule',
    r'#{2,3}\s+Branch Selection',
    r'#{2,3}\s+Answer Derivation',
    r'#{2,3}\s+Final Answer',
    r'\*\*Selected Rule',
    r'\*\*Answer',
]
FOOTER_RE  = re.compile('|'.join(FOOTER_MARKERS), re.IGNORECASE)
VALID_RE   = re.compile(r'VALID\s*✓|→\s*VALID|Verdict:\s*VALID', re.IGNORECASE)
INVALID_RE = re.compile(r'INVALID\s*✗|→\s*INVALID|Verdict:\s*INVALID', re.IGNORECASE)


def parse_trace(trace: str) -> ParsedTrace:
    headers      = list(BRANCH_PATTERN.finditer(trace))
    if not headers:
        return ParsedTrace(preamble="", branches=[], footer=trace)

    preamble         = trace[: headers[0].start()]
    last_branch_end  = headers[-1].end()
    footer_match     = FOOTER_RE.search(trace, last_branch_end)

    branches = []
    for i, hdr in enumerate(headers):
        body_start = hdr.start()
        body_end   = (headers[i + 1].start() if i + 1 < len(headers)
                      else (footer_match.start() if footer_match else len(trace)))
        body       = trace[body_start:body_end]
        is_valid   = bool(VALID_RE.search(body)) and not bool(INVALID_RE.search(body))
        branches.append(Branch(header=hdr.group(0).strip(), body=body, is_valid=is_valid))

    footer = trace[footer_match.start():] if footer_match else ""
    return ParsedTrace(preamble=preamble, branches=branches, footer=footer)


def reconstruct(parsed: ParsedTrace) -> str:
    parts = [parsed.preamble] if parsed.preamble else []
    for b in parsed.branches:
        parts.append(b.body)
    if parsed.footer:
        parts.append(parsed.footer)
    return "\n".join(p.rstrip() for p in parts if p.strip())


def trace_chars(parsed: ParsedTrace) -> int:
    return len(reconstruct(parsed))


# ---------------------------------------------------------------------------
# Trace normalization
# ---------------------------------------------------------------------------

def _extract_last_boxed(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _answers_match(predicted: str | None, answer: str | None) -> bool:
    if not predicted or not answer:
        return False
    if predicted == answer:
        return True
    try:
        predicted_num = float(predicted)
        answer_num = float(answer)
    except Exception:
        return False

    if answer_num == 0:
        return abs(predicted_num - answer_num) <= 1e-12
    return abs(predicted_num - answer_num) / abs(answer_num) <= NUMERIC_REL_TOL


def _extract_rule_text(parsed: ParsedTrace) -> str:
    candidates = [b for b in parsed.branches if b.is_valid] or parsed.branches
    if not candidates:
        return "Rule not explicitly stated in the trace."

    best = candidates[-1]
    hyp_match = re.search(r'Hypothesis:\s*(.+)', best.body)
    if hyp_match:
        return hyp_match.group(1).strip()

    header_text = best.header.split(":", 1)[-1].strip()
    return header_text or "Rule not explicitly stated in the trace."


def _insert_before(match_re: re.Pattern[str], text: str, block: str) -> str:
    match = match_re.search(text)
    if not match:
        return text.rstrip() + "\n\n" + block
    return text[:match.start()].rstrip() + "\n\n" + block + "\n\n" + text[match.start():].lstrip()


def _hard_cap(text: str, answer_text: str | None, budget_chars: int) -> str:
    if len(text) <= budget_chars:
        return text.rstrip()

    footer_starts = [
        match.start()
        for match in (
            SELECTED_RULE_RE.search(text),
            DERIVATION_MARKERS.search(text),
            FINAL_ANSWER_RE.search(text),
        )
        if match
    ]
    if footer_starts:
        footer_start = min(footer_starts)
        footer = text[footer_start:].rstrip()
        if answer_text and f"\\boxed{{{answer_text}}}" not in footer:
            footer = footer.rstrip() + f"\n\n### Final Answer\n$\\boxed{{{answer_text}}}$"

        if len(footer) < budget_chars:
            joiner = "\n\n"
            head_budget = budget_chars - len(footer) - len(joiner)
            head = text[:footer_start][:max(head_budget, 0)].rstrip()
            result = head + (joiner if head else "") + footer.lstrip()
            if len(result) <= budget_chars:
                return result.rstrip()

    if answer_text:
        answer_block = f"\n\n### Final Answer\n$\\boxed{{{answer_text}}}$"
        head_budget = max(budget_chars - len(answer_block), 0)
        return (text[:head_budget].rstrip() + answer_block).rstrip()

    return text[:budget_chars].rstrip()


def normalize_trace(
    trace: str,
    answer: str | None = None,
    *,
    enforce_answer: bool = False,
    budget_chars: int | None = None,
) -> str:
    result = trace.rstrip()
    parsed = parse_trace(result)
    rule_text = _extract_rule_text(parsed)

    if not SELECTED_RULE_RE.search(result):
        rule_block = f"### Selected Rule\n{rule_text}"
        anchor = DERIVATION_MARKERS if DERIVATION_MARKERS.search(result) else FINAL_ANSWER_RE
        result = _insert_before(anchor, result, rule_block)

    if not DERIVATION_MARKERS.search(result):
        derivation_block = (
            "### Answer Derivation\n"
            "Applying the selected rule to the target input "
            "(see validated branch above)."
        )
        result = _insert_before(FINAL_ANSWER_RE, result, derivation_block)

    final_answer = (
        answer.strip()
        if enforce_answer and answer
        else (_extract_last_boxed(result) or (answer.strip() if answer else None))
    )

    if final_answer and not FINAL_ANSWER_RE.search(result):
        result = result.rstrip() + f"\n\n### Final Answer\n$\\boxed{{{final_answer}}}$"
    elif final_answer and enforce_answer and f"\\boxed{{{final_answer}}}" not in result:
        result = result.rstrip() + f"\n\n### Final Answer\n$\\boxed{{{final_answer}}}$"

    if budget_chars is not None:
        result = _hard_cap(result, final_answer, budget_chars)

    return result


# ---------------------------------------------------------------------------
# Scoring — PoE logprobs (preferred) or structural heuristic (fallback)
# ---------------------------------------------------------------------------

def _completion_echo_logprobs_available(client: OpenAI, model: str) -> bool:
    """Returns True if the provider supports fixed-answer echo scoring."""
    try:
        resp = client.completions.create(
            model=model,
            prompt="hi",
            max_tokens=0,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )
        return resp.choices[0].logprobs is not None
    except Exception:
        return False


def _chat_logprobs_available(client: OpenAI, model: str) -> bool:
    """Returns True if the provider supports chat token logprobs."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with hi."}],
            max_tokens=4,
            temperature=0.0,
            logprobs=True,
        )
        return resp.choices[0].logprobs is not None
    except Exception:
        return False


def _detect_logprob_mode(client: OpenAI, model: str) -> str:
    if _completion_echo_logprobs_available(client, model):
        return "completion_echo"
    if _chat_logprobs_available(client, model):
        return "chat_generation"
    return "none"


def _poe_score(client: OpenAI, model: str, puzzle_prompt: str,
               branch: Branch, answer: str) -> float | None:
    """Score a fixed boxed answer using completion echo logprobs when available."""
    if not branch.is_valid:
        return float('-inf')

    if not answer:
        return None

    answer_text = f"$\\boxed{{{answer}}}$"
    context = (
        f"{puzzle_prompt}\n\n"
        f"Based on the following reasoning:\n{branch.body}\n\n"
        f"The answer is:\n{answer_text}"
    )
    answer_start = len(context) - len(answer_text)

    try:
        resp = client.completions.create(
            model=model,
            prompt=context,
            max_tokens=0,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )
        logprobs = resp.choices[0].logprobs
        if logprobs is None or logprobs.token_logprobs is None or logprobs.text_offset is None:
            return None

        answer_token_logprobs = [
            logprob
            for logprob, offset in zip(logprobs.token_logprobs, logprobs.text_offset)
            if offset >= answer_start and logprob is not None
        ]
        if not answer_token_logprobs:
            return None
        return sum(answer_token_logprobs) / len(answer_token_logprobs)
    except Exception:
        return None


def _poe_score_chat(client: OpenAI, model: str, puzzle_prompt: str,
                    branch: Branch, answer: str) -> float | None:
    """
    Chat-native scoring fallback.

    This is not fixed-answer scoring; it measures whether the model can
    regenerate the known boxed answer from the branch under greedy decoding,
    then uses the generated token logprobs as the branch score.
    """
    if not branch.is_valid:
        return float('-inf')

    if not answer:
        return None

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CHAT_SCORE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{puzzle_prompt}\n\n"
                        f"Candidate reasoning branch:\n{branch.body}\n\n"
                        "Return only the final boxed answer supported by this branch."
                    ),
                },
            ],
            max_tokens=max(32, min(128, len(answer) * 2 + 16)),
            temperature=0.0,
            logprobs=True,
        )
        message = resp.choices[0].message.content or ""
        predicted = _extract_last_boxed(message)
        logprobs = resp.choices[0].logprobs
        tokens = logprobs.content if logprobs else None
        if not tokens:
            return None
        if not _answers_match(predicted, answer):
            return float("-inf")
        return sum(token.logprob for token in tokens) / len(tokens)
    except Exception:
        return None


def _heuristic_score(branch: Branch, footer: str) -> float:
    """
    Structure-aware score when logprobs are unavailable.

    Priority (highest score = keep last):
      0   = winning VALID branch (referenced in footer's Selected Rule section)
     -1   = other VALID branches (explored but not selected)
     -inf = INVALID branches (explicitly failed)
    """
    if not branch.is_valid:
        return float('-inf')

    # Check if this branch's header keyword appears in the footer
    # (the Selected Rule section typically names the winning branch)
    branch_keyword = re.sub(r'#{2,3}\s+Branch\s+\d+[:\s]*', '', branch.header).strip()
    if branch_keyword and branch_keyword.lower() in footer.lower():
        return 0.0   # winning branch — keep
    return -1.0      # valid but not selected — can prune


# ---------------------------------------------------------------------------
# Derivation truncation (last-resort when branches alone aren't enough)
# ---------------------------------------------------------------------------

def _truncate_derivation(footer: str, budget_chars: int) -> str:
    """
    Shorten the derivation section within the footer to fit the budget.
    Always preserves the Selected Rule and Final Answer sections.
    """
    deriv_match = DERIVATION_MARKERS.search(footer)
    if not deriv_match:
        return footer

    answer_match = re.search(r'#{2,3}\s+Final Answer', footer, re.IGNORECASE)
    if not answer_match:
        return footer

    before_deriv  = footer[:deriv_match.start()]
    final_section = footer[answer_match.start():]
    available     = budget_chars - len(before_deriv) - len(final_section) - 100  # 100 char margin

    deriv_body    = footer[deriv_match.start():answer_match.start()]
    if available > 200:
        # Keep as much of the derivation as fits, cut at last newline
        truncated = deriv_body[:available]
        cut = truncated.rfind('\n')
        if cut > 0:
            truncated = truncated[:cut]
        deriv_body = truncated.rstrip() + "\n*(derivation truncated for length)*\n\n"

    return before_deriv + deriv_body + final_section


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Cache logprob mode per (base_url, model) so we only probe once
_logprob_cache: dict[tuple, str] = {}


def prune_if_needed(
    client:        OpenAI,
    model:         str,
    trace:         str,
    puzzle_prompt: str,
    answer:        str,
    use_logprobs:  bool | None = None,
) -> tuple[str, bool, str]:
    """
    If trace fits within budget → return (trace, False, "none").
    If over budget → prune branches and return (pruned_trace, True, method_used).

    Pruning uses PoE logprob scoring when available, otherwise falls back
    to structure-aware heuristic scoring.

    Args:
        use_logprobs: If False, skip logprob scoring.
                      If True/None, probe supported logprob mode once (cached).
    """
    if len(trace) <= BUDGET_CHARS:
        return trace, False, "none"

    parsed = parse_trace(trace)

    if not parsed.branches:
        result = normalize_trace(
            trace,
            answer=answer,
            enforce_answer=bool(answer),
            budget_chars=BUDGET_CHARS,
        )
        return result, True, "hard cap"

    # Determine scoring method
    if use_logprobs is False:
        logprob_mode = "none"
    else:
        cache_key = (getattr(client, 'base_url', ''), model)
        if cache_key not in _logprob_cache:
            detected_mode = _detect_logprob_mode(client, model)
            if detected_mode != "none":
                _logprob_cache[cache_key] = detected_mode
            logprob_mode = detected_mode
        else:
            logprob_mode = _logprob_cache[cache_key]

    if logprob_mode == "completion_echo":
        scores_available = True
        method = "PoE answer echo logprobs"
        for branch in parsed.branches:
            score = _poe_score(client, model, puzzle_prompt, branch, answer)
            if score is None:
                scores_available = False
                break
            branch.score = score

        if not scores_available:
            method = "structural heuristic fallback"
            for branch in parsed.branches:
                branch.score = _heuristic_score(branch, parsed.footer)
    elif logprob_mode == "chat_generation":
        scores_available = True
        method = "chat answer generation logprobs"
        any_exact_match = False
        for branch in parsed.branches:
            score = _poe_score_chat(client, model, puzzle_prompt, branch, answer)
            if score is None:
                scores_available = False
                break
            if score != float("-inf"):
                any_exact_match = True
            branch.score = score

        if not scores_available or not any_exact_match:
            method = "structural heuristic fallback"
            for branch in parsed.branches:
                branch.score = _heuristic_score(branch, parsed.footer)
    else:
        method = "structural heuristic"
        for branch in parsed.branches:
            branch.score = _heuristic_score(branch, parsed.footer)

    # Sort ascending — lowest score pruned first
    parsed.branches.sort(key=lambda b: b.score)

    # Phase 1: prune branches
    while trace_chars(parsed) > BUDGET_CHARS and len(parsed.branches) > 1:
        parsed.branches.pop(0)

    # Phase 2: if still over budget (one branch left but footer is huge),
    # truncate the derivation section
    if trace_chars(parsed) > BUDGET_CHARS:
        remaining = BUDGET_CHARS - len(parsed.preamble) - sum(len(b.body) for b in parsed.branches)
        parsed.footer = _truncate_derivation(parsed.footer, max(remaining, 2000))

    # Phase 3: if STILL over budget (single long branch or massive body),
    # hard-truncate the last branch body to fit
    if trace_chars(parsed) > BUDGET_CHARS and parsed.branches:
        fixed_len = len(parsed.preamble) + len(parsed.footer) + 200  # margin
        avail = BUDGET_CHARS - fixed_len
        if avail > 500:
            branch = parsed.branches[-1]
            truncated = branch.body[:avail]
            cut = truncated.rfind('\n')
            if cut > 200:
                truncated = truncated[:cut]
            branch.body = truncated.rstrip() + "\n\n*(branch truncated for length)*\n"

    result = normalize_trace(
        reconstruct(parsed),
        answer=answer,
        enforce_answer=bool(answer),
        budget_chars=BUDGET_CHARS,
    )

    return result, True, method
