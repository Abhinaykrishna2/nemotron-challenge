"""
PoE-based branch pruner for Tree of Thought reasoning traces.

Only invoked when a generated trace exceeds the 8000-token inference budget.

Scoring strategy (in order of preference):
  1. PoE logprob scoring — log P(answer | prompt + branch) via LLM logprobs.
     Requires the provider to support logprobs (Cerebras does; OpenRouter/WandB does not).
  2. Structure-aware heuristic fallback (when logprobs unavailable):
       a. INVALID branches → score -inf  (remove first, they're explicitly wrong)
       b. VALID branches not referenced in footer → score -1  (remove next)
       c. VALID branch referenced in footer (the selected/winning branch) → score 0 (keep last)
       d. If still over budget → truncate derivation section, never branches or answer

From poe_architecture.txt:
  log P(s|p) = (1/m) * Σ_j log P̂_j(s|p) - log Z
  Each branch is a "view"; we keep the views that most support the final answer.
"""

import re
from dataclasses import dataclass, field
from openai import OpenAI

TOKEN_BUDGET  = 8000
CHARS_PER_TOK = 4
BUDGET_CHARS  = TOKEN_BUDGET * CHARS_PER_TOK   # 32 000 chars

DERIVATION_MARKERS = re.compile(
    r'(#{2,3}\s+Answer Derivation|#{2,3}\s+Derivation)', re.IGNORECASE
)


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
# Scoring — PoE logprobs (preferred) or structural heuristic (fallback)
# ---------------------------------------------------------------------------

def _logprobs_available(client: OpenAI, model: str) -> bool:
    """Quick probe: returns True if the provider supports logprobs."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "hi"}],
            max_tokens=1,
            logprobs=True,
            temperature=0.0,
        )
        return resp.choices[0].logprobs is not None
    except Exception:
        return False


def _poe_score(client: OpenAI, model: str, puzzle_prompt: str,
               branch: Branch, answer: str) -> float:
    """log P(answer | prompt + branch) via logprobs — real PoE scoring."""
    if not branch.is_valid:
        return float('-inf')
    context = (
        f"{puzzle_prompt}\n\n"
        f"Based on the following reasoning:\n{branch.body}\n\n"
        f"The answer is:"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user",      "content": context},
                {"role": "assistant", "content": f"$\\boxed{{{answer}}}$"},
            ],
            max_tokens=1,
            logprobs=True,
            temperature=0.0,
        )
        tokens = resp.choices[0].logprobs.content if resp.choices[0].logprobs else []
        if not tokens:
            return -999.0
        return sum(t.logprob for t in tokens) / len(tokens)
    except Exception:
        return -999.0


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

# Cache logprob support per (base_url, model) so we only probe once
_logprob_cache: dict[tuple, bool] = {}


def prune_if_needed(
    client:        OpenAI,
    model:         str,
    trace:         str,
    puzzle_prompt: str,
    answer:        str,
    use_logprobs:  bool | None = None,
) -> tuple[str, bool]:
    """
    If trace fits within budget → return (trace, False).
    If over budget → prune branches and return (pruned_trace, True).

    Pruning uses PoE logprob scoring when available, otherwise falls back
    to structure-aware heuristic scoring.

    Args:
        use_logprobs: If explicitly True/False, skip the logprob probe.
                      If None, probe the provider once (cached).
    """
    if len(trace) <= BUDGET_CHARS:
        return trace, False

    parsed = parse_trace(trace)

    if not parsed.branches:
        # No branch structure — truncate derivation or hard-cap
        if answer not in trace[-500:]:
            trace = trace[:BUDGET_CHARS].rstrip() + f"\n\n### Final Answer\n$\\boxed{{{answer}}}$"
        return trace[:BUDGET_CHARS + 200], True

    # Determine scoring method
    if use_logprobs is None:
        # Probe once per provider+model (legacy/standalone usage)
        cache_key = (getattr(client, 'base_url', ''), model)
        if cache_key not in _logprob_cache:
            _logprob_cache[cache_key] = _logprobs_available(client, model)
        use_logprobs = _logprob_cache[cache_key]

    # Score branches
    for branch in parsed.branches:
        if use_logprobs:
            branch.score = _poe_score(client, model, puzzle_prompt, branch, answer)
        else:
            branch.score = _heuristic_score(branch, parsed.footer)

    method = "PoE logprobs" if use_logprobs else "structural heuristic"

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

    # Ensure answer is in footer
    if answer not in parsed.footer:
        parsed.footer = parsed.footer.rstrip() + f"\n\n### Final Answer\n$\\boxed{{{answer}}}$"

    return reconstruct(parsed), True
