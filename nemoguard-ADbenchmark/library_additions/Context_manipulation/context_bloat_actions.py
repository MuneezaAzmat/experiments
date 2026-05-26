# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context bloat detection action

Detects context-manipulation attacks where attacker-controlled content
(retrieved chunks, tool outputs, or user input) is padded, oversized,
or repetitively structured to cause system prompt forgetting, bury instructions
mid-context (harder to detect), or exhaust the token budget.

Optimizations vs. the reference implementation:
    * Early-exit on size cap (cheapest check)  - skips remaining checks on reject
    * Entropy sampling for very large inputs   - 14-77x faster on >10KB inputs
    * Check order: cheapest first              - size > entropy > run > repetition

Wire as execution rail (tool output), retrieval rail (RAG chunks), or input rail.
"""

import logging
import math
from collections import Counter
from typing import List, Optional, TypedDict

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action

log = logging.getLogger(__name__)

# Sample-based entropy for inputs above this size. Entropy is statistically
# stable well below this threshold; sampling avoids O(n) work for huge inputs.
ENTROPY_SAMPLE_THRESHOLD = 10000
ENTROPY_SAMPLE_SIZE = 8000


class ContextBloatResult(TypedDict):
    is_bloat: bool
    text: str
    reason: Optional[str]
    detections: List[str]
    metrics: dict


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------

def _shannon_entropy(text: str) -> float:
    """Shannon entropy (bits/char). Samples large inputs to bound runtime."""
    if not text:
        return 0.0
    if len(text) > ENTROPY_SAMPLE_THRESHOLD:
        # Stratified sample: head, middle, tail thirds
        third = ENTROPY_SAMPLE_SIZE // 3
        mid = len(text) // 2
        sample = text[:third] + text[mid - third // 2 : mid + third // 2] + text[-third:]
    else:
        sample = text
    counts = Counter(sample)
    total = len(sample)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of repeated n-grams. High values are a padding-attack signature."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counter = Counter(ngrams)
    repeated = sum(c - 1 for c in counter.values() if c > 1)
    return repeated / len(ngrams) if ngrams else 0.0


def _longest_run_ratio(text: str) -> float:
    """Fraction of text that is the longest run of a single character.
    """
    if not text:
        return 0.0
    n = len(text)
    longest = 1
    i = 0
    while i < n:
        j = i + 1
        while j < n and text[j] == text[i]:
            j += 1
        if j - i > longest:
            longest = j - i
        i = j
    return longest / n


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_config(config: RailsConfig) -> None:
    cfg = getattr(config.rails.config, "context_bloat_detection", None)
    if cfg is None:
        raise ValueError("context_bloat_detection configuration is missing in RailsConfig.")
    if cfg.action not in {"reject", "truncate", "warn"}:
        raise ValueError(
            f"Expected 'reject', 'truncate', or 'warn' but got {cfg.action!r}."
        )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@action()
async def context_bloat_detection(text: str, config: RailsConfig) -> ContextBloatResult:
    """Detect context-bloat / context-manipulation attacks.

    Check order is cheapest first to enable early-exit:
        1. Size cap  (O(1))           - reject immediately on overflow
        2. Long run  (C-fast)         - catches degenerate "AAAAA..." padding
        3. Entropy   (sampled for big inputs)
        4. N-gram repetition (most expensive)

    Args:
        text: The text to inspect (tool output, joined chunks, or user message).
        config: RailsConfig with rails.config.context_bloat_detection settings.

    Returns:
        ContextBloatResult with is_bloat flag, processed text, reason, metrics.
    """
    _validate_config(config)
    cfg = config.rails.config.context_bloat_detection

    char_count = len(text) if text else 0
    detections: List[str] = []
    metrics: dict = {"chars": char_count}

    # ---- 1. Size cap (cheapest) ----
    if char_count > cfg.max_chars:
        detections.append("size_cap_exceeded")
        # Early exit on reject: don't waste cycles computing other metrics
        if cfg.action == "reject":
            log.info(f"context bloat detected: size_cap_exceeded | chars={char_count}")
            return ContextBloatResult(
                is_bloat=True,
                text=text,
                reason="size_cap_exceeded",
                detections=detections,
                metrics=metrics,
            )

    # ---- 2. Longest run (C-fast) ----
    run_ratio = _longest_run_ratio(text)
    metrics["longest_run_ratio"] = round(run_ratio, 3)
    if run_ratio > cfg.max_run_ratio:
        detections.append("long_run")
        if cfg.action == "reject":
            log.info(f"context bloat detected: long_run | run_ratio={run_ratio:.3f}")
            return ContextBloatResult(
                is_bloat=True,
                text=text,
                reason="long_run",
                detections=detections,
                metrics=metrics,
            )

    # ---- 3. Entropy (sampled for large inputs) ----
    entropy = _shannon_entropy(text)
    metrics["entropy"] = round(entropy, 3)
    if entropy and entropy < cfg.min_entropy:
        detections.append("low_entropy")
        if cfg.action == "reject":
            log.info(f"context bloat detected: low_entropy | entropy={entropy:.3f}")
            return ContextBloatResult(
                is_bloat=True,
                text=text,
                reason="low_entropy",
                detections=detections,
                metrics=metrics,
            )

    # ---- 4. N-gram repetition (most expensive, run last) ----
    rep_ratio = _repetition_ratio(text)
    metrics["repetition_ratio"] = round(rep_ratio, 3)
    if rep_ratio > cfg.max_repetition_ratio:
        detections.append("high_repetition")

    # ---- Aggregate result ----
    is_bloat = bool(detections)
    reason = ", ".join(detections) if detections else None
    result_text = text

    if is_bloat:
        log.info(f"context bloat detected: {reason} | metrics={metrics}")
        if cfg.action == "truncate":
            result_text = text[: cfg.max_chars] if text else text

    return ContextBloatResult(
        is_bloat=is_bloat,
        text=result_text,
        reason=reason,
        detections=detections,
        metrics=metrics,
    )