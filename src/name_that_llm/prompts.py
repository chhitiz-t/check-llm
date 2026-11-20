"""Deterministic Gemma 4 prompt formatting for 18-way authorship attribution.

Phase 2 upgrades:
1. Preserves exact Unicode codepoints (never normalizes `’` vs `'`, `“`/`”` vs `"`, `—` vs `-`),
   which strongly separate Anthropic/ZAI, OpenAI, Google, Qwen, and Moonshot families.
2. Injects a compact deterministic `<stylometry>` header inside the user message so Gemma 4 12B
   can attend jointly to exact surface typography counts AND deep semantic/syntactic phrasing
   (crucial for separating intra-family siblings like `gemini-3.7-flash` vs `3.8-flash` and
   `gpt-5.6-luna` vs `sol` vs `terra`).
3. Keeps `SYSTEM_PROMPT` + user preamble 100% static so vLLM's Automatic Prefix Caching
   (`enable_prefix_caching=True`) caches the ~235-token prefix ONCE per GPU across ~4,000 rows.
4. Disables Gemma 4 thinking mode (`enable_thinking=False`) so output token #0 is directly `A`..`R`.
"""

from __future__ import annotations

from typing import Any

from name_that_llm.constants import CLASS_CODES, CODE_TO_MODEL

_LEGEND_LINES = "\n".join(f"{code} = {CODE_TO_MODEL[code]}" for code in CLASS_CODES)

SYSTEM_PROMPT = (
    "You are an expert LLM stylometry and authorship attribution classifier.\n"
    "Given a short text passage and its surface typography telemetry, identify which of the "
    "18 candidate frontier language models wrote it.\n"
    "Weigh both exact punctuation/quote habits and deeper phrasing, rhythm, and sentence structure.\n"
    "Respond with ONLY the single uppercase letter (A through R) from the mapping below:\n"
    f"{_LEGEND_LINES}"
)

USER_PREAMBLE = (
    "Classify the authoring language model of the passage below.\n"
    "Output a single letter (A-R) and nothing else.\n\n"
)


def build_stylometry_tag(passage: str) -> str:
    """Compute fast, zero-dependency surface typography & sentence-rhythm telemetry."""
    words = passage.split()
    n_words = len(words)
    sents = [s.strip() for s in passage.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    n_sents = max(len(sents), 1)
    avg_sw = n_words / n_sents

    em_dash = passage.count("—")
    en_dash = passage.count("–")
    hyphen = passage.count("-")
    curly_apos = passage.count("’")
    straight_apos = passage.count("'")
    curly_quote = passage.count("“") + passage.count("”")
    straight_quote = passage.count('"')
    semicolon = passage.count(";")
    colon = passage.count(":")
    parens = passage.count("(") + passage.count(")")

    return (
        "<stylometry>"
        f"words={n_words} | sents={n_sents} | avg_sent_words={avg_sw:.1f} | "
        f"em_dash={em_dash} | en_dash={en_dash} | hyphen={hyphen} | "
        f"curly_apos={curly_apos} | straight_apos={straight_apos} | "
        f"curly_quote={curly_quote} | straight_quote={straight_quote} | "
        f"semicolon={semicolon} | colon={colon} | parens={parens}"
        "</stylometry>"
    )


def build_user_message(
    passage: str,
    max_chars: int = 2400,
    include_stylometry_header: bool = True,
) -> str:
    # Preserve exact Unicode characters; only trim outer whitespace
    cleaned = passage.strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]

    if include_stylometry_header:
        stylo = build_stylometry_tag(cleaned)
        return f"{USER_PREAMBLE}{stylo}\n<passage>\n{cleaned}\n</passage>"
    return f"{USER_PREAMBLE}<passage>\n{cleaned}\n</passage>"


def build_messages(
    passage: str,
    label_code: str | None = None,
    max_chars: int = 2400,
    include_stylometry_header: bool = True,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_message(
                passage,
                max_chars=max_chars,
                include_stylometry_header=include_stylometry_header,
            ),
        },
    ]
    if label_code is not None:
        messages.append({"role": "assistant", "content": label_code})
    return messages


def render_inference_prompt(
    tokenizer: Any | None,
    passage: str,
    max_chars: int = 2400,
    enable_thinking: bool = False,
    include_stylometry_header: bool = True,
) -> str:
    """Render the exact generation prompt ending right before the class code token."""
    messages = build_messages(
        passage=passage,
        label_code=None,
        max_chars=max_chars,
        include_stylometry_header=include_stylometry_header,
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    # Canonical Gemma 4 chat template fallback
    thought_prefix = "" if enable_thinking else "<|thought>\n<thought|>"
    return (
        f"<bos><|turn>system\n{messages[0]['content']}<turn|>\n"
        f"<|turn>user\n{messages[1]['content']}<turn|>\n"
        f"<|turn>model\n{thought_prefix}"
    )
