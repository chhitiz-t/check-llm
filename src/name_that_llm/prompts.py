"""Deterministic Gemma 4 prompt formatting for 18-way authorship attribution.

Design notes for vLLM + Gemma 4 12B:
1. All prompts share an identical static system instruction + code table at the front,
   allowing vLLM's Automatic Prefix Caching (`enable_prefix_caching=True`) to compute
   the instruction KV cache ONCE per GPU and reuse it across all ~4,000 test passages.
2. Gemma 4's thinking mode is explicitly disabled (`enable_thinking=False`) so token #0
   of the assistant response is directly the single-token class code ('A'..'R').
"""

from __future__ import annotations

from typing import Any

from name_that_llm.constants import CLASS_CODES, CODE_TO_MODEL

_LEGEND_LINES = "\n".join(f"{code} = {CODE_TO_MODEL[code]}" for code in CLASS_CODES)

SYSTEM_PROMPT = (
    "You are an expert LLM stylometry and authorship attribution classifier.\n"
    "Given a short text passage, identify which of the 18 candidate frontier language models wrote it.\n"
    "Focus on stylistic fingerprints: vocabulary, cadence, sentence structure, punctuation, and phrasing.\n"
    "Respond with ONLY the single uppercase letter (A through R) from the mapping below:\n"
    f"{_LEGEND_LINES}"
)


def build_user_message(passage: str, max_chars: int = 2400) -> str:
    cleaned = passage.strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return (
        "Classify the authoring language model of the passage between <passage> and </passage>.\n"
        "Output a single letter (A-R) and nothing else.\n\n"
        f"<passage>\n{cleaned}\n</passage>"
    )


def build_messages(passage: str, label_code: str | None = None, max_chars: int = 2400) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(passage, max_chars=max_chars)},
    ]
    if label_code is not None:
        messages.append({"role": "model" if False else "assistant", "content": label_code})
    return messages


def render_inference_prompt(
    tokenizer: Any | None,
    passage: str,
    max_chars: int = 2400,
    enable_thinking: bool = False,
) -> str:
    """Render the exact generation prompt ending right before the class code token."""
    messages = build_messages(passage=passage, label_code=None, max_chars=max_chars)
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

    # Fallback canonical Gemma 4 template when running without a HF tokenizer loaded
    thought_prefix = "" if enable_thinking else "<|thought>\n<thought|>"
    return (
        f"<bos><|turn>system\n{messages[0]['content']}<turn|>\n"
        f"<|turn>user\n{messages[1]['content']}<turn|>\n"
        f"<|turn>model\n{thought_prefix}"
    )
