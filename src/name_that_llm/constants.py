"""Competition label definitions and deterministic single-token mappings."""

from __future__ import annotations

# Exact 18 valid model labels from https://www.kaggle.com/competitions/name-that-llm/overview
VALID_MODELS: tuple[str, ...] = (
    # OpenAI (5)
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    # Google (4)
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    # Anthropic (5)
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    # Qwen (1)
    "qwen3.8-max",
    # ZAI (2)
    "glm-5.3",
    "glm-5.3-flash",
    # Moonshot AI (1)
    "kimi-k3",
)

NUM_CLASSES: int = len(VALID_MODELS)
assert NUM_CLASSES == 18, f"Expected 18 valid models, got {NUM_CLASSES}"

# Single-character uppercase class codes ('A'..'R'), each guaranteed to be a single token
# in Gemma 4's 262K SentencePiece vocabulary. Using single tokens lets vLLM score all 18
# classes in ONE forward pass (max_tokens=1, logprobs=20).
CLASS_CODES: tuple[str, ...] = tuple(chr(ord("A") + i) for i in range(NUM_CLASSES))

MODEL_TO_IDX: dict[str, int] = {m: i for i, m in enumerate(VALID_MODELS)}
IDX_TO_MODEL: dict[int, str] = {i: m for i, m in enumerate(VALID_MODELS)}

MODEL_TO_CODE: dict[str, str] = {m: CLASS_CODES[i] for i, m in enumerate(VALID_MODELS)}
CODE_TO_MODEL: dict[str, str] = {CLASS_CODES[i]: m for i, m in enumerate(VALID_MODELS)}
CODE_TO_IDX: dict[str, int] = {CLASS_CODES[i]: i for i in range(NUM_CLASSES)}
