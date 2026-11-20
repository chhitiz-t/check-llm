# Phase 2 Report — Gemma 4 12B Prompt Conditioning, Token Audit & VRAM Optimization

## 1. Prompt & Unicode Integrity Audit (`2,699` Train + `3` Test Passages)
- **Tokenization measurement mode:** Canonical Gemma 4 Template + 262K SentencePiece Subword Estimator
- **Thinking mode disabled (`enable_thinking=False`):** Verified (`<|thought>\n<thought|>` suffix present on `100%` of prompts so generated token `#0` is `A`..`R`).
- **Stylometry header (`<stylometry>`):** Enabled (`words`, `sents`, `avg_sent_words`, `em_dash`, `en_dash`, `hyphen`, `curly_apos`, `straight_apos`, `curly_quote`, `straight_quote`, `semicolon`, `colon`, `parens`).
- **Unicode preservation:** `100%` lossless across `—`, `–`, `’`, `'`, `“`, `”` on all `2,699` passages.

## 2. vLLM Automatic Prefix Caching & Sequence Length Budget
- **Exact shared static prefix across all prompts:** `874` chars (`~244` tokens cached ONCE per T4 GPU via `enable_prefix_caching=True` — **~54.2%** of median prompt tokens skipped during prefill!).
- **Total prompt + target token lengths (`N = 2,699`):**
  - Min: `362` tokens (`1,353` chars)
  - Median (p50): `450` tokens (`1,793` chars)
  - p95: `613` tokens
  - p99: `650` tokens
  - Max: `688` tokens (`3,024` chars)
- **Truncation under `qlora.max_seq_length = 768`:** `0 / 2699` (`0.00%`)
- **Truncation under `vllm.max_model_len = 1024`:** `0 / 2699` (`0.00%`)

## 3. Phase 3 Training Step & VRAM Optimizations Verified
- **Effective batch size:** `16` (`per_device_train_batch_size=2` × `gradient_accumulation_steps=8`)
- **Optimizer steps per fold (`3` epochs):** `405` steps (`135` steps/epoch)
- **`262K` Vocab `lm_head` tensor (`logits_to_keep=2` + left-padding):** Reduced forward logit buffer at p95 length from **`613.0 MB`** (`[2, 613, 262144]`) down to **`2.0 MB`** (`[2, 2, 262144]`) — saving ~1.5–2.0 GB of autograd peak memory while preserving Gemma 4's native `30 * tanh(z / 30)` logit soft-capping.

## 4. Concrete Rendered Prompt Example (`train.csv` Row 0)
- **Ground-Truth Model:** `qwen3.8-max` $\rightarrow$ Single-Token Target: **`O`**

```text
<bos><|turn>system
You are an expert LLM stylometry and authorship attribution classifier.
Given a short text passage and its surface typography telemetry, identify which of the 18 candidate frontier language models wrote it.
Weigh both exact punctuation/quote habits and deeper phrasing, rhythm, and sentence structure.
Respond with ONLY the single uppercase letter (A through R) from the mapping below:
A = gpt-5.5
B = gpt-5.6-luna
C = gpt-5.6-sol
D = gpt-5.6-terra
E = gpt-6-astra
F = gemini-3.1-pro-preview
G = gemini-3.6-flash
H = gemini-3.7-flash
I = gemini-3.8-flash
J = claude-fable-5
K = claude-opus-4-6
L = claude-opus-4-8
M = claude-opus-5
N = claude-sonnet-5
O = qwen3.8-max
P = glm-5.3
Q = glm-5.3-flash
R = kimi-k3<turn|>
<|turn>user
Classify the authoring language model of the passage below.
Output a single letter (A-R) and nothing else.

<stylometry>words=192 | sents=8 | avg_sent_words=24.0 | em_dash=0 | en_dash=0 | hyphen=0 | curly_apos=1 | straight_apos=0 | curly_quote=0 | straight_quote=0 | semicolon=2 | colon=1 | parens=0</stylometry>
<passage>
When a language dies, a particular astronomy dims: not only names for stars, but whole grammars of attention that taught speakers how to read the night. Some tongues map the sky by kinship, binding the Milky Way to ancestors, rivers, or the migration paths of birds; others encode lunar phases as verbs, so time is not a line but a recurring breath. A word can carry the exact color of dawn on snow, the etiquette of speaking to thunder, or the mathematics woven into knot calendars and oral star charts. With extinction, those relations collapse into silence. No dictionary can resurrect the sky those speakers inhabited, nor the quiet ethics it taught. We lose metaphors that made the universe intimate, ethical, and alive, cosmologies in which darkness is a listener, comets are warnings carried by relatives, and constellations are not fixed pictures but seasonal responsibilities. The loss is not merely lexical; it is the erasure of possible human answers to wonder. What remains is a smaller sky, seen through fewer eyes, and a future that can no longer hear one of humanity’s original ways of saying where we stand under distant stars.
</passage><turn|>
<|turn>model
<|thought>
<thought|>O<turn|>
```
