# Phase 1 Report — Data Validation, Stylometric EDA & CV Split

## 1. Dataset Integrity Summary
- **Training passages (`train.csv`):** `2,699` rows (0 nulls, 0 duplicate IDs)
- **Unique training prompts (`prompt_id`):** `150`
- **Candidate models represented:** `18 / 18`
- **Class imbalance ratio (`max / min`):** `1.01x` (`min=149`, `max=150`)
- **Passage word length (`train.csv`):** min `36`, median `101`, max `274` words
- **Passage char length (`train.csv`):** min `230`, median `670`, max `1900` chars
- **Interactive `test.csv` rows:** `3`

## 2. Leak-Free 5-Fold `StratifiedGroupKFold(prompt_id)` Summary
- **Zero prompt leakage verified:** Every `prompt_id` belongs to exactly 1 validation fold.

|   fold |   rows |   unique_prompts |   unique_models |   mean_words |
|-------:|-------:|-----------------:|----------------:|-------------:|
|      0 |    539 |               30 |              18 |        115.9 |
|      1 |    540 |               30 |              18 |        114.1 |
|      2 |    540 |               30 |              18 |        117.1 |
|      3 |    540 |               30 |              18 |        112.6 |
|      4 |    540 |               30 |              18 |        113.6 |

## 3. Per-Model Distribution, Balanced Loss Weights & Length Fingerprints

| Code   | Model                  |   Count |   Share (%) |   Loss Weight |   Prompts | Words (mean±std)   | Words [min, max]   |
|:-------|:-----------------------|--------:|------------:|--------------:|----------:|:-------------------|:-------------------|
| A      | gpt-5.5                |     150 |        5.56 |         1     |       150 | 110.3 ± 54.7       | [39, 245]          |
| B      | gpt-5.6-luna           |     150 |        5.56 |         1     |       150 | 115.4 ± 57.8       | [38, 246]          |
| C      | gpt-5.6-sol            |     150 |        5.56 |         1     |       150 | 118.9 ± 60.5       | [40, 261]          |
| D      | gpt-5.6-terra          |     150 |        5.56 |         1     |       150 | 113.3 ± 54.5       | [43, 239]          |
| E      | gpt-6-astra            |     150 |        5.56 |         1     |       150 | 117.8 ± 59.1       | [36, 256]          |
| F      | gemini-3.1-pro-preview |     150 |        5.56 |         1     |       150 | 119.3 ± 59.6       | [38, 262]          |
| G      | gemini-3.6-flash       |     150 |        5.56 |         1     |       150 | 111.2 ± 54.1       | [40, 238]          |
| H      | gemini-3.7-flash       |     150 |        5.56 |         1     |       150 | 117.1 ± 59.8       | [40, 252]          |
| I      | gemini-3.8-flash       |     150 |        5.56 |         1     |       150 | 100.7 ± 52.8       | [41, 238]          |
| J      | claude-fable-5         |     149 |        5.52 |         1.006 |       149 | 112.5 ± 58.2       | [39, 252]          |
| K      | claude-opus-4-6        |     150 |        5.56 |         1     |       150 | 115.1 ± 60.9       | [42, 274]          |
| L      | claude-opus-4-8        |     150 |        5.56 |         1     |       150 | 123.6 ± 61.6       | [42, 251]          |
| M      | claude-opus-5          |     150 |        5.56 |         1     |       150 | 123.4 ± 62.3       | [39, 257]          |
| N      | claude-sonnet-5        |     150 |        5.56 |         1     |       150 | 120.0 ± 64.3       | [39, 267]          |
| O      | qwen3.8-max            |     150 |        5.56 |         1     |       150 | 108.3 ± 56.8       | [39, 265]          |
| P      | glm-5.3                |     150 |        5.56 |         1     |       150 | 110.3 ± 56.8       | [36, 244]          |
| Q      | glm-5.3-flash          |     150 |        5.56 |         1     |       150 | 119.2 ± 58.2       | [37, 256]          |
| R      | kimi-k3                |     150 |        5.56 |         1     |       150 | 107.4 ± 53.7       | [39, 239]          |

## 4. Key Stylometric & Unicode Typography Fingerprints
- **Critical finding:** All 2,699 passages are single paragraphs (`0%` newlines, `0%` `**bold**`), and **Unicode punctuation separates provider families sharply** — we must NEVER apply NFKC/ASCII quote normalization before Gemma 4 tokenization!
  - **Anthropic (`claude-*`) & ZAI (`glm-5.3*`):** `63%–83%` em-dashes (`—`), `0%` curly apostrophes (`’`), `57%–73%` straight ASCII apostrophes (`'`).
  - **Google (`gemini-*`):** `0%–0.7%` em-dashes (`—`), low colon usage (`17%–23%`).
  - **OpenAI (`gpt-5.6-luna` / `gpt-5.6-sol`):** `51%–53%` curly apostrophes (`’`), `<1%` straight apostrophes (`'`).
  - **Moonshot (`kimi-k3`):** Mixes both curly (`32%`) and straight (`23%`) apostrophes + `52%` em-dashes + `77%` colons.

| Model                  |   Em-dash `—` (%) |   Curly `’` (%) |   Straight `'` (%) |   Curly `“”` (%) |   Semicolon `;` (%) |   Colon `:` (%) |
|:-----------------------|------------------:|----------------:|-------------------:|-----------------:|--------------------:|----------------:|
| qwen3.8-max            |               3.3 |            28.7 |               12   |              0.7 |                39.3 |            47.3 |
| claude-sonnet-5        |              73.3 |             0   |               72.7 |              0   |                23.3 |            50.7 |
| gemini-3.6-flash       |               0   |            19.3 |               21.3 |              0   |                22.7 |            20   |
| gpt-5.6-terra          |              22.7 |            46.7 |                5.3 |              8   |                52   |            73.3 |
| glm-5.3-flash          |              79.3 |             0   |               67.3 |              0   |                48   |            65.3 |
| gemini-3.1-pro-preview |               0   |             6   |               10.7 |              0   |                19.3 |            16.7 |
| gemini-3.8-flash       |               0   |            24.7 |               11.3 |              2   |                28   |            18   |
| claude-opus-4-6        |              75.3 |             0   |               57.3 |              0   |                15.3 |            42   |
| gpt-5.6-luna           |              36.7 |            51.3 |                0.7 |             10.7 |                42.7 |            70.7 |
| claude-opus-4-8        |              62.7 |             0   |               60   |              0   |                24   |            49.3 |
| gemini-3.7-flash       |               0.7 |            30.7 |               14   |              0   |                23.3 |            22.7 |
| claude-fable-5         |              83.2 |             0   |               59.7 |              0   |                33.6 |            59.1 |
| claude-opus-5          |              81.3 |             0   |               63.3 |              0   |                54   |            69.3 |
| kimi-k3                |              52   |            32   |               23.3 |              9.3 |                61.3 |            77.3 |
| gpt-5.5                |              13.3 |            42   |               11.3 |              9.3 |                60.7 |            65.3 |
| gpt-6-astra            |               8.7 |            44   |                8.7 |              7.3 |                52.7 |            65.3 |
| glm-5.3                |              82   |             0   |               58   |              0   |                34   |            62   |
| gpt-5.6-sol            |              27.3 |            52.7 |                0   |             14.7 |                62   |            64   |

## 5. Phase 1 Leak-Free 5-Fold Baseline Benchmark (CPU TF-IDF + Stylometry)
- **5-Fold OOF Balanced Accuracy:** `0.34263` (`34.26%` vs `5.56%` random guess — **6.17x** random)
- **5-Fold OOF Multi-Class LogLoss:** `1.95093`

### Per-Model Baseline Recall
| model                  |   support |   predicted |   recall |   precision |
|:-----------------------|----------:|------------:|---------:|------------:|
| gpt-5.5                |       150 |         174 |   0.38   |      0.3276 |
| gpt-5.6-luna           |       150 |         138 |   0.28   |      0.3043 |
| gpt-5.6-sol            |       150 |          95 |   0.16   |      0.2526 |
| gpt-5.6-terra          |       150 |         101 |   0.16   |      0.2376 |
| gpt-6-astra            |       150 |         207 |   0.58   |      0.4203 |
| gemini-3.1-pro-preview |       150 |         203 |   0.7133 |      0.5271 |
| gemini-3.6-flash       |       150 |         136 |   0.3467 |      0.3824 |
| gemini-3.7-flash       |       150 |         156 |   0.3867 |      0.3718 |
| gemini-3.8-flash       |       150 |         176 |   0.32   |      0.2727 |
| claude-fable-5         |       149 |          75 |   0.1074 |      0.2133 |
| claude-opus-4-6        |       150 |         197 |   0.5067 |      0.3858 |
| claude-opus-4-8        |       150 |         109 |   0.2467 |      0.3394 |
| claude-opus-5          |       150 |         208 |   0.56   |      0.4038 |
| claude-sonnet-5        |       150 |         177 |   0.3867 |      0.3277 |
| qwen3.8-max            |       150 |         184 |   0.44   |      0.3587 |
| glm-5.3                |       150 |          99 |   0.1267 |      0.1919 |
| glm-5.3-flash          |       150 |         139 |   0.24   |      0.259  |
| kimi-k3                |       150 |         125 |   0.2267 |      0.272  |

### Top 10 Confused Model Pairs (Guidance for Phase 2 Gemma 4 12B SFT)
| true_model       | pred_model             |   count |
|:-----------------|:-----------------------|--------:|
| gemini-3.7-flash | gemini-3.8-flash       |      40 |
| gemini-3.8-flash | gemini-3.7-flash       |      40 |
| glm-5.3          | glm-5.3-flash          |      31 |
| glm-5.3          | claude-opus-5          |      31 |
| gemini-3.6-flash | gemini-3.7-flash       |      31 |
| gemini-3.6-flash | gemini-3.1-pro-preview |      27 |
| kimi-k3          | gpt-5.5                |      26 |
| gemini-3.6-flash | gemini-3.8-flash       |      26 |
| gpt-5.6-terra    | gpt-5.6-luna           |      24 |
| gpt-5.6-terra    | gpt-5.5                |      24 |
