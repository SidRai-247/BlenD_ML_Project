# BLEnD + Region-Aware RAG — Project Guide & Plan

Working doc. Last updated 2026-09-30.

---

## 1. What BLEnD is (the part we use)

BLEnD asks the **same 500 everyday-culture questions** about 16 countries, e.g.
*"What is a common school cafeteria food in {country}?"*. Five native annotators per
country answered each question; their answers (with vote counts) are the gold labels.

We only use the **Short-Answer Questions (SAQ)** track, for 5 countries:

| Country | Local language | Questions | Scored* | Languages we run | Resource level |
|---|---|---|---|---|---|
| US | English | 500 | 461 | en | high |
| Spain | Spanish | 500 | 469 | es + en | high |
| South Korea | Korean | 500 | 475 | ko + en | mid |
| Azerbaijan | Azerbaijani | 500 | 469 | az + en | low |
| Ethiopia | Amharic | 500 | 472 | am + en | low |

\*The official scorer skips a question when ≥3 annotators said "no answer / not
applicable" or ≥5 said "I don't know". Accuracy is computed over the **Scored** column.

Every country has the same 6 categories: Food (105), Sport (88), Holidays/Celebration/Leisure (92),
Education (84), Work life (68), Family (63).

### What's in the repo (`BLEnD/`)

| Path | Contents | Do we use it? |
|---|---|---|
| `data/questions/{C}_questions.csv` | ID, Topic, `Question` (**local language**), `Translation` (**English**) | Yes. The column names are swapped relative to what you'd expect |
| `data/annotations/{C}_data.json` | gold answers per question ID: local + English variants, vote counts, "idk" counts | Yes, for scoring only |
| `data/prompts/{C}_prompts.csv` | prompt templates; the paper used `inst-4` and `pers-3` | Yes |
| `model_inference.py`, `utils.py` | the authors' inference code (API models + HF) | **No**: broken as released (typos, pulls prompts from Google Sheets). We write our own |
| `evaluation/exact_match.py` | the official scorer `soft_exact_match` | **Yes, unchanged**, so our numbers compare to the paper |
| `evaluation/evaluate.py` | wrapper around the scorer | No: crashes on English runs; we call `soft_exact_match` directly |
| `evaluation/mc_data/`, `data_SemEval/` | multiple-choice track, SemEval extra countries | No (MCQ is a stretch goal) |

### How a model is evaluated (the paper's protocol)

1. **Two prompts per question** (greedy decoding, temperature 0):
   - `inst-4`: *"Read the following question and provide a single answer without any explanations.\n\nQuestion: {q}\nAnswer:"*
   - `pers-3`: *"You are a person from {country} who is trying to explain your country's culture to a foreigner. Answer the following question, providing a single answer without any explanations.\n\n{q}"*
   The prompt is written in the same language as the question (a Korean question gets the Korean template).
2. **Scoring (soft exact match).** A response is correct if **any** annotator answer appears inside it after
   lemmatising/stemming both sides (spaCy for English, Spark-NLP for Spanish/Amharic, Okt for Korean,
   a stemmer for Azerbaijani). English answer variants are always accepted too.
   Example, Ethiopia *"common school cafeteria food?"*, gold = {biscuits, samosa, bread}:
   - response "Bread." → correct (1)
   - response "Injera" → wrong (0)
3. **Country score** = % correct over scored questions, **averaged over the two prompts**.
   Reported separately for the local-language and English versions.

Why this matters for us: long, chatty answers can get lucky (more words, more chances to match) and
answers in the wrong script can miss. We keep "single answer" prompts and cap output length.

### Paper reference numbers (SAQ, same protocol) — what we compare against

Local language (Table 10):

| Model | US | ES | KR | AZ | ET |
|---|---|---|---|---|---|
| GPT-4 | 83.2 | 79.0 | 81.0 | 62.1 | 25.9 |
| Llama-3.1-70B | 84.9 | 75.4 | 65.3 | 59.5 | 17.6 |
| Qwen1.5-14B | 78.7 | 56.8 | 52.2 | 34.0 | 3.3 |
| Aya-101 | 53.4 | 45.8 | 32.8 | 35.8 | 17.8 |

English version of the same questions (Table 11):

| Model | ES | KR | AZ | ET |
|---|---|---|---|---|
| GPT-4 | 67.9 | 69.7 | 64.6 | 46.0 |
| Llama-3.1-70B | 59.4 | 62.1 | 59.5 | 44.6 |
| Qwen1.5-14B | 55.1 | 54.5 | 51.9 | 39.7 |
| Aya-101 | 35.7 | 30.7 | 31.9 | 26.4 |

The pattern we want to move: US/Spain score high and Ethiopia low. For low-resource languages, English beats the local language.

---

## 2. Model choice (decided)

**Hardware we actually have:** the institute PC has an RTX 3060 (12 GB), ~2.6 GB free disk, and a firewall
that blocks model downloads. So the mid-term uses **small 4-bit models served by the PC's existing Ollama**.
No Python environment is needed; our script just calls Ollama's local API.

| Short name | Model | Params | Format | License | Source |
|---|---|---|---|---|---|
| `gemma3-4b` | `gemma3:4b` (Google Gemma 3 4B-it) | 4.3B | Q4_K_M | Gemma terms | already on the PC |
| `mistral-7b` | `mistral:latest` (Mistral-7B-Instruct v0.3) | 7.2B | Q4_0 | Apache-2.0 | already on the PC |
| `qwen2.5-3b` | `qwen2.5:3b` (Qwen2.5-3B-Instruct) | 3.1B | Q4_K_M | Qwen research | imported 2026-09-30 from `Qwen/Qwen2.5-3B-Instruct-GGUF` |

Measured speed: ~0.1–0.3 s per answer once loaded, so a 9k-prompt run takes well under an hour per model.

**Deviation from the SOP, stated in the report:** the SOP planned Qwen3-8B, Aya-101, GPT-OSS-20B, Gemma 2
and Llama 3.1 on a 32 GB GPU. Those don't fit this machine (12 GB GPU, disk, blocked downloads;
Ollama 0.6.3 can't run Qwen3). We present the 3–7B 4-bit models as the *resource-constrained* setting,
which is also where RAG should help most. The larger SOP models can be added on a stronger machine;
the runner is written to allow more backends.

---

## 3. Our method

### Knowledge base (done ✅)
`KB_Articles/{Country}_KB*.jsonl`, short English "evidence cards" (`text_en`) from non-BLEnD sources
(UNESCO, national statistics offices, ministries, tourism boards, LoC…). This avoids contamination.

| Country | KB v1 cards | KB v2 additions (2026-10-05) | Total used by RAG v2 |
|---|---|---|---|
| US | 150 (25 per category) | — | 150 |
| Spain | 150 (25 per category) | — | 150 |
| South Korea | 130 (sports/education thin) | — | 130 |
| Azerbaijan | 150 (cards very short, ~14 words) | 146 | 296 |
| Ethiopia | 140 (partial) | 149 | 289 |

**Decision: English-only KB.** We index `text_en` (+ keywords) only.

**KB v2 (question-driven extension, Azerbaijan + Ethiopia).** KB v1 is institutional and niche, and it
contains a gold answer for only 10–17% of questions. KB v2 adds mainstream everyday facts, one or more
cards per BLEnD *question type* ("The most common X in Ethiopia is Y …"), with local-language terms in
brackets. Sources: labour law, Wikipedia (cuisine, education, holidays, sport, culture, economy) and
news. The cards live in `rag_pipeline/kb_v2/{ethiopia,azerbaijan}_cards.py`, and
`python rag_pipeline/kb_v2/build_kb_v2.py` writes `KB_Articles/{Country}_KB_v2_additions.jsonl`. Each
card stores `written_for` (the question IDs whose template motivated it). That field is used **only**
for the "leave own cards out" check, never at retrieval time. Gold answers were not consulted while
writing, but the cards were written after the RAG v1 error analysis, so KB v2 is question-aware. About a
third of the cards are `source_quality = high` (checked against a source); the rest are `medium`.

### Pipeline

```
                 ┌──────────────── BASELINE ────────────────┐
question (local  │  prompt(inst-4 / pers-3, same language)  │──► LLM ──► answer ──► official scorer
 or English) ────┤                                          │
                 └───────────────── RAG ────────────────────┘
                   1. retrieval query = English version of the question
                      (for a local-language question it is used ONLY for search)
                   2. search in THIS country's KB only
                      v1: 0.5·BM25 + 0.5·bge-m3 → always top-3
                      v2: bge-m3 top-20 → cross-encoder rerank → keep ≤3 cards with score ≥ τ=0.45
                   3. v2 gate: 0 cards → baseline prompt → reuse the baseline answer
                   4. prompt = [context instruction + cards] + original prompt in the ORIGINAL language
                   5. LLM ──► answer ──► official scorer
```

Key rule: **translation is used only for searching.** The LLM always gets the original question in its
original language, so the comparison with the baseline stays clean.

**Decision (2026-10-04): the KB is treated as English-only, and the search query is always BLEnD's own English
version of the question** (`Translation` column, a human "oracle" translation), for simplicity. This holds for
the whole project, not just the mid-term. Machine-translated queries (LLM or NLLB) are an optional
extra for the final report, only if time allows.

### RAG v1 (Always-RAG): exact design (decided 2026-10-04)

Follows SOP §III-A: region-aware **hybrid BM25 + dense** retrieval restricted to the target country's KB,
then grounded generation with the same prompts. All choices are fixed **before** looking at RAG results.
Nothing is tuned on the test questions.

| Part | Choice | Why |
|---|---|---|
| Documents | one card = one document: `text_en` + its `keywords` | KB cards are short (15–40 words), so no chunking is needed. Keywords add exact terms for BM25 |
| Index scope | **one index per country** (US 150, ES 150, KR 130, AZ 150, ET 140 cards) | "region-aware": a Spain question can only retrieve Spain cards |
| Query | the question's **English** text (BLEnD's own English version = oracle translation), used for **both** the local-language and the English setting of that question | KB is English-only. The same retrieved cards for both settings keep local vs English comparable |
| Lexical | **BM25Okapi** (`rank_bm25`, k1 = 1.5, b = 0.75). Tokens: lowercase, `\w+`, English stop-words removed (scikit-learn list) | SOP names `rank_bm25`. Standard defaults |
| Dense | **`BAAI/bge-m3`** (MIT, ungated, 568M params, 2.3 GB) dense vectors via `sentence-transformers`, L2-normalised, **cosine similarity** | Named in the SOP. Free and open. Multilingual, so the same model later serves PS-3 (local-language query → English KB) |
| Fusion | per query, min-max normalise both score lists over the country's cards, then **hybrid = 0.5·dense + 0.5·BM25** | Simple convex combination with equal weights, fixed a priori, a standard hybrid baseline. Unlike rank fusion (RRF), it keeps score magnitudes, which PS-2 needs |
| k | **top-3** cards per question | ~50–120 words of context. Small enough not to drown 3–7B models |
| Prompt | context block prepended to the **unchanged** baseline prompt (`inst-4` / `pers-3`, original language) | The only difference from the baseline is the context. That gives a clean ablation |
| Logged per question | top-3 card ids + their dense cosine, raw BM25 and hybrid scores; top-1 raw scores | Becomes the retrieval-confidence signal for the Adaptive-RAG gate (PS-2) and for error analysis |

**RAG prompt** (context in English, since the KB is English; the question part is exactly the baseline prompt):
```
Background information about {country} (may or may not be relevant):
[1] {card 1 text_en}
[2] {card 2 text_en}
[3] {card 3 text_en}

{baseline prompt: inst-4 or pers-3, in the question's language, with the question}
```

**Where it runs:** retrieval is done **once, offline, on the laptop**. That is 2,500 English queries (500 × 5 countries)
× ~150 cards per country, a few minutes on CPU. The result is saved to `results/retrieval/oracle_en_top3.jsonl`.
`build_jobs.py --condition rag` then writes the RAG prompts into `jobs/rag.jsonl`. The **PC side is unchanged**:
the same `pc.py push/start/pull` and `run_ollama.py`, 9,000 prompts per model, then `score.py --run rag`.

**Retrieval diagnostic (sanity only, never used to edit the KB or tune anything):** answer-hit@3, the % of
questions where any top-3 card contains one of the gold English answers. It tells us how often the KB
*could* help. It is reported next to the accuracy change.

**Planned sensitivity checks (final report, not mid-term):** k ∈ {1, 3, 5}; BM25-only and dense-only
vs hybrid. These are reported as ablations, and the main number stays the a-priori setting above.

**Result (2026-10-05):** RAG v1 lowers accuracy on all models (−8.9 to −14.3 points); see the report.

### RAG v2: exact design (decided and run 2026-10-05)

Why v1 failed (diagnosis, details in the report):
- With a gold-bearing card in the top 3, v1 helps a lot (Azerbaijani 26→51, Amharic 8→30). That happens
  for only 7–14% of questions, so for the rest the cards are noise.
- Models refuse ("the information is not provided", 7–20% of answers) or copy distractors from the cards.
- BM25 hurts: dense-only beats every fusion weight (e.g. ET hit@3 11.0 → 17.4).
- SemEval-2026 Task 7 (built on BLEnD) agrees: CultRAG found plain RAG ≤ the LLM-only baseline, with
  KB coverage the bottleneck; king001 (1st) used BLEnD's own QA pairs as the KB (≈ leakage, not copied).
  We take king001's "no context if nothing relevant" rule.

| Part | v1 | **v2** |
|---|---|---|
| KB | KB v1 | KB v1 + **KB v2 additions** (AZ, ET only) |
| Query | BLEnD English question | same |
| Candidates | 0.5·dense + 0.5·BM25, min-max per query | **α·dense + (1−α)·BM25, α = 1.0** (chosen on dev from {0.8, 1.0} by recall@20: 33.5 vs 34.0) → **top-20** |
| Reranker | — | **`cross-encoder/ms-marco-MiniLM-L-12-v2`**, (EN question, card) → relevance 0..1, max length 256. (`BAAI/bge-reranker-v2-m3` was tried: ~1 pair/s on the laptop CPU, ≈12 h, so it was dropped) |
| Cut-off | — | keep cards with relevance **≥ τ = 0.45**, at most **K = 3** → 0–3 cards per question |
| τ tuning | — | grid 0.05…0.95; maximise **F0.5** of "card mentions a gold answer" over the sent cards, on **dev countries US/Spain/South Korea only**; frozen for AZ/ET |
| Gate (Adaptive-RAG, PS-2) | — | 0 cards → prompt identical to baseline → **baseline answer reused** (greedy decoding), no LLM call |
| Prompt | "Background information about {country} (may or may not be relevant):" | "Background information about {country}. Use it only if it directly helps to answer the question; otherwise ignore it and answer from your own knowledge. Always give an answer and never say that the information is not provided." |
| Models / decoding / scorer | as baseline | same |

Retrieval stats (v2): the share of questions that get context is US 7%, ES 42%, KR 42%, AZ 99.8%, ET 99.8%.
Answer-hit for AZ/ET is 64.6% / 58.9% (v1: 9.8% / 11.0%). With each question's own `written_for` cards
removed, dense hit@3 is still AZ 35.2% / ET 37.9%.

**Result (2026-10-05), SEM-B 3-model mean, baseline → v1 → v2:**

| Setting | Baseline | v1 | v2 |
|---|---|---|---|
| US en | 74.9 | 53.5 | 73.2 |
| Spain es / en | 56.3 / 50.8 | 36.8 / 34.3 | 50.0 / 44.5 |
| Korea ko / en | 48.0 / 47.7 | 39.2 / 35.8 | 43.7 / 42.3 |
| **Azerbaijan az / en** | 23.6 / 47.6 | 18.0 / 30.6 | **53.3 / 65.4** |
| **Ethiopia am / en** | 7.9 / 34.1 | 12.3 / 23.7 | **28.0 / 48.0** |
| Per model (gemma / mistral / qwen) | 43.4 / 50.9 / 36.1 | 29.1 / 38.4 / 27.2 | **51.0 / 55.1 / 43.3** |
| High − low resource gap | 27.2 | 18.7 | **2.0** |

Every AZ/ET setting improves for every model (+8.6 to +36.9). High-resource countries (KB v1 only) lose
1.7–6.4 points. Full table: `results/comparison_baseline_rag_ragv2.csv`.

**Run commands (v2):**
```bash
python rag_pipeline/kb_v2/build_kb_v2.py                       # KB v2 jsonl files
python rag_pipeline/retrieve_v2.py                             # → results/retrieval/v2_rerank_top3.jsonl (cached per country)
python rag_pipeline/build_jobs.py --run rag_v2 --pc-run rag_v2_pc --prompt-version v2 \
    --retrieval results/retrieval/v2_rerank_top3.jsonl --countries Azerbaijan Ethiopia
python rag_pipeline/pc.py push/start/status/pull --run rag_v2_pc   # only prompts that have cards
python rag_pipeline/merge_gated.py --run rag_v2 --pc-run rag_v2_pc # + baseline answers for 0-card questions
python rag_pipeline/score.py --run rag_v2                         # Spanish/Amharic in WSL
# same with --run rag_v2_dev --pc-run rag_v2_dev_pc --countries US Spain South_Korea
```

### Conditions
| Condition | What | Status |
|---|---|---|
| Baseline | no retrieval | ✅ |
| RAG v1 (Always-RAG) | hybrid 0.5/0.5, always top-3, v1 prompt | ✅ |
| RAG v2 (Adaptive) | KB v2 + dense → rerank → τ cut-off → gate, v2 prompt | ✅ |
| Oracle-KB upper bound | cards built from BLEnD answers, a **separate** run and a separately labelled row | next |
| KB v2 for US/ES/KR | question-driven cards for the high-resource countries too | optional |
| Category analysis | Country×Category, Method×Category tables and heatmaps | final (no new LLM calls) |

Useful fact: with greedy decoding, Adaptive-RAG needs **no new LLM runs**. For each question
we take the RAG answer if the gate fires and the baseline answer if not, so gates can be tuned offline.

### Size of the job (per model, per condition)
US: 500 q × 2 prompts = 1,000. The other 4 countries: 500 × 2 languages × 2 prompts = 2,000 each.
Total **9,000 generations**. Baseline + RAG = 18k per model, 54k for our 3 models. The baseline took **~5.5 h** of
PC time (gemma 28 min, mistral 3.7 h, qwen 1.3 h). RAG prompts are longer, so expect ~6–7 h.

---

## 4. Work plan (one big task at a time)

| # | Task | Output | For mid-term? |
|---|---|---|---|
| 1 | **GPU box setup**: SSH access, env (vLLM, transformers, Java for the scorer), HF login + accept Llama/Gemma licenses | working env | ✅ |
| 2 | **Code skeleton** `rag_pipeline/`: data loader (handles BLEnD quirks), prompt builder, inference runner (vLLM, resumable), outputs in BLEnD's CSV format | runs end to end | ✅ |
| 3 | **Scorer wrapper**: calls the official `soft_exact_match`; per-question 0/1 kept for later analysis | `scores.csv` | ✅ |
| 4 | **Pilot**: Qwen3-8B, 50 questions/country, baseline → check outputs are short and clean, check scores are sane vs paper | sanity check | ✅ |
| 5 | **Baseline full run**: 3 Tier-1 models × 5 countries × (local + en) × 2 prompts | Baseline table | ✅ |
| 6 | **Retriever**: BM25 + bge-m3 per country, top-k with scores logged; retrieval diagnostic (how often a top-k card mentions a gold answer; diagnostic only, never used to edit the KB) | retrieval stats | ✅ |
| 7 | **Always-RAG full run** + Baseline vs RAG delta table | RAG table | ✅ |
| 8 | Mid-term slides/report: KB stats, baseline vs paper, RAG deltas, 3–5 qualitative examples | submission | ✅ |
| 9 | Diagnosis of v1 + SemEval-2026 research → RAG v2 design | PLAN/report | ✅ |
| 10 | KB v2 (question-driven cards, AZ + ET) | `KB_Articles/*_KB_v2_additions.jsonl` | ✅ |
| 11 | RAG v2: dense + rerank + τ + gate + prompt v2, all 9 settings | 3-way table | ✅ |
| 12 | Oracle-KB upper bound (separate, labelled) | upper-bound row | next |
| 13 | MT-based pivot vs oracle English query | cross-lingual table | final |
| 14 | Category analysis + error analysis + final report | final | final |

### Code layout (actual)
```
rag_pipeline/
  config.py          countries, languages, models (Ollama tags), paths, PC host
  data.py            load questions/prompts/annotations (fixes BLEnD quirks), scorable filter
  build_jobs.py      questions (+ retrieved cards) → jobs/<run>.jsonl; --prompt-version v1|v2, --pc-run, --countries
  remote/run_ollama.py   runs on the PC: jobs → Ollama /api/chat (greedy) → results/<run>/<model>.jsonl (resumable)
  pc.py              push / start / status / pull via ssh+scp
  retrieve.py        RAG v1 retrieval (hybrid 0.5/0.5, top-3) → results/retrieval/oracle_en_top3.jsonl
  retrieve_v2.py     RAG v2 retrieval (dense top-20 → MiniLM rerank → τ) → results/retrieval/v2_rerank_top3.jsonl
  merge_gated.py     RAG v2 gate: PC answers for prompts with cards + baseline answers for the rest
  score.py           official BLEnD soft_exact_match (module stubs), per-group cache → scores/table
  kb_v2/             KB v2 card sources + build_kb_v2.py
results/
  baseline/ rag/ rag_v2/ rag_v2_dev/      raw/, groups/, per_question.csv, scores.csv, table.csv
  rag_v2_pc/ rag_v2_dev_pc/               raw PC answers for the prompts that got cards
  retrieval/                              retrieval files, rerank cache, logs
  comparison_baseline_rag_ragv2.csv       3-way table
```
