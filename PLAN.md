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

| Country | Cards | Note |
|---|---|---|
| US | 150 | 25 per category |
| Spain | 150 | 25 per category |
| South Korea | 130 | sports/education thin |
| Azerbaijan | 150 | cards are very short (~14 words) |
| Ethiopia | 140 | partial |

**Decision: English-only KB.** We index `text_en` only.

### Pipeline

```
                 ┌──────────────── BASELINE ────────────────┐
question (local  │  prompt(inst-4 / pers-3, same language)  │──► LLM ──► answer ──► official scorer
 or English) ────┤                                          │
                 └───────────────── RAG ────────────────────┘
                   1. retrieval query = English version of the question
                      (for a local-language question we translate it ONLY for search)
                   2. hybrid search in THIS country's KB only:
                      BM25 (keywords) + bge-m3 embeddings (meaning), fused → top-k cards
                   3. prompt = [retrieved English cards] + original prompt in the ORIGINAL language
                   4. LLM ──► answer ──► official scorer
```

Key rule: **translation is used only for searching.** The LLM always gets the original question in its
original language, so the comparison with the baseline stays clean.

Translation source for the search query: BLEnD already ships a human English version of every question
(`Translation` column). For the mid-term we use that ("oracle translation"). For the final report
we add machine translation (the LLM itself or NLLB) to show the realistic pipeline and measure the gap.

### Conditions
| Condition | What | Status |
|---|---|---|
| Baseline | no retrieval | **mid-term** |
| Always-RAG | retrieve for every question | **mid-term** |
| Adaptive-RAG | a gate decides per question: retrieve or not (retrieval score threshold / category prior) | final |
| Category analysis | Country×Category, Method×Category tables and heatmaps | final (no new LLM calls) |

Useful fact: with greedy decoding, Adaptive-RAG needs **no new LLM runs**. For each question
we take the RAG answer if the gate fires and the baseline answer if not, so gates can be tuned offline.

### Size of the job (per model, per condition)
US: 500 q × 2 prompts = 1,000. The other 4 countries: 500 × 2 languages × 2 prompts = 2,000 each.
Total **9,000 generations**. Baseline + RAG = 18k per model, 54k for the Tier-1 trio. That is small for vLLM.

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
| 9 | Tier-2 models (Aya-101, GPT-OSS-20B) | more rows | final |
| 10 | Adaptive-RAG gate (offline, reuses 5 + 7) | 3-way table | final |
| 11 | MT-based pivot vs oracle English query | cross-lingual table | final |
| 12 | Category analysis + error analysis + final report | final | final |

### Proposed code layout
```
rag_pipeline/
  config.py        countries, languages, models, paths
  data.py          load questions/prompts/annotations (fixes BLEnD quirks)
  retriever.py     BM25 + bge-m3 hybrid, per-country index
  prompts.py       baseline & RAG prompt builders
  run.py           inference: --model --condition {baseline,rag} --countries ...
  score.py         wraps BLEnD/evaluation/exact_match.soft_exact_match
  analysis/        tables & plots
results/
  {condition}/{model}-{Country}_{Language}_{prompt}_result.csv   (BLEnD format)
  scores.csv
```
