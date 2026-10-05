# How a Baseline Run Works — End to End

This explains, from zero, what happens when we "run the baseline": where the questions are,
how they reach the LLM, where the answers are saved, and how they are scored.

> **Baseline** = ask the LLM the question with **no extra information** (no RAG).
> It measures what the model already "knows" about each culture.
> RAG runs the exact same flow with retrieved KB text added to the prompt. Section 9 explains
> RAG v1 (Always-RAG) and the current version, **RAG v2** (reranked, gated retrieval).

---

## 0. The big picture

```
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
 │ 1. QUESTIONS         │   │ 2. PROMPT            │   │ 3. LLM           │   │ 4. RESPONSE FILE     │
 │ BLEnD/data/questions │──►│ BLEnD/data/prompts   │──►│ Ollama on GPU PC │──►│ results/<run>/raw/   │
 │ {Country}_questions  │   │ inst-4 / pers-3      │   │gemma/mistral/qwen│   │ <model>.jsonl        │
 └──────────────────────┘   └──────────────────────┘   └──────────────────┘   └──────────┬───────────┘
                                                                                         │
 ┌──────────────────────┐   ┌──────────────────────────────────────────────┐             │
 │ 6. SUMMARY TABLE     │◄──│ 5. SCORER (official BLEnD code)              │◄────────────┘
 │ results/scores.csv   │   │ BLEnD/evaluation/exact_match.py              │   + gold answers from
 │ model×country×lang   │   │ soft_exact_match(): is any gold answer       │   BLEnD/data/annotations/
 └──────────────────────┘   │ inside the model's response? → 1 / 0         │   {Country}_data.json
                            └──────────────────────────────────────────────┘
```

Steps 1, 2 and 5 are **BLEnD's own files**, which we reuse. Steps 3, 4 and 6 are **our scripts**.

---

## 1. Where the questions are

`BLEnD/data/questions/{Country}_questions.csv`. There is one file per country and **500 rows** in each.

| Column | Meaning | Example (South_Korea) |
|---|---|---|
| `ID` | question id, **the same across all countries** | `Al-en-09` |
| `Topic` | category (Food, Sport, Family, Education, Holidays/Celebration/Leisure, Work life) | `Food` |
| `Question` | the question in the **local language** ⚠️ | `대한민국의 놀이공원에서 인기 있는 간식은 무엇인가요?` |
| `Translation` | the question in **English** ⚠️ | `What is a popular snack at an amusement park in South Korea?` |

⚠️ The names look backwards but this is how BLEnD ships them: `Question` = local, `Translation` = English.
(For the US both columns are English.)

**Which versions we run:**

| Country | Local-language run | English run |
|---|---|---|
| US | — (local is English) | ✅ |
| Spain | ✅ Spanish | ✅ |
| South_Korea | ✅ Korean | ✅ |
| Azerbaijan | ✅ Azerbaijani | ✅ |
| Ethiopia | ✅ Amharic | ✅ |

That gives 9 (country, language) pairs.

---

## 2. How a question becomes a prompt

`BLEnD/data/prompts/{Country}_prompts.csv` holds prompt templates. The paper used two of them, `inst-4` and `pers-3`,
and **every question is asked twice, once with each**. The template has an `English` column and a
`Translation` column (the same template in the local language). The question's language decides which one is used.

**inst-4 (direct instruction):**
```
Read the following question and provide a single answer without any explanations.

Question: {q}
Answer:
```

**pers-3 (persona):**
```
You are a person from South Korea who is trying to explain your country's culture to a foreigner.
Answer the following question, providing a single answer without any explanations.

{q}
```

`{q}` is replaced by the question text. Example, a Korean run with `inst-4`:
```
다음 질문을 읽고 설명 없이 단 하나의 답변만을 제공하시오.

질문: 대한민국의 놀이공원에서 인기 있는 간식은 무엇인가요?
답변:
```

Why two prompts? A single wording can accidentally help or hurt a model. Averaging two makes the score more stable.
This is the paper's protocol, so we copy it exactly.

Small quirk: `Azerbaijan_prompts.csv` lists `inst-4` and `pers-3` twice. We take the first row, as a pandas lookup would.

---

## 3. How the prompt is given to the LLM

The LLMs run on the institute GPU PC inside **Ollama**, a program that is already installed and running there
as a service. Ollama loads a model onto the GPU and answers HTTP requests at `http://localhost:11434`.
Our script is plain Python (only `requests` + `pandas`, both already on the PC). For each prompt it sends
one request and gets the answer back, so **no Python environment has to be installed**.

1. **Chat format.** Instruction-tuned models expect a conversation, not raw text. We send
   `{"model": "gemma3:4b", "messages": [{"role": "user", "content": <prompt>}]}` to `/api/chat`, and
   Ollama wraps it in the model's own *chat template* (its special tokens).
2. **Greedy decoding** (`temperature = 0`). The model always picks its single most likely next word.
   The run is then deterministic: the same prompt always gives the same answer. The paper did the same.
3. **Max length.** `num_predict = 512` new tokens, as in the paper. In practice the answers are a few words long.
4. **Speed.** The first request loads the model (~30–50 s), then each answer takes ~0.1–0.3 s.

Models (see PLAN.md §2):

| Short name | Ollama model | Size |
|---|---|---|
| `gemma3-4b` | `gemma3:4b` | 4.3B, 4-bit |
| `mistral-7b` | `mistral:latest` (v0.3) | 7.2B, 4-bit |
| `qwen2.5-3b` | `qwen2.5:3b` | 3.1B, 4-bit |

---

## 4. Where the responses go

All prompts of a run are first written to one jobs file, `jobs/<run>.jsonl` (`build_jobs.py`), one line per
prompt with `job_id = "Country|Language|prompt_no|ID"`. On the PC, `remote/run_ollama.py` writes **one JSONL
per model**, which `pc.py pull` copies back:

```
results/<run>/raw/<model>.jsonl
e.g.  results/baseline/raw/gemma3-4b.jsonl
{"job_id": "South_Korea|Korean|inst-4|Al-en-09", "model": "gemma3:4b", "response": "츄러스", "seconds": 0.21}
```

Count per model: 9 (country, language) pairs × 2 prompts × 500 = **9,000 answers**; 27,000 for 3 models.

The runner is **resumable**. Job IDs already in the output file are skipped, so a crash or SSH drop
doesn't waste finished work.

`score.py` then converts each (model, country, language, prompt) group into **exactly the CSV format
BLEnD's scorer expects** (`results/<run>/csv/…_result.csv`, columns ID / question / prompt / response /
prompt_no) and calls the official function on it.

---

## 5. How the answers are scored (the official logic)

**File:** `BLEnD/evaluation/exact_match.py`, function `soft_exact_match(...)`.
**Gold answers:** `BLEnD/data/annotations/{Country}_data.json`, e.g.

```json
"Al-en-09": {
  "question":    "대한민국의 놀이공원에서 인기 있는 간식은 무엇인가요?",
  "en_question": "What is a popular snack at an amusement park in South Korea?",
  "annotations": [
    {"answers": ["츄러스", "추러스"], "en_answers": ["churros"],                  "count": 4},
    {"answers": ["콜팝"],             "en_answers": ["cola and popcorn chicken"], "count": 1},
    {"answers": ["핫도그"],           "en_answers": ["corn dog"],                 "count": 1}
  ],
  "idks": {"idk": 0, "no-answer": 0, "not-applicable": 0}
}
```
`answers` = what annotators wrote (local language), `en_answers` = English translation,
`count` = how many of the 5 annotators gave it, `idks` = how many said "I don't know / no answer / not applicable".

### Step by step, for each question

**a) Skip unanswerable questions.** If `no-answer + not-applicable ≥ 3` or `idk ≥ 5`, the question is not
counted at all. (That is why the US has 461 scored questions, not 500.)

**b) Clean the response** (`evaluation_utils.get_llm_response_by_id` → `delete_prompt_from_answer`):
remove any echo of the prompt, drop a leading label like `answer:` / `답변:`, lowercase it,
strip trailing periods. E.g. `"Answer: Churros."` → `"churros"`.

**c) Match against gold answers** (`lemma_check`). Go through the gold answers from most to fewest votes:
   - For a **local-language run**, try each local `answers` first, then the `en_answers`.
   - For an **English run**, try only the `en_answers`.
   - A gold answer **matches** if either:
     1. it appears as a substring of the response (`"churros" in "churros"` ✔), or
     2. after splitting both into *base forms* (lemmas/stems), **every** word of the gold answer appears among the
        response's words. This forgives grammar: "hot dogs" vs "hot dog", Korean particles, Spanish plurals.

     Tools per language: English → spaCy `en_core_web_sm`; Spanish & Amharic → Spark-NLP lemmatizer;
     Korean → Okt (konlpy); Azerbaijani → aznlp stemmer.

**d) Score the question:** `1` if any gold answer matched, else `0`.
(There is also a weighted variant, "SEM-W", worth `votes / max_votes`. The paper reports the plain
1/0 version, "SEM-B", and so do we.)

### Worked examples (Korea, question above)

| Model response | Cleaned | Match? | Score |
|---|---|---|---|
| `츄러스` | `츄러스` | local answer `츄러스` is inside | **1** |
| `Churros.` | `churros` | English answer `churros` is inside | **1** |
| `떡볶이` | `떡볶이` | no gold answer inside | **0** |
| `Popular snacks include churros, corn dogs and more…` | … | contains `churros` | **1** ← long answers can get lucky |

The last row is a known weakness of this metric. That is why the prompts insist on "a single answer".

### From question scores to the numbers in the table

```
score(model, country, language, prompt) = (# questions scored 1) / (# scored questions) × 100
FINAL(model, country, language)         = average of the inst-4 score and the pers-3 score
```

This FINAL number is what the paper reports in its Tables 10/11, and what we compare against.

---

## 6. The summary table

`score.py` collects every FINAL number into `results/<run>/scores.csv` and `table.csv`, and prints a table like:

| Model | US-en | ES-es | ES-en | KR-ko | KR-en | AZ-az | AZ-en | ET-am | ET-en |
|---|---|---|---|---|---|---|---|---|---|
| gemma3-4b | … | … | … | … | … | … | … | … | … |
| mistral-7b | … | | | | | | | | |
| qwen2.5-3b | … | | | | | | | | |

We also keep **per-question** 0/1 scores (`results/<run>/per_question.csv`, and one file per group in
`results/<run>/groups/`). The RAG comparisons and the category analysis are built on those without
re-running any LLM.

---

## 7. So what exactly is "broken" in BLEnD, and what do we do about it?

The **data** (questions, prompts, gold answers) and the **scoring logic** are fine. The problems are in the
glue scripts around them:

| BLEnD file | Problem | What we do |
|---|---|---|
| `model_inference.py` | crashes on startup (`parser.add.argument` typo); downloads prompts from the authors' Google Sheet instead of the CSVs; expects files named `*_full_final_questions.csv`, which don't exist; built for 2024 API models | **Replace** with our own `rag_pipeline/build_jobs.py` + `pc.py` + `remote/run_ollama.py` (steps 2–4 above) |
| `evaluation/evaluate.py` | for English runs it calls the multiple-choice scorer with undefined variables, so it crashes | **Replace** with `rag_pipeline/score.py`, which calls `soft_exact_match` directly |
| `evaluation/exact_match.py` | the logic is fine, but at the top it imports tools for **all 13 languages**, some of them folders not in the repo (`SUSTEM`, `stemmer`, `indic_nlp_library`). The import alone fails | Import the file **unchanged**, register harmless stand-ins for the languages we don't use (Sundanese, Greek, Assamese, …), and install the real tools for our 5 languages |

The official scoring function itself runs **untouched**, so our numbers are directly comparable to the paper.

---

## 8. The commands (baseline)

Everything is started from the laptop; the PC only runs Ollama.

```bash
python rag_pipeline/build_jobs.py --run baseline        # jobs/baseline.jsonl (9,000 prompts)
python rag_pipeline/pc.py push   --run baseline         # copy runner + jobs to the PC
python rag_pipeline/pc.py start  --run baseline         # gemma → mistral → qwen, detached on the PC
python rag_pipeline/pc.py status --run baseline         # progress
python rag_pipeline/pc.py pull   --run baseline         # answers → results/baseline/raw/
python rag_pipeline/score.py --run baseline --languages English,Korean,Azerbaijani   # Windows venv
python rag_pipeline/score.py --run baseline --languages Spanish,Amharic              # WSL (needs Spark)
```

A pilot came first (`build_jobs.py --run pilot --limit 20`) to check that answers are short and clean
before spending GPU time on the full run.

### Files at a glance
| What | Where | Who wrote it |
|---|---|---|
| Questions | `BLEnD/data/questions/{Country}_questions.csv` | BLEnD |
| Prompt templates | `BLEnD/data/prompts/{Country}_prompts.csv` | BLEnD |
| Gold answers | `BLEnD/data/annotations/{Country}_data.json` | BLEnD |
| Scoring logic | `BLEnD/evaluation/exact_match.py` (+ `evaluation_utils.py`) | BLEnD (used unchanged) |
| Jobs builder | `rag_pipeline/build_jobs.py` → `jobs/<run>.jsonl` | us |
| PC control + runner | `rag_pipeline/pc.py`, `rag_pipeline/remote/run_ollama.py` | us |
| Scorer wrapper | `rag_pipeline/score.py` → `results/<run>/{scores,table,per_question}.csv` | us |
| Knowledge bases | `KB_Articles/{Country}_KB*.jsonl` (v1), `KB_Articles/{Country}_KB_v2_additions.jsonl` (v2) | us |
| Retrievers | `rag_pipeline/retrieve.py` (v1), `rag_pipeline/retrieve_v2.py` (v2) | us |
| Gate / merge | `rag_pipeline/merge_gated.py` | us |
| Model answers | `results/<run>/raw/<model>.jsonl` | generated |
| 3-way comparison | `results/comparison_baseline_rag_ragv2.csv` | generated |

---

## 9. How a RAG run works (v1 and the current v2)

RAG adds **one step before the prompt**: find a few KB cards about the question's country and put them
in front of the unchanged baseline prompt. Everything after that (Ollama, scoring) is identical.

```
 question (e.g. Amharic) ─┐
                          │  English version of the same question (BLEnD's own translation)
                          ▼
              ┌─────────────────────────┐
              │ RETRIEVAL (laptop, once)│   only this country's KB cards
              └───────────┬─────────────┘
                          ▼
          0 cards ───────────────────────► baseline prompt → reuse the baseline answer   (v2 only)
          1–3 cards ─────► [instruction + cards] + baseline prompt in the ORIGINAL language → PC → answer
                                                                                              │
                                                                         official scorer ◄────┘
```

### RAG v1 — Always-RAG (first attempt)
1. Score every card of the country: **0.5 × BM25 + 0.5 × bge-m3 cosine** (each min-max normalised).
2. Always take the **top 3**.
3. Prompt header: *"Background information about {country} (may or may not be relevant):"*.

**Result:** it hurt all models (−9 to −14 points). The KB contained the answer for only 7–14% of
questions, so most cards were noise; models refused ("not provided") or copied wrong facts from cards.

### RAG v2 — current version
1. **KB:** KB v1 + KB v2 additions for Azerbaijan and Ethiopia (question-driven everyday cards).
2. **Candidates:** dense-only (bge-m3 cosine, α = 1.0) → top 20 cards.
3. **Rerank:** a cross-encoder (`ms-marco-MiniLM-L-12-v2`) reads (question, card) together and gives a
   relevance score from 0 to 1.
4. **Cut-off:** keep at most 3 cards with score ≥ **τ = 0.45**. α and τ were chosen **only on US / Spain /
   South Korea**, never on Azerbaijan / Ethiopia.
5. **Gate:** if no card passes, the prompt is exactly the baseline prompt. Decoding is greedy, so the answer
   is the baseline answer, and we reuse it instead of asking the LLM again.
6. **Prompt v2 header:** *"Background information about {country}. Use it only if it directly helps to
   answer the question; otherwise ignore it and answer from your own knowledge. Always give an answer and
   never say that the information is not provided."*

**Result:** every Azerbaijan/Ethiopia setting improves for every model (+8.6 to +36.9); the high/low resource
gap falls from 27.2 to 2.0 points. US/Spain/Korea (old KB only) lose 1.7–6.4 points.

```bash
python rag_pipeline/kb_v2/build_kb_v2.py          # write KB v2 files
python rag_pipeline/retrieve_v2.py                # retrieval + rerank + τ → results/retrieval/v2_rerank_top3.jsonl
python rag_pipeline/build_jobs.py --run rag_v2 --pc-run rag_v2_pc --prompt-version v2 \
       --retrieval results/retrieval/v2_rerank_top3.jsonl --countries Azerbaijan Ethiopia
python rag_pipeline/pc.py push|start|status|pull --run rag_v2_pc     # only prompts that got cards
python rag_pipeline/merge_gated.py --run rag_v2 --pc-run rag_v2_pc   # fill the rest with baseline answers
python rag_pipeline/score.py --run rag_v2                            # (Amharic/Spanish in WSL)
```
