# How a Baseline Run Works — End to End

This explains, from zero, what happens when we "run the baseline": where the questions are,
how they reach the LLM, where the answers are saved, and how they are scored.

> **Baseline** = ask the LLM the question with **no extra information** (no RAG).
> It measures what the model already "knows" about each culture.
> Later, RAG runs the exact same flow with retrieved KB text added to the prompt.

---

## 0. The big picture

```
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
 │ 1. QUESTIONS         │   │ 2. PROMPT            │   │ 3. LLM           │   │ 4. RESPONSE FILE     │
 │ BLEnD/data/questions │──►│ BLEnD/data/prompts   │──►│ Ollama on GPU PC │──►│ results/baseline/    │
 │ {Country}_questions  │   │ inst-4 / pers-3      │   │gemma/mistral/qwen│   │ *_result.csv         │
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

One CSV per **(model, country, language, prompt)**:

```
results/baseline/{model}-{Country}_{Language}_{prompt}_result.csv
e.g.  results/baseline/gemma3-4b-South_Korea_Korean_inst-4_result.csv
```

This name pattern and these columns are **exactly what BLEnD's scorer expects**, so the scorer can read our files:

| ID | question | prompt | response | prompt_no |
|---|---|---|---|---|
| Al-en-09 | 대한민국의 놀이공원에서… | 다음 질문을 읽고… | 츄러스 | inst-4 |

Count per model: 9 (country, language) pairs × 2 prompts = **18 files × 500 rows = 9,000 answers**.
For 3 models: 54 files, 27,000 answers.

The runner is **resumable**. If a file already has some IDs, those are skipped, so a crash or SSH drop
doesn't waste finished work.

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

Our summarise step collects every FINAL number into `results/scores.csv`, and prints a table like:

| Model | US-en | ES-es | ES-en | KR-ko | KR-en | AZ-az | AZ-en | ET-am | ET-en |
|---|---|---|---|---|---|---|---|---|---|
| gemma3-4b | … | … | … | … | … | … | … | … | … |
| mistral-7b | … | | | | | | | | |
| qwen2.5-3b | … | | | | | | | | |

We also keep **per-question** 0/1 scores (`results/baseline/*_response_score.csv`). Later, the RAG
comparison and the category analysis are built on those without re-running any LLM.

---

## 7. So what exactly is "broken" in BLEnD, and what do we do about it?

The **data** (questions, prompts, gold answers) and the **scoring logic** are fine. The problems are in the
glue scripts around them:

| BLEnD file | Problem | What we do |
|---|---|---|
| `model_inference.py` | crashes on startup (`parser.add.argument` typo); downloads prompts from the authors' Google Sheet instead of the CSVs; expects files named `*_full_final_questions.csv`, which don't exist; built for 2024 API models | **Replace** with our own runner `rag_pipeline/run.py` (steps 2–4 above) |
| `evaluation/evaluate.py` | for English runs it calls the multiple-choice scorer with undefined variables, so it crashes | **Replace** with `rag_pipeline/score.py`, which calls `soft_exact_match` directly |
| `evaluation/exact_match.py` | the logic is fine, but at the top it imports tools for **all 13 languages**, some of them folders not in the repo (`SUSTEM`, `stemmer`, `indic_nlp_library`). The import alone fails | Import the file **unchanged**, register harmless stand-ins for the languages we don't use (Sundanese, Greek, Assamese, …), and install the real tools for our 5 languages |

The official scoring function itself runs **untouched**, so our numbers are directly comparable to the paper.

---

## 8. What the commands will look like

(These scripts are written in the next task; this shows how they'll be used.)

```bash
# on the GPU PC
python rag_pipeline/run.py   --condition baseline --model gemma3-4b          # 18 files, ~9k answers
python rag_pipeline/run.py   --condition baseline --model mistral-7b
python rag_pipeline/run.py   --condition baseline --model qwen2.5-3b
python rag_pipeline/score.py --condition baseline                           # official scorer on all files
python rag_pipeline/summarize.py                                           # prints/saves the table
```

A pilot comes first (`--limit 50`: 50 questions per file) to check that answers are short and clean
before spending GPU time on the full run.

### Files at a glance
| What | Where | Who wrote it |
|---|---|---|
| Questions | `BLEnD/data/questions/{Country}_questions.csv` | BLEnD |
| Prompt templates | `BLEnD/data/prompts/{Country}_prompts.csv` | BLEnD |
| Gold answers | `BLEnD/data/annotations/{Country}_data.json` | BLEnD |
| Scoring logic | `BLEnD/evaluation/exact_match.py` (+ `evaluation_utils.py`) | BLEnD (used unchanged) |
| Inference runner | `rag_pipeline/run.py` | us |
| Scorer wrapper | `rag_pipeline/score.py` | us |
| Summary | `rag_pipeline/summarize.py` → `results/scores.csv` | us |
| Model answers | `results/baseline/*_result.csv` | generated |
| Per-question scores | `results/baseline/*_response_score.csv` | generated |
