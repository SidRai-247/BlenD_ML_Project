# How I Used the LLM Assistant — Key Prompts

**Project:** Region-aware RAG on the BLEnD benchmark (Siddharth Rai, 12342050)
**Assistant:** Claude Code, an LLM coding agent that can read files and run commands, with my permission

This file lists the most important prompts I gave the assistant during the project, in order, with a
note on what each one achieved. The prompts are quoted from the conversation. Spelling mistakes are
fixed, but the wording and content are unchanged.

## How I kept control

- **I set the scope and the constraints.** The assistant proposed options; I chose:
  - which countries and which free models to use
  - an English-only knowledge base, with retrieval queries always in English
  - that the oracle knowledge base is a separate, labelled result
- **I kept the institute GPU machine safe.** I ran the assistant in *manual approval mode*, so I saw and
  approved every command before it ran there. Deleting anything was explicitly forbidden.
- **I asked for explanations before accepting anything.** I asked how the pipeline, the scorer and the
  retrieval worked, and what each parameter meant, before running them.
- **I protected evaluation integrity.** The official BLEnD scorer was used unchanged. I checked that
  speed-ups (caching) did not change its logic. Tuning was done only on the high-resource countries.
- **I worked one big task at a time.** I asked for concise answers and committed at each milestone.

---

## 1. Context and scope

> "I am working on an ML project on making a RAG-based solution to improve LLM answers to
> country/region-related questions. Please read the PDF files — BLEnD (the original paper) and the SOP
> file — carefully to understand what was done and what I am planning to do as per the SOP. … In our
> limited scope we only work on 5 countries and only on a few free LLM models. … Read all the necessary
> files to get the project context."

*The assistant read the paper, the SOP and the repository before doing anything.*

> "Current plan: we will use the KB only in English as of now, so for KB matching, if a question is asked
> in a local language we will internally translate it to English for KB matching only, not in what is fed
> to the LLM. … Explain me the flow — what is available, what is done, what needs to be done — like a
> plan. … There is a mid-term submission, so I want to show that we have made KBs, run the baseline (LLMs
> without RAG) and see its score, then make the basic RAG and see the score. … Please work one big task at a
> time. Take your time."

*This fixed the core design rule (English KB; translation used only for search) and the mid-term goals.*

## 2. Choosing models with explicit constraints

> "First decide 3 LLM models that can be downloaded and run for free, with no license or any other
> nonsense; we need full control. Can these be run using the scripts we have or will make?"

*Licence-free, locally run models only: Gemma 3 4B, Mistral 7B and Qwen2.5 3B, run with Ollama.*

## 3. Safe use of a shared institute machine

> "ssh … In Desktop there is a folder Siddharth_ML_Project; work inside that only. Do the setup to run
> these 3 LLM models on this remote PC. In between, keep explaining to me concisely what you are doing.
> I have set manual mode to see what commands you run; since this is an institute PC we must be careful."

> "Just check, DO NOT DELETE ANYTHING. Also see where the rest of the space is."

*All remote work stayed inside one folder, every command was approved by me, and nothing was deleted.*

## 4. Understanding the pipeline before running it

> "What is the entire flow: where are the questions, how are they given to the LLM, which script stores
> the results, how is it evaluated — by what code, and what is its logic? Please explain this like I have
> no prior knowledge, and also write it in FLOW.md."

> "What will the full pipeline be to run the full baseline … can all this be done using a script? Please
> explain this flow, then run the pilot test (which must include the scoring as well; I want it end to
> end)."

*I insisted on a small end-to-end pilot, including scoring, before spending GPU hours on 27,000 answers.*

## 5. Protecting evaluation integrity

> "Can't we run the scorer here in Windows? Please check this first, because we have most things set up
> here."

> "I hope you are not changing the original scorer logic with the help of the cache. … Explain very
> concisely."

*The official BLEnD scorer runs unmodified; the assistant confirmed the cache gives identical scores.*

## 6. Designing the RAG method against the SOP

> "Now let's do the RAG part. As mentioned in the SOP, we will combine the BM25 and embedding scores,
> right? Can you figure out the exact details — what embedder to use freely, how to combine them — be
> consistent with the SOP, then explain it to me in short and write it in PLAN.md."

> "Just to confirm: suppose we have 2 questions, 1 in the local language and 1 its English translation in
> the data. In RAG we use the English version for both, so when the models are asked the 2 questions, the
> same 3 KB articles will be sent. … This is what we are doing currently, right?"

*The design was written down and confirmed before running, so local and English results stay
comparable.*

## 7. Critical analysis when results were bad

> "Are these results very bad, or does our RAG need refinement, or does it make sense? Why did the
> Ethiopia English score reduce? Azerbaijan is a low-resource country, so its score should improve. I want
> you to research what the winning teams did, take inspiration from them, and suggest changes … the
> retrieval logic, the 0.5/0.5 weights … DO DEEP RESEARCH and propose a solution to me."

*This led to a diagnosis that RAG helps only when the right card is retrieved. It also led to a review of
SemEval-2026 Task 7 (built on BLEnD) and a concrete redesign, RAG v2.*

## 8. Demanding exact clarity before approving changes

> "Answer concisely: what change exactly are you proposing in retrieval? What weight for BM25, which
> encoder, what threshold, how are relevant and irrelevant decided, is it given to the model as a bool,
> is the k value changed? Provide clarity."

> "What is the meaning of 'a question can't see its own cards'? What does Hit@3 mean? Answer concisely."

> "You built the KB question-wise, but it is a common KB bank, right? Per-question retrieval will work on
> the entire set of cards, right?"

*Every parameter was explained before I approved it. The last check confirmed there was no hidden
per-question shortcut at retrieval time.*

## 9. Transparency about the knowledge base

> "OK, so after C we'll add these oracle KB articles to improve scores, but in the report we'll mention
> that. Sounds good?"

*Agreed rule: anything built from benchmark answers is a separate run, reported as a labelled upper
bound and never mixed into the main result.*

## 10. Experimental completeness

> "We ran the new RAG version for Ethiopia and Azerbaijan only, but we did the rerank and new retrieval
> for the first 3 countries. Don't we need to run them on the PC too?"

> "Yeah, so we will finally have 3 tables: baseline, the old RAG, this new RAG."

*All 9 settings were evaluated under all three systems, so the report also shows where the method still
costs accuracy (the high-resource countries).*

## 11. Reporting and documentation

> "Make the LaTeX report have all tables, explain the approach of RAG v1 and RAG v2, the values, the stats,
> all the parameter and hyper-parameter values (like the threshold) that were used. Then also update
> PLAN.md and FLOW.md with respect to the current RAG version."

*The report states every setting, how each was chosen, and the limitations of the method.*

---

## Outcome

- **Baseline:** 27,000 answers, scored with BLEnD's official scorer.
- **RAG v1:** the first attempt; it lowered accuracy, and we analysed why.
- **RAG v2:** improved every low-resource setting by +8.6 to +36.9 points, and reduced the gap between
  high- and low-resource cultures from 27.2 to 2.0 points.
- **Caveats:** stated in the report.
