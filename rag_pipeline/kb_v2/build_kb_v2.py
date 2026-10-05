"""Write the question-driven KB v2 cards to KB_Articles/<Country>_KB_v2_additions.jsonl (same schema as v1).

    python rag_pipeline/kb_v2/build_kb_v2.py

Extra fields: `written_for` (BLEnD question IDs whose *question type* motivated the card, used for the
leave-own-cards-out check) and `kb_version`.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT  # noqa: E402
from data import load_questions  # noqa: E402

import azerbaijan_cards  # noqa: E402
import ethiopia_cards  # noqa: E402

MODULES = {"Ethiopia": ethiopia_cards, "Azerbaijan": azerbaijan_cards}


def main():
    for country, mod in MODULES.items():
        valid_ids = set(load_questions(country)["ID"])
        out = ROOT / "KB_Articles" / f"{country}_KB_v2_additions.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for i, (category, src, ids, text, keywords) in enumerate(mod.CARDS, 1):
                bad = [q for q in ids if q not in valid_ids]
                assert not bad, f"{country} card {i}: unknown question ids {bad}"
                title, publisher, url, quality = mod.SOURCES[src]
                card = {
                    "id": f"{country.upper()}_V2_{i:04d}", "region": country, "subregion": "National",
                    "locality": None, "category": category, "topic": keywords[0],
                    "evidence_type": "everyday_practice", "text_original": text, "language_original": "en",
                    "text_en": text, "cultural_scope": "national", "time_scope": "contemporary",
                    "source": {"type": "reference", "title": title, "publisher": publisher, "url": url,
                               "accessed_date": "2026-10-05"},
                    "source_quality": quality, "keywords": keywords,
                    "written_for": ids, "kb_version": "v2_question_driven",
                }
                f.write(json.dumps(card, ensure_ascii=False) + "\n")
        covered = {q for c in mod.CARDS for q in c[2]}
        print(f"{country}: {len(mod.CARDS)} cards -> {out.name}; question types covered: "
              f"{len(covered)}/{len(valid_ids)}")


if __name__ == "__main__":
    main()
