"""Readers for BLEnD's released files (handles the dataset's quirks in one place)."""
import json

import pandas as pd

from config import COUNTRY_LANG, DATA_DIR


def load_questions(country):
    """ID, Topic, question in the local language and in English.

    Quirk: in data/questions/*.csv the `Question` column is the LOCAL-language text and
    `Translation` is the ENGLISH text.
    """
    df = pd.read_csv(DATA_DIR / "questions" / f"{country}_questions.csv", encoding="utf-8")
    df = df.rename(columns={"Question": "q_local", "Translation": "q_en"})
    return df[["ID", "Topic", "q_local", "q_en"]]


def question_text(row, country, language):
    """The question in the requested language (for the US the local version is the English one)."""
    return row["q_local"] if language == COUNTRY_LANG[country] else row["q_en"]


def load_prompt_template(country, prompt_no, language):
    """Prompt template with a `{q}` slot, in English or in the local language.

    Quirk: Azerbaijan_prompts.csv lists inst-4/pers-3 twice; like BLEnD's code we take the first row.
    """
    df = pd.read_csv(DATA_DIR / "prompts" / f"{country}_prompts.csv", encoding="utf-8")
    row = df[df["id"] == prompt_no].iloc[0]
    return row["English"] if language == "English" else row["Translation"]


def load_annotations(country):
    with open(DATA_DIR / "annotations" / f"{country}_data.json", encoding="utf-8") as f:
        return json.load(f)


def is_scorable(ann):
    """Same filter as BLEnD's soft_exact_match: skip questions most annotators could not answer."""
    idks = ann["idks"]
    return not (idks["no-answer"] + idks["not-applicable"] >= 3 or idks["idk"] >= 5
                or len(ann["annotations"]) == 0)
