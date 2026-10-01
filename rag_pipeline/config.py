"""Shared settings: which countries/languages/prompts/models we run, and where files live."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLEND_DIR = ROOT / "BLEnD"
DATA_DIR = BLEND_DIR / "data"
JOBS_DIR = ROOT / "jobs"
RESULTS_DIR = ROOT / "results"

# Our 5 countries and their local language (names as used by BLEnD's files and scorer).
COUNTRY_LANG = {
    "US": "English",
    "Spain": "Spanish",
    "South_Korea": "Korean",
    "Azerbaijan": "Azerbaijani",
    "Ethiopia": "Amharic",
}

# The two prompts the BLEnD paper averages over.
PROMPT_NOS = ["inst-4", "pers-3"]

# short name -> Ollama model tag on the GPU PC
MODELS = {
    "gemma3-4b": "gemma3:4b",
    "mistral-7b": "mistral:latest",
    "qwen2.5-3b": "qwen2.5:3b",
}

# GPU PC
REMOTE_HOST = "user@10.50.61.82"
REMOTE_DIR = "Desktop/Siddharth_ML_Project"  # relative to the remote home directory


def country_language_pairs():
    """Every (country, language) we evaluate: local language, plus English for non-English countries."""
    for country, lang in COUNTRY_LANG.items():
        yield country, lang
        if lang != "English":
            yield country, "English"
