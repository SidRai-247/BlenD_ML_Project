"""RAG v3, step 2 (laptop): learn the gate "use the RAG answer or keep the baseline answer?" and score it honestly.

    python rag_pipeline/gate/train_gate.py

Training rows are the answers where the cards changed the outcome (help = 1: RAG right, baseline wrong; help = 0: the
reverse). The gate never sees the gold answers at prediction time, only the features from features.py.

No question is ever scored by a gate that was trained on it (cross-fitting):
  cv    5-fold GroupKFold, grouped by question TEMPLATE. BLEnD asks the same template in every country, language,
        prompt and model, so a template is either entirely in training or entirely in the test fold.
  loco  leave-one-country-out: to predict country C, train only on the OTHER 4 countries (and, as above, never on
        the test templates). This is the "new country" setting.
Inside each training split, an inner 3-fold GroupKFold picks the classifier (logistic regression vs gradient
boosting, by AUC) and the probability threshold t (by baseline-relative gain). RAG answer is used iff P(help) > t.

Outputs in results/gate/:
  predictions.csv   P(help) and the decision per row, for every protocol x feature set
  table.csv         SEM-B per setting (mean of 3 models, avg of inst-4/pers-3): baseline, v2, v3 (each), oracle
  report.txt        AUCs, chosen models/thresholds, rows switched, logistic-regression weights
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # gradient boosting is ~10x slower with OpenMP oversubscription on Windows

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

GATE_DIR = RESULTS_DIR / "gate"
KEYS = ["model", "country", "language", "prompt_no", "ID"]
RETRIEVAL = ["r_max", "r_mean", "r_min", "r_gap", "n_cards"]
BEHAVIOUR = ["base_agree", "base_in_cards", "rag_in_cards", "changed", "rag_base_overlap", "base_len", "rag_len"]
CATEGORICAL = ["model", "prompt_no", "topic"]
CONTEXT = ["is_english"]
THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.05), 2)
SEED = 0


def feature_sets(df):
    sets = {"retrieval": RETRIEVAL + CONTEXT, "no_judge": RETRIEVAL + BEHAVIOUR + CONTEXT}
    if "judge_yes" in df and df["judge_yes"].notna().all():
        sets["full"] = RETRIEVAL + BEHAVIOUR + CONTEXT + ["judge_yes"]
    return sets


def make_model(kind, numeric):
    pre = ColumnTransformer([("num", StandardScaler(), numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    if kind == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    else:
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                             class_weight="balanced", random_state=SEED)
    return make_pipeline(pre, clf)


def gain(y, p, t):
    """Change in #correct answers vs the baseline if RAG is used where p > t (y: 1 help, 0 hurt)."""
    use = p > t
    return int((y[use] == 1).sum() - (y[use] == 0).sum())


def fit_select(train, numeric):
    """Inner 3-fold grouped CV on `train` -> (fitted model, kind, threshold)."""
    X, y, g = train[numeric + CATEGORICAL], train["help"].to_numpy(), train["template"]
    best = None
    for kind in ["logreg", "gbm"]:
        oof = np.zeros(len(train))
        for tr, te in GroupKFold(n_splits=3).split(X, y, g):
            oof[te] = make_model(kind, numeric).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y, oof)
        t = max(THRESHOLDS, key=lambda t: gain(y, oof, t))
        if best is None or auc > best[1]:
            best = (kind, auc, t)
    kind, _, t = best
    return make_model(kind, numeric).fit(X, y), kind, t


def splits(df, protocol):
    """Yield (train_mask, test_mask). Test = all rows (incl. ties) of the held-out templates (and country)."""
    folds = GroupKFold(n_splits=5).split(df, groups=df["template"])
    test_templates = [set(df["template"].iloc[te]) for _, te in folds]
    for tt in test_templates:
        in_test_tpl = df["template"].isin(tt)
        if protocol == "cv":
            yield ~in_test_tpl, in_test_tpl
        else:
            for c in sorted(df["country"].unique()):
                yield ~in_test_tpl & (df["country"] != c), in_test_tpl & (df["country"] == c)


def cross_fit(df, numeric, protocol, log):
    p = pd.Series(np.nan, index=df.index)
    t = pd.Series(np.nan, index=df.index)
    chosen = []
    labelled = df["help"].notna()
    for tr, te in splits(df, protocol):
        model, kind, thr = fit_select(df[tr & labelled], numeric)
        p[te] = model.predict_proba(df.loc[te, numeric + CATEGORICAL])[:, 1]
        t[te] = thr
        chosen.append((kind, thr))
    kinds = pd.Series([k for k, _ in chosen]).value_counts().to_dict()
    thrs = pd.Series([x for _, x in chosen]).describe()[["min", "50%", "max"]].round(2).to_dict()
    log(f"  inner-CV choices: {kinds}; thresholds {thrs}")
    return p, t


def sem_b(pq):
    """BLEnD SEM-B per (country, language): mean over scored questions, avg over prompts, then mean of 3 models."""
    g = pq[pq["score"].notna()].groupby(["model", "country", "language", "prompt_no"])["score"].mean() * 100
    return g.groupby(["model", "country", "language"]).mean().groupby(["country", "language"]).mean()


def main():
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    df = pd.read_csv(GATE_DIR / "features.csv")
    base = pd.read_csv(RESULTS_DIR / "baseline" / "per_question.csv")
    preds = df[KEYS + ["score_b", "score_v", "help"]].copy()
    labelled = df["help"].notna()
    log(f"{len(df)} rows with cards; {labelled.sum()} decisive (help {int(df['help'].sum())}, "
        f"hurt {int((df['help'] == 0).sum())})")

    conditions = {}
    for fs_name, numeric in feature_sets(df).items():
        for protocol in ["cv", "loco"]:
            name = f"{protocol}_{fs_name}"
            log(f"\n[{name}] features: {numeric + CATEGORICAL}")
            p, t = cross_fit(df, numeric, protocol, log)
            use = p > t
            preds[f"p_{name}"], preds[f"use_{name}"] = p.round(4), use.astype(int)
            y = df.loc[labelled, "help"]
            log(f"  AUC {roc_auc_score(y, p[labelled]):.3f}; RAG kept for {use.mean():.1%} of rows; "
                f"help kept {use[labelled & (df['help'] == 1)].mean():.1%}, hurt kept "
                f"{use[labelled & (df['help'] == 0)].mean():.1%}")
            for c in sorted(df["country"].unique()):
                m = labelled & (df["country"] == c)
                if df.loc[m, "help"].nunique() == 2:
                    log(f"    {c:12} AUC {roc_auc_score(df.loc[m, 'help'], p[m]):.3f}  RAG kept {use[df['country'] == c].mean():.1%}")
            conditions[f"v3_{name}"] = np.where(use, df["score_v"], df["score_b"])
    preds.to_csv(GATE_DIR / "predictions.csv", index=False)

    # Full-benchmark scores: rows without cards keep the baseline answer in every condition.
    def with_rows(score_for_carded):
        s = df[KEYS].copy()
        s["s"] = score_for_carded
        pq = base.merge(s, on=KEYS, how="left")  # unscored (skip-filtered) rows are NaN in both runs
        pq["score"] = pq["s"].fillna(pq["score"])
        return sem_b(pq)

    table = pd.DataFrame({"baseline": sem_b(base), "rag_v2": with_rows(df["score_v"])})
    for name, sc in conditions.items():
        table[name] = with_rows(sc)
    table["oracle_gate"] = with_rows(np.fmax(df["score_b"], df["score_v"]))
    order = [("US", "English"), ("Spain", "Spanish"), ("Spain", "English"), ("South_Korea", "Korean"),
             ("South_Korea", "English"), ("Azerbaijan", "Azerbaijani"), ("Azerbaijan", "English"),
             ("Ethiopia", "Amharic"), ("Ethiopia", "English")]
    table = table.loc[order]
    high = table.loc[[k for k in order if k[0] in ("US", "Spain", "South_Korea")]].mean()
    low = table.loc[[k for k in order if k[0] in ("Azerbaijan", "Ethiopia")]].mean()
    table.loc[("High-resource avg", ""), :] = high
    table.loc[("Low-resource avg", ""), :] = low
    table.loc[("All 9 avg", ""), :] = table.loc[order].mean()
    table = table.round(1)
    table.to_csv(GATE_DIR / "table.csv")
    log("\nSEM-B (%), mean of 3 models, avg of inst-4/pers-3:\n")
    log(table.to_string())

    # Interpretability: logistic-regression weights on all decisive rows (standardised features).
    numeric = list(feature_sets(df).values())[-1]
    lr = make_model("logreg", numeric).fit(df.loc[labelled, numeric + CATEGORICAL], df.loc[labelled, "help"])
    names = lr[0].get_feature_names_out()
    w = pd.Series(lr[-1].coef_[0], index=names).sort_values()
    log("\nLogistic-regression weights (positive -> use the RAG answer):\n")
    log(w.round(3).to_string())
    (GATE_DIR / "report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
