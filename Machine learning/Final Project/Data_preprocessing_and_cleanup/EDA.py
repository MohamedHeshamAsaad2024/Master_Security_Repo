"""
File Name: eda_isot_cleaned.py

Description:
    One-shot EDA script for the Kaggle Fake/Real News Dataset (ISOT), executed in a fixed order:
        1) Class Balance
        2) Document Length Distribution
        3) Vocabulary Richness
        4) Top TF-IDF Tokens per Class
        5) N-gram Effect Inspection
        6) Source Leakage Check
        7) Class-Specific Length Bias

    This script intentionally runs WITHOUT any CLI arguments or modes.
    You only need to update the CONFIGURATION section below once.

How it works:
    - Loads Fake.csv and True.csv
    - Applies the SAME cleaning logic used in feature_pipeline via:
        load_isot_dataset() + prepare_dataframe()
    - Runs EDA and saves:
        - PNG plots
        - eda_summary.txt (key numeric results)
        - top_tokens.txt (top TF-IDF tokens per class)

Requirements:
    - Place this file next to feature_pipeline.py (so imports work)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ==========================================================
# CONFIGURATION (EDIT THESE PATHS ONCE, THEN RUN THE FILE)
# ==========================================================
FAKE_CSV_PATH = "Internal_Dataset/Fake.csv"
TRUE_CSV_PATH = "Internal_Dataset/True.csv"
OUT_DIR = "Output/eda_out"

# EDA knobs (safe defaults)
COUNT_MAX_FEATURES = 5000
TFIDF_MAX_FEATURES = 10000
TOP_K_TOKENS = 20
NGRAM_MAX_VALUES = (1, 2, 3)

# Use the same preprocessing config you use for training
# (Import from your feature pipeline module)
from features_pipeline import FeatureConfig, load_isot_dataset, prepare_dataframe


# -----------------------------
# Helpers
# -----------------------------
def _ensure_out_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_fig(out: Path, filename: str) -> None:
    plt.tight_layout()
    plt.savefig(out / filename, dpi=150)
    plt.close()


def _write_lines(out: Path, lines: List[str], filename: str = "eda_summary.txt") -> None:
    (out / filename).write_text("\n".join(lines), encoding="utf-8")


def _class_counts(y: np.ndarray) -> Dict[str, int]:
    y = np.asarray(y, dtype=int)
    return {"fake_0": int(np.sum(y == 0)), "real_1": int(np.sum(y == 1))}


def _word_counts(texts: pd.Series) -> np.ndarray:
    # word counts using whitespace split; robust and fast
    return texts.astype(str).str.split().map(len).to_numpy(dtype=int)


def _contains_url_or_www(texts: pd.Series) -> np.ndarray:
    return texts.astype(str).str.contains(r"https?://|www\.", regex=True, na=False).to_numpy()


def _contains_email(texts: pd.Series) -> np.ndarray:
    return texts.astype(str).str.contains(r"[\w\.-]+@[\w\.-]+\.\w+", regex=True, na=False).to_numpy()


# -----------------------------
# EDA Steps (run in fixed order)
# -----------------------------
def step1_class_balance(df: pd.DataFrame, out: Path, summary: List[str]) -> None:
    summary.append("=== 1) Class Balance ===")
    y = df["label"].to_numpy(dtype=int)

    counts = pd.Series(y).value_counts().sort_index()
    ratios = pd.Series(y).value_counts(normalize=True).sort_index()

    summary.append(f"Counts: {counts.to_dict()}")
    summary.append(f"Ratios: {ratios.to_dict()}")

    plt.figure()
    counts.plot(kind="bar")
    plt.title("Class Counts (0=fake, 1=real)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    _save_fig(out, "01_class_counts.png")

    plt.figure()
    ratios.plot(kind="bar")
    plt.title("Class Ratios (0=fake, 1=real)")
    plt.xlabel("Class")
    plt.ylabel("Ratio")
    _save_fig(out, "02_class_ratios.png")
    summary.append("")


def step2_document_length_distribution(df: pd.DataFrame, out: Path, summary: List[str]) -> None:
    summary.append("=== 2) Document Length Distribution ===")
    lengths = _word_counts(df["content"])
    summary.append(f"Mean word count: {float(np.mean(lengths)):.2f}")
    summary.append(f"Median word count: {float(np.median(lengths)):.2f}")
    summary.append(f"Min/Max word count: {int(np.min(lengths))}/{int(np.max(lengths))}")

    plt.figure()
    plt.hist(lengths, bins=60)
    plt.title("Document Word Count Distribution (All)")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    _save_fig(out, "03_doc_length_all.png")

    # Per-class plots
    for label in [0, 1]:
        plt.figure()
        plt.hist(lengths[df["label"].to_numpy(dtype=int) == label], bins=60)
        plt.title(f"Document Word Count Distribution (Class={label})")
        plt.xlabel("Word count")
        plt.ylabel("Frequency")
        _save_fig(out, f"04_doc_length_class_{label}.png")

    summary.append("")


def step3_vocabulary_richness(df: pd.DataFrame, out: Path, summary: List[str]) -> None:
    summary.append("=== 3) Vocabulary Richness ===")
    # Proxy: number of unique tokens per document within a capped vocabulary.
    vec = CountVectorizer(max_features=COUNT_MAX_FEATURES, stop_words="english")
    X = vec.fit_transform(df["content"].astype(str).tolist())
    uniq_per_doc = np.asarray((X > 0).sum(axis=1)).ravel()

    summary.append(f"CountVectorizer max_features={COUNT_MAX_FEATURES}")
    summary.append(f"Unique-token proxy mean: {float(np.mean(uniq_per_doc)):.2f}")
    summary.append(f"Unique-token proxy median: {float(np.median(uniq_per_doc)):.2f}")
    summary.append(f"Unique-token proxy min/max: {int(np.min(uniq_per_doc))}/{int(np.max(uniq_per_doc))}")

    plt.figure()
    plt.hist(uniq_per_doc, bins=60)
    plt.title(f"Vocabulary Richness Proxy (max_features={COUNT_MAX_FEATURES})")
    plt.xlabel("Unique tokens per document (within capped vocab)")
    plt.ylabel("Frequency")
    _save_fig(out, "05_vocab_richness_proxy.png")
    summary.append("")


def step4_top_tfidf_tokens_per_class(df: pd.DataFrame, out: Path, summary: List[str]) -> None:
    summary.append("=== 4) Top TF-IDF Tokens per Class ===")

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),  # baseline for interpretability
        stop_words="english",
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X = tfidf.fit_transform(df["content"].astype(str).tolist())
    terms = np.array(tfidf.get_feature_names_out())
    y = df["label"].to_numpy(dtype=int)

    fake_mean = X[y == 0].mean(axis=0).A1
    real_mean = X[y == 1].mean(axis=0).A1

    fake_top = terms[np.argsort(fake_mean)[-TOP_K_TOKENS:]][::-1]
    real_top = terms[np.argsort(real_mean)[-TOP_K_TOKENS:]][::-1]

    summary.append(f"TF-IDF max_features={TFIDF_MAX_FEATURES}, ngram_range=(1,2), top_k={TOP_K_TOKENS}")
    summary.append("Top tokens for FAKE (label=0): " + ", ".join(fake_top.tolist()))
    summary.append("Top tokens for REAL (label=1): " + ", ".join(real_top.tolist()))
    summary.append("")

    (out / "top_tokens.txt").write_text(
        "Top tokens for FAKE (label=0):\n"
        + ", ".join(fake_top.tolist())
        + "\n\nTop tokens for REAL (label=1):\n"
        + ", ".join(real_top.tolist())
        + "\n",
        encoding="utf-8",
    )


def step5_ngram_effect_inspection(df: pd.DataFrame, out: Path, summary: List[str]) -> None:
    summary.append("=== 5) N-gram Effect Inspection ===")
    n_values = list(NGRAM_MAX_VALUES)
    feature_counts: List[int] = []

    for n in n_values:
        tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, n),
            stop_words="english",
            sublinear_tf=True,
            strip_accents="unicode",
        )
        X = tfidf.fit_transform(df["content"].astype(str).tolist())
        feature_counts.append(int(X.shape[1]))
        summary.append(f"ngram_range=(1,{n}) -> features={int(X.shape[1])}")

    plt.figure()
    plt.plot(n_values, feature_counts, marker="o")
    plt.title(f"Feature Count vs ngram_max (max_features cap={TFIDF_MAX_FEATURES})")
    plt.xlabel("ngram_max")
    plt.ylabel("Number of features produced")
    _save_fig(out, "06_ngram_feature_growth.png")
    summary.append("")


def step6_source_leakage_check(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, out: Path, summary: List[str]) -> None:
    """
    Source leakage check is strongest when you compare BEFORE vs AFTER cleaning.
    We compute URL/email presence on:
        - raw combined content (title+text, minimal fillna)
        - cleaned content (after prepare_dataframe + remove_urls_emails)
    """
    summary.append("=== 6) Source Leakage Check (URLs / Emails) ===")

    # Build raw content (minimal normalization) to measure leakage before removal
    raw_tmp = raw_df.copy()
    raw_tmp["title"] = raw_tmp["title"].fillna("").astype(str)
    raw_tmp["text"] = raw_tmp["text"].fillna("").astype(str)
    raw_content = (raw_tmp["title"] + " " + raw_tmp["text"]).astype(str)

    raw_has_url = _contains_url_or_www(raw_content)
    raw_has_email = _contains_email(raw_content)

    clean_has_url = _contains_url_or_www(cleaned_df["content"])
    clean_has_email = _contains_email(cleaned_df["content"])

    summary.append(f"Raw:    % docs with URL/www: {100.0 * float(np.mean(raw_has_url)):.2f}%")
    summary.append(f"Raw:    % docs with email:   {100.0 * float(np.mean(raw_has_email)):.2f}%")
    summary.append(f"Cleaned:% docs with URL/www: {100.0 * float(np.mean(clean_has_url)):.2f}%")
    summary.append(f"Cleaned:% docs with email:   {100.0 * float(np.mean(clean_has_email)):.2f}%")

    # Simple bar plot
    plt.figure()
    vals = [
        100.0 * float(np.mean(raw_has_url)),
        100.0 * float(np.mean(clean_has_url)),
        100.0 * float(np.mean(raw_has_email)),
        100.0 * float(np.mean(clean_has_email)),
    ]
    labels = ["raw_url", "clean_url", "raw_email", "clean_email"]
    plt.bar(labels, vals)
    plt.title("Source Leakage Signals (percentage of docs)")
    plt.ylabel("% of documents")
    _save_fig(out, "07_source_leakage_signals.png")
    summary.append("")


def step7_class_specific_length_bias(df: pd.DataFrame, out: Path, summary: List[str]) -> None:
    summary.append("=== 7) Class-Specific Length Bias ===")
    lengths = _word_counts(df["content"])
    y = df["label"].to_numpy(dtype=int)

    fake_lengths = lengths[y == 0]
    real_lengths = lengths[y == 1]

    summary.append(f"Fake (0) mean/median: {float(np.mean(fake_lengths)):.2f} / {float(np.median(fake_lengths)):.2f}")
    summary.append(f"Real (1) mean/median: {float(np.mean(real_lengths)):.2f} / {float(np.median(real_lengths)):.2f}")

    # Visual comparison as separate histograms (avoid styling assumptions)
    plt.figure()
    plt.hist(fake_lengths, bins=60)
    plt.title("Word Count Distribution — Fake (0)")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    _save_fig(out, "08_length_bias_fake.png")

    plt.figure()
    plt.hist(real_lengths, bins=60)
    plt.title("Word Count Distribution — Real (1)")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    _save_fig(out, "09_length_bias_real.png")

    summary.append("")


# -----------------------------
# Main (single-run, fixed order)
# -----------------------------
def main() -> None:
    out = _ensure_out_dir(OUT_DIR)
    summary: List[str] = []

    # Use your training preprocessing config (same defaults as your pipeline)
    config = FeatureConfig()

    # Load raw (for leakage BEFORE cleaning comparison)
    raw_df = load_isot_dataset(FAKE_CSV_PATH, TRUE_CSV_PATH)

    # Cleaned df for EDA (this is what you said you will use)
    cleaned_df = prepare_dataframe(raw_df, config)

    # Run EDA steps in required order
    step1_class_balance(cleaned_df, out, summary)
    step2_document_length_distribution(cleaned_df, out, summary)
    step3_vocabulary_richness(cleaned_df, out, summary)
    step4_top_tfidf_tokens_per_class(cleaned_df, out, summary)
    step5_ngram_effect_inspection(cleaned_df, out, summary)
    step6_source_leakage_check(raw_df, cleaned_df, out, summary)
    step7_class_specific_length_bias(cleaned_df, out, summary)

    _write_lines(out, summary, "eda_summary.txt")


if __name__ == "__main__":
    main()