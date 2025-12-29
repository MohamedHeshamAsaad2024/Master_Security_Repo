"""
Module Name: feature_pipeline

Description:
    Feature extraction + preprocessing pipeline for Kaggle Fake/Real News Dataset (ISOT).
    This module loads raw Fake.csv and True.csv files, cleans and vectorizes the text,
    and produces standardized machine-learning ready feature matrices.

Outputs (in out_dir):
    - X_train_unscaled.npz
        Sparse TF-IDF feature matrix for the training set without scaling.
        This is used mainly for Multinomial Naive Bayes and XGBoost models.

    - X_test_unscaled.npz
        Sparse TF-IDF feature matrix for the test set without scaling.
        This is used mainly for Multinomial Naive Bayes and XGBoost models.

    - X_train_scaled.npz
        Standardized sparse feature matrix for the training set
        (TF-IDF + optional subject features after StandardScaler).
        Used primarily for Logistic Regression and SVM.

    - X_test_scaled.npz
        Standardized sparse feature matrix for the test set.
        Used primarily for Logistic Regression and SVM.

    - y_train.csv
        Label vector for the training set.
        Values: 0 = Fake News, 1 = Real News.

    - y_test.csv
        Label vector for the test set.

    - Artifacts (Needed for reproducability and inference-time deployment):
        - tfidf.joblib
            Fitted TfidfVectorizer object used to convert text into TF-IDF features.

        - subject_ohe.joblib (only if include_subject=True)
            One-Hot Encoder for the subject column.

        - scaler.joblib
            Fitted StandardScaler used to normalize sparse TF-IDF features.

    - Processing Summary:
        - config.json
            Snapshot of FeatureConfig parameters used for this experiment.

        - split_info.json
            Metadata about the train/test split (random seed, test size, sample counts).

        - stats.json
            Dataset statistics including number of samples, number of features,
            class distribution, and final matrix shapes.
"""

# -----------------------------
# Imports
# -----------------------------
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.sparse import csr_matrix, hstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class FeatureConfig:
    # =========================
    # Core experiment control
    # =========================

    random_state: int = 42
    # Controls all randomness in the pipeline (train/test split).
    # Changing this value changes which samples go to train/test and
    # may slightly affect model accuracy.

    test_size: float = 0.20
    # Fraction of dataset used as final unseen test set.
    # 0.20 = 80% training / 20% testing (best trade-off for this dataset).

    # =========================
    # Feature inclusion
    # =========================

    include_subject: bool = False
    # If True, the "subject" column is one-hot encoded and appended as features.
    # This can boost accuracy artificially by leaking category bias.
    # Keep False for realistic generalization.

    drop_date: bool = True
    # Always drop the "date" column.
    # Dates encode publication source patterns and create severe data leakage.

    # =========================
    # Missing value handling
    # =========================

    fill_missing_with_empty: bool = True
    # Replace NaN title/text with empty strings.
    # Prevents vectorizer crashes when missing data exists.

    # =========================
    # Duplicate handling
    # =========================

    drop_exact_duplicates: bool = True
    # Remove identical articles (based on combined content).
    # Prevents memorization where identical samples appear in train & test.

    # =========================
    # Text cleaning
    # =========================

    lowercase: bool = True
    # Convert all text to lowercase.
    # Reduces vocabulary size and improves TF-IDF consistency.

    remove_urls_emails: bool = True
    # Remove URLs and email addresses.
    # Prevents learning publisher/source shortcuts.

    normalize_whitespace: bool = True
    # Collapse repeated spaces/newlines into a single space.
    # Improves tokenization stability.

    # =========================
    # TF-IDF feature extraction
    # =========================

    ngram_min: int = 1
    ngram_max: int = 2
    # Word n-gram range.
    # (1,1) → single words only
    # (1,2) → words + short phrases (best baseline)
    # (1,3) → better context but much larger feature space.

    max_features: int = 20000
    # Maximum vocabulary size.
    # Too small → underfitting.
    # Too large → RAM explosion and overfitting.

    min_df: int = 2
    # Ignore words appearing in fewer than 2 documents.
    # Removes typos and noisy tokens.

    max_df: float = 0.95
    # Ignore words appearing in >95% of documents.
    # Removes overly common tokens missed by stopword list.

    stop_words: Optional[str] = "english"
    # Remove common English stopwords (the, is, at...).
    # Boosts accuracy by discarding non-informative tokens.

    sublinear_tf: bool = True
    # Use log-scaled term frequency (1 + log(tf)).
    # Prevents very frequent words from dominating feature weights.

    strip_accents: Optional[str] = "unicode"
    # Normalize accented characters (résumé → resume).
    # Improves consistency across different encodings.

    # =========================
    # Scaling controls
    # =========================

    produce_scaled: bool = True
    # If True, also produce StandardScaler-normalized matrices.
    # Required for Logistic Regression and SVM stability.

    scaler_with_mean: bool = False
    # Must always be False for sparse TF-IDF matrices.
    # Setting this to True will densify matrices and crash RAM.

# -----------------------------
# Public API
# -----------------------------
"""
    Function to produce ready-to-use feature set
    End-to-end:
        - Load Fake.csv / True.csv
        - Create labels (fake=0, real=1)
        - Drop date column (not used)
        - Clean + optionally drop duplicates
        - Split train/test (stratified)
        - Fit TF-IDF (+ optional subject OHE) on train
        - Transform test
        - Produce unscaled + scaled sparse matrices
        - Save outputs + artifacts + metadata
"""
def build_and_save_features(
    fake_csv_path: str | os.PathLike,
    true_csv_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    config: FeatureConfig,
) -> None:
    # Validate the output paths and prepare them for storage at the end of the function execution
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    # Load the Fake News data set which consists of 2 CSV files one containing true news and the other containing fake news
    # The output will be a single dataframe containing concatenated true and false samples
    df = load_isot_dataset(fake_csv_path, true_csv_path)

    # Perform the Pre-processing and Clean up
    df = prepare_dataframe(df, config)

    # Split the dataset into labels vector and features matrix
    y = df["label"].to_numpy(dtype=int)

    # Perform Train/Test split with the configurable split size, random state and consider stratification to ensure split balance as well as shuffle
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
        shuffle=True,
    )

    y_train = train_df["label"].to_numpy(dtype=int)
    y_test  = test_df["label"].to_numpy(dtype=int)

    X_train_df = train_df.drop(columns=["label"])
    X_test_df  = test_df.drop(columns=["label"])

    # Fit/transform for the test and train datasets
    X_train_unscaled, X_train_scaled, artifacts, stats_train = fit_transform_train(X_train_df, y_train, config)
    X_test_unscaled, X_test_scaled, stats_test = transform_test(X_test_df, y_test, artifacts, config)

    # Save matrices
    save_npz(out / "X_train_unscaled.npz", X_train_unscaled)
    save_npz(out / "X_test_unscaled.npz", X_test_unscaled)

    if config.produce_scaled:
        save_npz(out / "X_train_scaled.npz", X_train_scaled)
        save_npz(out / "X_test_scaled.npz", X_test_scaled)

    # Save labels
    _save_labels_csv(out / "y_train.csv", y_train)
    _save_labels_csv(out / "y_test.csv", y_test)

    # Save artifacts
    dump(artifacts["tfidf"], art / "tfidf.joblib")
    if artifacts.get("subject_ohe") is not None:
        dump(artifacts["subject_ohe"], art / "subject_ohe.joblib")
    if artifacts.get("scaler") is not None:
        dump(artifacts["scaler"], art / "scaler.joblib")

    # Save config + split info + stats
    _save_json(art / "config.json", asdict(config))
    _save_json(art / "split_info.json", {
        "random_state": config.random_state,
        "test_size": config.test_size,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "stratified": True,
    })
    _save_json(art / "stats.json", {
        "train": stats_train,
        "test": stats_test,
        "final": {
            "X_train_unscaled_shape": [int(X_train_unscaled.shape[0]), int(X_train_unscaled.shape[1])],
            "X_test_unscaled_shape": [int(X_test_unscaled.shape[0]), int(X_test_unscaled.shape[1])],
            "scaled_outputs_written": bool(config.produce_scaled),
            "include_subject": bool(config.include_subject),
        }
    })


"""
    Load ready-to-use matrices and labels.

    Args:
        out_dir: Folder produced by build_and_save_features().
        scaled: If True, loads X_train_scaled.npz / X_test_scaled.npz.
                If False, loads unscaled.

    Returns:
        X_train, X_test, y_train, y_test
"""
def load_feature_matrices(
    out_dir: str | os.PathLike,
    scaled: bool = False,
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:

    out = Path(out_dir)
    xtr_path = out / ("X_train_scaled.npz" if scaled else "X_train_unscaled.npz")
    xte_path = out / ("X_test_scaled.npz" if scaled else "X_test_unscaled.npz")

    X_train = load_npz(xtr_path).tocsr()
    X_test = load_npz(xte_path).tocsr()

    y_train = pd.read_csv(out / "y_train.csv")["label"].to_numpy(dtype=int)
    y_test = pd.read_csv(out / "y_test.csv")["label"].to_numpy(dtype=int)
    return X_train, X_test, y_train, y_test

"""
    Load fitted feature artifacts for inference-time transformation.
"""
def load_artifacts(out_dir: str | os.PathLike) -> Dict[str, Any]:

    out = Path(out_dir)
    art = out / "artifacts"
    artifacts: Dict[str, Any] = {
        "tfidf": load(art / "tfidf.joblib"),
        "subject_ohe": None,
        "scaler": None,
    }
    if (art / "subject_ohe.joblib").exists():
        artifacts["subject_ohe"] = load(art / "subject_ohe.joblib")
    if (art / "scaler.joblib").exists():
        artifacts["scaler"] = load(art / "scaler.joblib")
    return artifacts

"""
    Transform raw records on the fly (deployment or tuning).

    Args:
        titles/texts: raw inputs
        subjects: optional list (required if include_subject=True)
        artifacts: loaded via load_artifacts()
        config: must match training config (especially include_subject)
        scaled: if True, apply scaler if available

    Returns:
        Sparse feature matrix (CSR)
"""
def transform_records(
    titles: list[str],
    texts: list[str],
    subjects: Optional[list[str]],
    artifacts: Dict[str, Any],
    config: FeatureConfig,
    scaled: bool = False,
) -> csr_matrix:

    if len(titles) != len(texts):
        raise ValueError("titles and texts must have the same length.")

    df = pd.DataFrame({"title": titles, "text": texts})
    if config.include_subject:
        if subjects is None:
            raise ValueError("subjects must be provided when include_subject=True.")
        if len(subjects) != len(titles):
            raise ValueError("subjects must have the same length as titles/texts.")
        df["subject"] = subjects

    df = _clean_df_for_inference(df, config)
    X_unscaled = _vectorize_df(df, artifacts, config)

    if scaled:
        scaler = artifacts.get("scaler")
        if scaler is None:
            raise ValueError("Scaled transform requested but scaler artifact is missing.")
        return scaler.transform(X_unscaled).tocsr()

    return X_unscaled

def load_welfake_external_eval(
    welfake_csv_path: str | os.PathLike,
    features_out_dir: str | os.PathLike,
    scaled: bool = True,
    limit: Optional[int] = None,
) -> Tuple[csr_matrix, np.ndarray]:
    """
    Load and transform the WELFake dataset using ISOT-trained feature artifacts.

    Args:
        welfake_csv_path: Path to WELFake_Dataset.csv (columns: title, text, label)
        features_out_dir: Folder generated by build_and_save_features()
        scaled: Apply StandardScaler (True for LR/SVM).
        limit: Optional number of rows to load.

    Returns:
        X_wel: Sparse feature matrix
        y_wel: Label vector (0=fake, 1=real)
    """

    df = pd.read_csv(welfake_csv_path)

    if limit is not None:
        df = df.head(int(limit)).copy()

    titles = df["title"].fillna("").astype(str).tolist()
    texts  = df["text"].fillna("").astype(str).tolist()
    y_wel  = df["label"].to_numpy(dtype=int)

    artifacts = load_artifacts(features_out_dir)
    cfg = FeatureConfig()

    X_wel = transform_records(
        titles=titles,
        texts=texts,
        subjects=None,
        artifacts=artifacts,
        config=cfg,
        scaled=scaled,
    )

    return X_wel, y_wel

# -----------------------------
# Internals
# -----------------------------
"""
    Load Fake.csv and True.csv and return a merged dataframe with label column.
    label: fake=0, real=1
"""
def load_isot_dataset(
    fake_csv_path: str | os.PathLike,
    true_csv_path: str | os.PathLike,
) -> pd.DataFrame:
    fake = pd.read_csv(fake_csv_path)
    true = pd.read_csv(true_csv_path)

    fake = fake.copy()
    true = true.copy()

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    return df


"""
    This function prepares the text that will be vectorized and fed to the models.
    It does the following:
        1. Ensure that the input data frame contains title and text columns
        2. Handle missing values by filling them with empty string to avoid vectorization crashes
        3. Drops the Date column if it exists and configured to drop it
        4. Combine the title and the text of each sample into a single column called 'content' to be used as single string for vectorization
        5. Apply string clean up (lowercase, URLs removal and whitespaces normalization)
        6. Keep only needed columns to be vectorized (content, label and subject if included)
        7. Drop any duplicated contents and subjects
        8. Drop any rows that became empty due to clean up
    Args:
        df: input data frame containing all dataset columns and label column
        config: Feature configuration object

    Returns:
        Data frame cleaned up and ready for vectorization and model feeding
"""
def prepare_dataframe(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    df = df.copy()

    # Ensure expected columns exist (Kaggle dataset typically has title/text/subject/date)
    for col in ["title", "text"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataset.")

    # Handle missing values by filling NaN values with empty string to avoid vectorization crashes
    if config.fill_missing_with_empty:
        df["title"] = df["title"].fillna("")
        df["text"] = df["text"].fillna("")
        if "subject" in df.columns:
            df["subject"] = df["subject"].fillna("")

    # Drop date if configured to do so
    if config.drop_date and "date" in df.columns:
        df = df.drop(columns=["date"])

    # Build combined text to be title + text which will be vectorized and fed to the models for training
    df["content"] = (df["title"].astype(str) + " " + df["text"].astype(str)).astype(str)

    # Clean the content column by applying the configured cleaning steps
    df["content"] = df["content"].apply(lambda s: _clean_text(s, config))

    # Keep only necessary columns. If subject is considered, keep its column to be fed to the model as well
    keep_cols = ["content", "label"]
    if config.include_subject:
        if "subject" not in df.columns:
            raise ValueError("include_subject=True but dataset has no 'subject' column.")
        keep_cols.insert(1, "subject")
    df = df[keep_cols]

    # Drop duplicates if any in the cleaned samples
    if config.drop_exact_duplicates:
        subset = ["content"] + (["subject"] if config.include_subject else [])
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
        after = len(df)
        # (stats saved later; no prints in library code)

    # Drop rows that became empty due to cleaning
    df = df[df["content"].str.len() > 0].reset_index(drop=True)

    return df


"""
    Applies TF-IDF vectorization (+ optional subject OHE) and scaler on training data.

    Args:
        train_df: input data frame containing the cleaned up content (and subject if included).
                  NOTE: train_df must NOT contain the label column.
        y_train: label vector for the training set (fake=0, real=1).
        config: Feature configuration object

    Returns:
        Returns unscaled/scaled matrices plus artifacts and stats.
"""
def fit_transform_train(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    config: FeatureConfig,
) -> Tuple[csr_matrix, csr_matrix, Dict[str, Any], Dict[str, Any]]:

    # Creates dictionary to contain objects for the vectorization process for reproducability purposes
    artifacts: Dict[str, Any] = {"tfidf": None, "subject_ohe": None, "scaler": None}

    # Create TF-IDF vectorization object according to the configurations
    tfidf = TfidfVectorizer(
        lowercase=False,  # already handled in _clean_text()
        ngram_range=(config.ngram_min, config.ngram_max),
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        stop_words=config.stop_words,
        sublinear_tf=config.sublinear_tf,
        strip_accents=config.strip_accents,
    )

    # Perform vectorization on the content column and store it in a numeric matrix
    X_text = tfidf.fit_transform(train_df["content"].astype(str).tolist()).tocsr()

    # Store the vectorization object for reproducability purposes
    artifacts["tfidf"] = tfidf

    # If subject is included, perform one-hot encoding for its possible values and add it to the feature matrix X
    X = X_text
    if config.include_subject:
        if "subject" not in train_df.columns:
            raise ValueError("include_subject=True but 'subject' column is missing in train_df.")

        # NOTE: Using sparse=True for wider scikit-learn compatibility
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
        X_subj = ohe.fit_transform(train_df[["subject"]].astype(str))
        artifacts["subject_ohe"] = ohe
        X = hstack([X_text, X_subj], format="csr")

    # If scaling is configured (Mandatory for LR and SVM), perform it on the feature matrix using StandardScaler
    X_scaled = X
    scaler = None
    if config.produce_scaled:
        scaler = StandardScaler(with_mean=config.scaler_with_mean)
        X_scaled = scaler.fit_transform(X).tocsr()

    # Store scaler object for reproducability purposes
    artifacts["scaler"] = scaler

    # Obtain statistics to be reported for analysis purposes
    y_train = np.asarray(y_train, dtype=int)
    stats = {
        "n_samples": int(X.shape[0]),
        "n_features_text": int(X_text.shape[1]),
        "include_subject": bool(config.include_subject),
        "n_features_total": int(X.shape[1]),
        "scaled": bool(config.produce_scaled),
        "class_counts": _class_counts(y_train),
    }

    # Return the un-scaled feature matrix X, the scaled feature matrix X, the artifacts objects and the statistics
    return X.tocsr(), X_scaled.tocsr(), artifacts, stats


"""
    Applies TF-IDF vectorization (+ optional subject OHE) and scaler on test data using the objects fit on the training set to avoid data leakage.

    Args:
        test_df: input data frame containing the cleaned up content (and subject if included) for the test set.
                 NOTE: test_df must NOT contain the label column.
        y_test: label vector for the test set (fake=0, real=1).
        artifacts: objects fit on the training set to be used for vectorizing the test set
        config: Feature configuration object

    Returns:
        Returns unscaled/scaled matrices plus stats.
"""
def transform_test(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    artifacts: Dict[str, Any],
    config: FeatureConfig,
) -> Tuple[csr_matrix, csr_matrix, Dict[str, Any]]:

    # Transform test data using fitted artifacts.
    X_unscaled = _vectorize_df(test_df, artifacts, config).tocsr()
    X_scaled = X_unscaled

    # Perform scaling to the test set if configured
    if config.produce_scaled:
        scaler = artifacts.get("scaler")
        if scaler is None:
            raise ValueError("produce_scaled=True but scaler artifact is missing.")
        X_scaled = scaler.transform(X_unscaled).tocsr()

    # Obtain statistics to be reported for analysis purposes
    y_test = np.asarray(y_test, dtype=int)
    stats = {
        "n_samples": int(X_unscaled.shape[0]),
        "n_features_total": int(X_unscaled.shape[1]),
        "include_subject": bool(config.include_subject),
        "scaled": bool(config.produce_scaled),
        "class_counts": _class_counts(y_test),
    }

    # Return the un-scaled feature matrix X, the scaled feature matrix X and the statistics for the test set
    return X_unscaled.tocsr(), X_scaled.tocsr(), stats


"""
    Applies TF-IDF vectorization (+ optional subject OHE) test data using the objects fit on the training set to avoid data leakage

    Args:
        test_df: input data frame containing the cleaned up content (and subject of included) for the test set
        artifacts: objects fit on the training set to be used for vectorizing the training set
        config: Feature configuration object

    Returns:
        Returns vectorized matrices plus artifacts.
"""
def _vectorize_df(df: pd.DataFrame, artifacts: Dict[str, Any], config: FeatureConfig) -> csr_matrix:
    tfidf: TfidfVectorizer = artifacts["tfidf"]
    X_text = tfidf.transform(df["content"].astype(str).tolist()).tocsr()

    X = X_text
    if config.include_subject:
        ohe: OneHotEncoder = artifacts.get("subject_ohe")
        if ohe is None:
            raise ValueError("include_subject=True but subject_ohe artifact is missing.")
        if "subject" not in df.columns:
            raise ValueError("include_subject=True but 'subject' column is missing in input df.")
        X_subj = ohe.transform(df[["subject"]].astype(str))
        X = hstack([X_text, X_subj], format="csr")

    return X.tocsr()


"""
    Prepares raw inference-time input records for feature transformation.

    This function performs the same preprocessing steps used during training to
    guarantee feature consistency between training and deployment as follows:
        1- Fills missing title/text/subject values with empty strings (if enabled)
        2- Concatenates `title` and `text` into a single `content` field
        3- Applies string cleanup based on FeatureConfig:
            - lowercase transformation
            - URL and email removal
            - whitespace normalization
        4- Keeps only the columns required by the vectorizer:
            - `content`
            - optionally `subject` if include_subject=True

    Args:
        df: Raw input dataframe containing title, text and optionally subject columns.
        config: Feature configuration object controlling cleaning behavior.

    Returns:
        DataFrame containing only the cleaned `content` column and optional `subject`
        column, ready for vectorization.
"""
def _clean_df_for_inference(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    df = df.copy()
    if config.fill_missing_with_empty:
        df["title"] = df["title"].fillna("")
        df["text"] = df["text"].fillna("")
        if config.include_subject and "subject" in df.columns:
            df["subject"] = df["subject"].fillna("")
    df["content"] = (df["title"].astype(str) + " " + df["text"].astype(str)).astype(str)
    df["content"] = df["content"].apply(lambda s: _clean_text(s, config))
    # Keep what vectorizer needs
    keep = ["content"] + (["subject"] if config.include_subject and "subject" in df.columns else [])
    return df[keep]

"""
    Regular expression used to detect and remove URLs and email addresses from text.
    This prevents data leakage where the model learns publisher identity or source domains
    instead of actual linguistic deception patterns in fake news articles.
    Example matches:
        - https://cnn.com/news
        - www.reuters.com
        - editor@fakewebsite.org
"""
_URL_EMAIL_RE = re.compile(
    r"(?i)\b((?:https?://|www\.)\S+|[\w\.-]+@[\w\.-]+\.\w+)\b"
)

"""
    Performs string clean up according to the configurations in config object as follows:
        1- Transforms the string from uppercase to lowercase
        2- Removes any URLs
        3- Reduces repeated whitespaces into a single whitespace
    Args:
        text: string text to be cleaned
        config: Feature configuration object

    Returns:
        Cleaned string ready to be vectorized
"""
def _clean_text(text: str, config: FeatureConfig) -> str:
    # Create a shadow copy of the text to be cleaned as string
    s = str(text)

    # Transform the string from uppercase to lowercase
    if config.lowercase:
        s = s.lower()

    # Removes any URLs
    if config.remove_urls_emails:
        s = _URL_EMAIL_RE.sub(" ", s)

    # Reduces repeated whitespaces into a single whitespace
    if config.normalize_whitespace:
        s = re.sub(r"\s+", " ", s).strip()

    return s

"""
    Count the number of the classes (fake=0, real=1) in the label column

    Args:
        y: label vector

    Returns:
        dictionary containing the number of fake and real news
"""
def _class_counts(y: np.ndarray) -> Dict[str, int]:
    # fake=0 real=1
    y = y.astype(int)
    return {
        "fake_0": int(np.sum(y == 0)),
        "real_1": int(np.sum(y == 1)),
    }


def _save_labels_csv(path: Path, y: np.ndarray) -> None:
    pd.DataFrame({"label": y.astype(int)}).to_csv(path, index=False)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
