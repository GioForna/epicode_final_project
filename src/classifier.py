"""Classificazione della coda: confronto tra modelli e salvataggio del migliore."""
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.data import load_split


def tfidf(max_features=50_000):
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=max_features,
        sublinear_tf=True, stop_words="english",
    )


def candidate_models():
    rs = config.RANDOM_STATE
    return {
        "Baseline (classe piu' frequente)": Pipeline(
            [("tfidf", tfidf()), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", ComplementNB(alpha=0.3))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(
            C=5, max_iter=2000, class_weight="balanced"))]),
        "Linear SVM": Pipeline([("tfidf", tfidf()), ("clf", CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced"), cv=3))]),
        "Rete neurale (MLP)": Pipeline([("tfidf", tfidf(20_000)), ("clf", MLPClassifier(
            hidden_layer_sizes=(256,), early_stopping=True, max_iter=30,
            random_state=rs))]),
    }


def compare_models(train, test):
    rows, fitted = [], {}
    for name, model in candidate_models().items():
        t0 = time.time()
        model.fit(train["text"], train["queue"])
        pred = model.predict(test["text"])
        rows.append({
            "modello": name,
            "accuracy": accuracy_score(test["queue"], pred),
            "f1_macro": f1_score(test["queue"], pred, average="macro"),
            "f1_weighted": f1_score(test["queue"], pred, average="weighted"),
            "tempo_training_s": round(time.time() - t0, 1),
        })
        fitted[name] = model
        print(f"{name:35s} acc={rows[-1]['accuracy']:.3f} f1_macro={rows[-1]['f1_macro']:.3f}")
    results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return results, fitted


def plot_confusion(model, test, path):
    labels = sorted(test["queue"].unique())
    cm = confusion_matrix(test["queue"], model.predict(test["text"]), labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Coda predetta")
    ax.set_ylabel("Coda reale")
    ax.set_title("Matrice di confusione (normalizzata per riga)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def threshold_analysis(model, test):
    """Per ogni soglia: quota di ticket smistati in automatico e accuracy su quelli."""
    proba = model.predict_proba(test["text"])
    conf = proba.max(axis=1)
    pred = model.classes_[proba.argmax(axis=1)]
    correct = pred == test["queue"].to_numpy()
    rows = []
    for t in np.arange(0.0, 0.95, 0.05):
        mask = conf >= t
        rows.append({
            "soglia": round(t, 2),
            "copertura_auto": mask.mean(),
            "accuracy_auto": correct[mask].mean() if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def plot_threshold(th, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(th["soglia"], th["copertura_auto"], marker="o", label="Ticket smistati in automatico")
    ax.plot(th["soglia"], th["accuracy_auto"], marker="o", label="Accuracy sui ticket smistati")
    ax.axvline(config.CONFIDENCE_THRESHOLD, color="grey", ls="--", label="Soglia scelta")
    ax.set_xlabel("Soglia di confidenza")
    ax.set_ylim(0, 1.02)
    ax.set_title("Guardrail: compromesso copertura / accuratezza")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def main():
    train, test = load_split()
    results, fitted = compare_models(train, test)
    results.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    best_name = results.iloc[0]["modello"]
    best = fitted[best_name]
    joblib.dump(best, config.CLASSIFIER_FILE)
    print("Miglior modello:", best_name)

    report = classification_report(test["queue"], best.predict(test["text"]), output_dict=True)
    pd.DataFrame(report).T.to_csv(config.REPORTS_DIR / "classification_report.csv")
    plot_confusion(best, test, config.FIG_DIR / "confusion_matrix.png")

    th = threshold_analysis(best, test)
    th.to_csv(config.REPORTS_DIR / "threshold_analysis.csv", index=False)
    plot_threshold(th, config.FIG_DIR / "threshold_tradeoff.png")

    summary = {"best_model": best_name, **results.iloc[0].drop("modello").to_dict()}
    (config.REPORTS_DIR / "classifier_summary.json").write_text(json.dumps(summary, indent=2))
    return results, best, th


if __name__ == "__main__":
    main()
