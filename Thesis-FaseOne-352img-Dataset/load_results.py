#!/usr/bin/env python3
"""
load_results.py
===============

Loads the experimental result tables of the *Phd-Partial-Results* repository and
reproduces the headline median-accuracy table reported in the paper
"Cross-Analysis of CNN Architectures for Navigational Aid of the Visually Impaired".

Locale / encoding notes
-----------------------
* All result CSV files use ``;`` as the column separator and ``,`` as the decimal
  mark (Portuguese locale).
* In the ``data_detailed`` files of **Approach A** the metrics (ACC/F1/ROC) are
  stored as **percentages** (e.g. ``94.12``); in the other files they are stored
  as **fractions** (e.g. ``0.9412``). The helper :func:`as_fraction` normalises both.

Usage
-----
    python load_results.py
"""
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_csv(path):
    """Read a results CSV using the repository locale (';' sep, ',' decimal)."""
    return pd.read_csv(path, sep=";", decimal=",", encoding="latin-1")


def find(pattern):
    """Return the first file matching ``pattern`` (relative to the repo root)."""
    matches = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matches: {pattern}")
    return matches[0]


def as_fraction(series):
    """Return accuracy on a 0-1 scale, dividing by 100 if values are percentages."""
    s = pd.to_numeric(series, errors="coerce")
    return s / 100.0 if s.median() > 1.5 else s


def median_acc(path, mask, col="ACC"):
    """Median accuracy of the rows of ``path`` selected by the boolean ``mask``."""
    df = load_csv(path)
    sub = df[mask(df)]
    return as_fraction(sub[col]).median(), len(sub)


def main():
    results = []

    # --- Reference model: DenseNet201 (donald.csv has columns CNN; ACC1..ACC10) ---
    ref = load_csv(find("Referencia/DADOS/donald.csv"))
    row = ref[ref["CNN"].astype(str).str.strip() == "DenseNet201"].iloc[0]
    accs = as_fraction(pd.Series([row[f"ACC{i}"] for i in range(1, 11)]))
    results.append(("Reference: DenseNet201", accs.median(), 10))

    # --- Approach A: EfficientNetB0+MobileNet + linear SVM (full features) ---
    results.append(("Approach A: EfficientNetB0+MobileNet + linear SVM",
                    *median_acc(
                        find("Abordagem01/Dados/*data_detailed*full*.csv"),
                        lambda d: (d["ExtractionMethod"] == "EfficientNetB0+MobileNet")
                                  & (d["Classification"] == "LinearSVM"))))

    # --- Approach B: MobileNet + PCA(40) + SVM (RBF) ---
    results.append(("Approach B: MobileNet + PCA(40) + SVM (RBF)",
                    *median_acc(
                        find("Abordagem02/Dados/*data_detailed*.csv"),
                        lambda d: (d["ExtractionMethod"] == "MobileNet")
                                  & (d["Reduction"] == "PCA")
                                  & (d["Components"].astype(str).str.strip() == "40")
                                  & (d["Classification"] == "RBF"))))

    # --- Approach C: MobileNet+ResNet50 + PCA(300) + linear SVM ---
    results.append(("Approach C: MobileNet+ResNet50 + PCA(300) + linear SVM",
                    *median_acc(
                        find("Abordagem03/Dados/*data_detailed*.csv"),
                        lambda d: (d["ExtractionMethod"] == "MobileNet+ResNet50")
                                  & (d["Reduction"] == "PCA")
                                  & (d["Components"].astype(str).str.strip() == "300")
                                  & (d["Classification"] == "LinearSVM"))))

    # --- Approach D: MobileNet+ResNet50 + PCA(100) + SVM (RBF) ---
    results.append(("Approach D: MobileNet+ResNet50 + PCA(100) + SVM (RBF)",
                    *median_acc(
                        find("Abordagem04/Dados/*data_detailed*.csv"),
                        lambda d: d["ExtractionMethod"].astype(str).str.contains("MobileNet\\+ResNet50")
                                  & (d["Reduction"] == "PCA")
                                  & (d["Components"].astype(str).str.strip() == "100")
                                  & (d["Classification"] == "RBF"))))

    # --- report ---
    print("\nMedian accuracy (10-fold cross-validation)")
    print("-" * 66)
    print(f"{'Acc.':>7}  {'folds':>5}  Model")
    print("-" * 66)
    for name, acc, n in results:
        print(f"{acc:7.4f}  {n:>5}  {name}")
    print("-" * 66)
    print("Expected: Ref 0.9412 | A 0.9412 | B 0.9421 | C 0.9559 | D 0.9277\n")


if __name__ == "__main__":
    main()
