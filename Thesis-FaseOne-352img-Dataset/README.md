# Phd-Partial-Results — Experimental Results

Partial experimental results of the doctoral research on **obstacle detection for the navigational aid of visually impaired people**, supporting the paper:

> **Cross-Analysis of CNN Architectures for Navigational Aid of the Visually Impaired** — J. A. R. Passerini, F. Breve.

This repository contains the **raw per-fold results, statistical analyses (Friedman / Kruskal-Wallis), and spreadsheets** produced in the study. It corresponds to the stage in which convolutional neural networks (CNNs) are used **only as feature extractors** (via transfer learning, without fine-tuning), combined with dimensionality reduction / feature selection and supervised classifiers, and compared against a fine-tuned reference model.

## Study at a glance

- **Dataset:** VIA dataset — 342 images (175 *clear path*, 167 *obstructed path*).
- **Feature extractors:** 25 CNN architectures (EfficientNet B0–B7, ResNet, DenseNet, VGG, Inception, Xception, MobileNet, NASNetMobile).
- **Dimensionality reduction / selection:** PCA, UMAP, Relief-F (vector sizes 2, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300), plus the *full* (no-reduction) option.
- **Classifiers (8):** SVM (linear), SVM (RBF), MLP, Logistic Regression, Random Forest, AdaBoost, Decision Tree, Gaussian Naïve Bayes.
- **Validation:** 10-fold cross-validation.
- **Statistical tests:** Friedman (primary; ranking) and Kruskal-Wallis (legacy omnibus). Software: IBM SPSS Statistics 20.

## Repository structure

```
.
├── Referencia/      # Reference model: 22 fine-tuned CNNs (baseline)
├── Abordagem01/     # Approach A — full features (no reduction)        | 272  combinations
├── Abordagem02/     # Approach B — single CNN + reduction/selection    | 7200 combinations
├── Abordagem03/     # Approach C — concatenate CNNs, then reduce        | 2592 combinations
└── Abordagem04/     # Approach D — reduce each CNN, then concatenate    | 2592 combinations
```

Each folder contains three subfolders:

| Subfolder | Content |
|-----------|---------|
| `Dados/` (or `DADOS/`) | Raw result tables in **CSV** (per-fold accuracy and detailed metrics). |
| `Excel/` (or `EXCEL/`) | Analysis spreadsheets (`.xls` / `.xlsx`). |
| `spss/` (or `SPSS/`) | SPSS data files (`.sav`) and statistical outputs (`.spv`) for the Friedman and Kruskal-Wallis tests. |

For Approaches B, C and D, the `Dados/`, `Excel/` and `SPSS/` folders are further split into:
- `Analise_CNN/` — statistics computed **per individual CNN** (Approach B) or **per CNN pair** (Approaches C, D), used to select the best technique for each extractor;
- `Analise_Final/` — the **final cross-extractor comparison** that selects the best technique of the approach.

### Folder ↔ paper mapping

| Folder | Paper | Description |
|--------|-------|-------------|
| `Referencia` | Reference model | 22 CNNs with added dense layers, fine-tuned on the VIA dataset (best: DenseNet201). |
| `Abordagem01` | **Approach A** | Full CNN feature vector (no reduction) fed directly to the classifiers. |
| `Abordagem02` | **Approach B** | Single CNN + PCA / UMAP / Relief-F, then classifier. |
| `Abordagem03` | **Approach C** | Concatenate the feature vectors of two CNNs, **then** reduce, then classify. |
| `Abordagem04` | **Approach D** | Reduce each CNN vector **first**, then concatenate, then classify. |

## File types and naming conventions

- **`data_detailed_*.csv`** — one row per *(model, fold)* with the detailed metrics (see schema below).
- **`acc_kfold_resume_*.csv`** — accuracy matrix: one row per model, one column per fold.
- **`donald.csv`** (reference) — per-fold accuracy of each of the 22 CNNs.
- **`*friedman*` / `Friedman_*` / `Kruskal_*`** — prepared data and outputs of the statistical tests.
- **`.sav`** = SPSS dataset, **`.spv`** = SPSS output viewer file.

> **Locale note.** CSV files use **`;`** as the column separator and **`,`** as the decimal mark (Portuguese locale). The suffix `(virgula)` in a filename indicates comma-decimal formatting, and `semPCC` means "without the PCC column".

### Schema of `data_detailed_*.csv`

| Column | Meaning |
|--------|---------|
| `ExtractionMethod` / `ExtrMeth_abr` | CNN (or CNN combination) used as extractor, and its abbreviation. |
| `Reduction` | `Full`, `PCA`, `UMAP` or `ReliefF`. |
| `Classification` / `Class_abr` | Classifier and its abbreviation. |
| `Metodo` / `MetodoABR` | Full technique identifier (extractor + reduction + classifier). |
| `Fold` | Fold index (1–10). |
| `Features` / `Components` | Input feature-vector size / number of components kept. |
| `Normalization` | Whether feature normalization was applied. |
| `ACC` | Accuracy. |
| `F1` | F1-score. |
| `ROC` | ROC-AUC. |
| `ExtractionTime` | Feature-extraction time (s). |
| `ReductionTime` | Dimensionality-reduction time (s). |
| `ClassfTrainning` | Classifier training time (s). |
| `ClassfPredict` | Classifier inference time (s). |

> **Important:** the full schema above (with `F1`, `ROC` and timing columns) is available for **Approach A (`Abordagem01`)**. For Approaches B, C and D, the `data_detailed_*.csv` files report **accuracy (`ACC`) only**. In `data_detailed`, `ACC` / `F1` / `ROC` are expressed as **percentages** (e.g., `94.12`); in `acc_kfold_resume_*` and `donald.csv` they are expressed as **fractions** (e.g., `0.9412`).

## Headline results (10-fold cross-validation)

| Model | Approach | Median accuracy |
|-------|----------|-----------------|
| DenseNet201 | Reference | 0.9412 |
| EfficientNetB0 + MobileNet + linear SVM | A | 0.9412 |
| MobileNet + PCA(40) + SVM (RBF) | B | 0.9421 |
| MobileNet + ResNet50 + PCA(300) + linear SVM | C | **0.9559** |
| MobileNet + ResNet50 + PCA(100) + SVM (RBF) | D | 0.9278 |

A Friedman test followed by the Nemenyi post-hoc found **no statistically significant difference** among the four approaches and the reference model, showing that the proposed approaches match the fine-tuned baseline **without training any CNN**.

## How to use these files

- The CSV tables can be opened with any spreadsheet or loaded in Python/R. In Python, read them with the correct locale, e.g.:

  ```python
  import pandas as pd
  df = pd.read_csv("Abordagem01/Dados/data_detailed_full (virgula)_semPCC.csv",
                   sep=";", decimal=",")
  ```

- The `.spv` files require IBM SPSS Statistics (or the free *SPSS Smartreader*) to be opened; the `.sav` files can also be read with `pyreadstat` or `pandas.read_spss`.

## Citation

If you use these results, please cite the corresponding paper and this repository. A formal citation entry will be added upon publication.

## Contact

Jefferson Antonio Ribeiro Passerini — jefferson.passerini@unesp.br
São Paulo State University (UNESP), Institute of Geosciences and Exact Sciences, Rio Claro, SP, Brazil.
