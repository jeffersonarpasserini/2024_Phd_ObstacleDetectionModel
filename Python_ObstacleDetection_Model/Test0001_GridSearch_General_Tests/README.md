# Friedman + Nemenyi Rank Evaluation Framework

This repository provides a complete statistical framework to **evaluate and rank machine learning models** using non-parametric tests. It is especially useful when model results may not follow a normal distribution or when comparing multiple models across multiple executions.

---

## 📌 Objective

To fairly compare classification models using:

- ✅ Global **normality testing** (D’Agostino and Pearson test)
- ✅ **Friedman test** for repeated-measures comparison across models
- ✅ **Post-hoc Nemenyi test** to identify statistically significant differences
- ✅ Computation of **weighted performance scores**

---

## 📁 Outputs Explained

After running the analysis, the following files are generated:

### `friedman_ranking.csv`
Detailed results for each model execution including:

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Weighted Score | Friedman Rank |
|-----------|----------|-----------|--------|----------|---------|----------------|----------------|
| Model A   | 0.85     | 0.80      | 0.90   | 0.85     | 0.88    | 0.872          | 1.0            |

- **Weighted Score**: Composite metric calculated with user-defined weights.
- **Friedman Rank**: Mean rank of each model across multiple runs (lower is better).
- Ordered from **best to worst**.

---

### `friedman_ranks.csv`
Final average Friedman ranking:

| Algorithm | Friedman Rank |
|-----------|----------------|
| Model A   | 1.0            |
| Model B   | 2.2            |
| Model C   | 3.1            |

- Ordered from best to worst models.

---

### `nemenyi_results.csv`
Pairwise significance comparison between models after the Friedman test.

|         | Model A | Model B | Model C |
|---------|---------|---------|---------|
| Model A | 1.0     | 0.03    | 0.001   |
| Model B | 0.03    | 1.0     | 0.05    |
| Model C | 0.001   | 0.05    | 1.0     |

- Values < 0.05 suggest statistically significant differences between two models.

---

### `friedman_stats.txt`
Raw output of the Friedman test:

```
Friedman Test: X² = 1614.57600000, p-value = 0.00000000
```

---

### `normality_test_results.txt`
Result of the global normality test for the selected metric (usually Accuracy):

```
Global Accuracy: stat = 168123.8932, p-value = 0.00000000
```

---

### `normality_distribution_global.png`
Density plot showing the global distribution of the selected metric.

- Helps determine if the distribution is normal.
- Automatically saved if `enable_plots = True`.

---

## ⚙️ Main Features

- ✅ Flexible integration with GridSearchCV outputs.
- ✅ Adjustable metric weights (default config):

```python
weights = {
    'Accuracy': 0.15,
    'Precision': 0.15,
    'Recall': 0.30,
    'F1-Score': 0.25,
    'ROC-AUC': 0.15
}
```

- ✅ Optional **median adjustment** (removing outliers) if data is not normally distributed.
- ✅ Visualizations and execution logs for all key steps.
- ✅ Progress bars to monitor long operations.

---

## 📦 Requirements

Install with:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn scikit-posthocs tqdm
```

---

## ▶️ How to Run

Update the script with your dataset path:

```python
ranker = AlgorithmRanker("your_gridsearch_results.csv", enable_plots=True)
ranker.read_gridsearch_results()
...
```

Then run:

```bash
python Statistical_Tests_Test001.py
```

---

## 📬 Questions?

Feel free to open an issue or contact the repository maintainer if you need help integrating this with your experiment pipeline.

---

