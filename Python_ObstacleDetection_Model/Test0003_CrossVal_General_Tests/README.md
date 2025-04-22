# 🧠 Statistical Evaluation and Model Selection – MobileNet Classification

## 📋 Overview

This project aims to implement a **robust model selection process for image classification** using variations of the **MobileNet architecture**. The evaluation considers multiple performance metrics, with special attention to a **custom weighted metric** designed to balance sensitivity and precision.

---

## ⚙️ Steps Performed

### 1. `GPU_Test003_CrossValidation.py`
- Performs **stratified k-fold cross-validation (k=10)**.
- Evaluates hundreds of MobileNet configurations (V1).
- Computes five main metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Also computes a **custom Weighted Score** with the following weights:
  - Accuracy: 15%
  - Precision: 15%
  - Recall: 30%
  - F1-Score: 25%
  - ROC-AUC: 15%
- Saves both raw and summarized results for statistical analysis.

---

### 2. `Prepared_Data_Friedman_Tests.py`
- Receives cross-validation results as input.
- Allows choosing any metric (`Accuracy`, `Recall`, `Weighted`, etc.).
- Prepares a dataset formatted for statistical tests (one model per row, folds as columns).
- Output: `friedman_prepared_data_<metric>.csv`

---

### 3. `Statistical_Tests_Test0003.py`
- Runs the **Friedman Test** to compare all models across folds.
- If p-value < 0.05, performs **post-hoc Nemenyi Test**.
- Outputs:
  - `<metric>_friedman_ranks.csv`
  - `<metric>_friedman_stats.txt`
  - `<metric>_nemenyi_results.csv`

---

### 4. `Select_Statistical_Similar_Models.py`
- Selects the **Top-N models** with the lowest Friedman rank (e.g., top 20).
- Keeps only models **statistically equivalent** to the top model according to the Nemenyi test (p ≥ 0.05).
- Output: `<metric>_modelos_selecionados.csv`

---

### 5. `TopN_Models_Nemenyi_tests_matrix.py`
- Generates a **cross-comparison matrix** of the Top-N selected models with **Nemenyi test p-values**.
- Useful for identifying statistically similar model groups.
- Output: `<metric>_top20_nemenyi_matrix.csv`

---

### 6. `Accuracy_Weighted_Compared_Models_Friedman_Tests.py`
- Compares models selected by two different metrics:
  - `Accuracy`
  - `Weighted Score`
- Generates a CSV comparing Friedman ranks:
  - `accuracy_vs_weighted_compared_friedman_tests.csv`

---

## 🔍 Key Results

- **390 models** were evaluated.
- **Friedman Test** revealed statistically significant differences (p < 0.0001).
- **22 final models** were selected:
  - 18 appeared in both selections (Accuracy and Weighted)
  - 2 were **exclusive to the Weighted metric**, showing it better captures overall performance.
- The weighted metric favored models with **higher recall and F1-Score**, even if their accuracy wasn’t the absolute highest.

---

## ✅ Conclusion

The adopted strategy:
- Uses a **statistically robust evaluation process**
- Ensures selected models are **not significantly different** from each other
- Considers more than accuracy alone, better reflecting the project’s goals

The final selection of 22 models provides a solid foundation for:
- Fine-tuning hyperparameters
- Training on new datasets
- Building ensemble models or preparing for deployment

---

## 📁 Output File Structure

```
📂
├── MobileNetV1_crossval_results_detail.csv
├── friedman_prepared_data_weighted.csv
├── weighted_friedman_ranks.csv
├── weighted_friedman_stats.txt
├── weighted_nemenyi_results.csv
├── weighted_modelos_selecionados.csv
├── weighted_top20_nemenyi_matrix.csv
├── accuracy_modelos_selecionados.csv
├── accuracy_friedman_ranks.csv
├── accuracy_nemenyi_results.csv
├── accuracy_vs_weighted_compared_friedman_tests.csv
```

---

© Research Project – MobileNet Image Classification with Statistical Analysis
