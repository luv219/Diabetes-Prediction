# Diabetes Prediction — End‑to‑End (Scikit‑learn)

This repository contains a single Jupyter notebook, **`Diabetes_Prediction_EndToEnd.ipynb`**, that walks through a complete,
reproducible machine‑learning workflow to predict diabetes from tabular clinical measurements (Pima Indians Diabetes dataset).

**What you get end‑to‑end**

- Automated dataset download (with header‑fix when needed) and a local cached copy under `data/diabetes.csv`
- Exploratory Data Analysis (EDA)
- Data cleaning (treating biologically impossible zeros as missing)
- Train/validation split with stratification
- A **scikit‑learn Pipeline** (impute → scale → classifier)
- Comparison of multiple models with **cross‑validated ROC‑AUC**
- Final model training and evaluation on the held‑out test set
- Curves: ROC and (optionally) Precision‑Recall
- **Permutation feature importance**
- **Model persistence** to `artifacts/` via `joblib`
- Example single‑row prediction with `predict_proba`

**Data sources used in the notebook**

- https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
- https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv


**Notebook outline**

1. 🩺 Diabetes Prediction from Health Data — End‑to‑End Notebook
2. 📥 Load Dataset
3. 🔎 Exploratory Data Analysis
4. 🧼 Data Cleaning & Preprocessing
5. ✂️ Train / Test Split
6. 🧪 Models & Pipelines
7. 🏁 Train Best Model & Evaluate on Test Set
8. 📈 ROC & Precision‑Recall Curves
9. 🔍 Feature Importance (Permutation)
10. 💾 Save Trained Pipeline
11. 🔮 Inference Helper
12. 📚 Data Dictionary

**Models compared**: GradientBoostingClassifier, LogisticRegression, RandomForestClassifier.

**Metrics / plots**: auc, classification_report, confusion_matrix, precision_recall_curve, precision_score, roc_auc_score, roc_curve.

## Requirements

- Python 3.10+ (tested with Python 3.12)
- Jupyter Lab or VS Code (with Python & Jupyter extensions)
- Key libraries:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `joblib`

## Quick Start

> **Windows 10/11 (CMD or PowerShell)**

1) **Create a virtual environment (optional but recommended)**

```bat
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# or CMD
.\.venv\Scripts\activate.bat
```

2) **Install packages** (minimal set)

```bat
pip install -U pip
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

3) **Launch Jupyter** and open the notebook

```bat
jupyter lab
# or
jupyter notebook
```

4) **Run the notebook** from start to finish (`Run All`)
   - It will download the dataset (once) into `data/diabetes.csv`
   - Trained pipeline is saved under `artifacts/` (e.g., `diabetes_model_logisticregression.joblib`)

## Project Layout (created at runtime)

```
.
├─ Diabetes_Prediction_EndToEnd.ipynb   # The notebook
├─ data/
│  └─ diabetes.csv                      # Auto-downloaded dataset cache
└─ artifacts/
   └─ diabetes_model_<model>.joblib     # Saved scikit-learn pipeline
```

## Using the Saved Model

Once the notebook runs, you will find a `.joblib` pipeline inside `artifacts/`. Load and use it like this:

```python
import joblib
import pandas as pd

pipe = joblib.load("artifacts/diabetes_model_logisticregression.joblib")  # or the best model you saved

# Example row (match the notebook's feature order)
sample = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 79,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 35
}
df = pd.DataFrame([sample])
proba = pipe.predict_proba(df)[0, 1]
pred  = int(proba >= 0.5)
print("Probability:", round(proba, 3), "Pred:", pred)
```

## Notes & Assumptions

- The dataset is the well-known *Pima Indians Diabetes* dataset. Some public mirrors omit headers; the notebook fixes headers when needed.
- Several physiological features (e.g., `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) can contain zeros that are not biologically meaningful;
  the notebook treats them as missing and imputes medians before scaling.
- Cross-validation uses stratified folds to preserve class balance.
- Random seeds are fixed where applicable for reproducibility.
- Feature importance is computed via permutation importance on the trained pipeline.
- All processing is encapsulated inside **scikit-learn Pipelines**, so the saved model reproduces preprocessing at inference time.

## Acknowledgments

- Dataset mirrors referenced in the notebook:
  - Plotly datasets: `diabetes.csv`
  - Jason Brownlee (Machine Learning Mastery) mirror: `pima-indians-diabetes.data.csv`
