# Datheon — Machine Learning & Data Science Internship

**Four end-to-end machine learning projects | November–December 2026**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/) [![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/) [![Projects](https://img.shields.io/badge/Projects-4%20Completed-success)](#projects-at-a-glance)

A portfolio of four practical, independently implemented projects completed as part of a project-based **Machine Learning & Data Science internship with Datheon**. The repository demonstrates the full experimental workflow: dataset inspection, exploratory data analysis (EDA), preprocessing, feature engineering, training, validation, performance analysis, visualization, and model persistence.

> **Scope:** These are educational, reproducible machine learning experiments—not production-deployed applications. Reported scores refer to the specific datasets and evaluation procedures documented below.

## Table of contents

- [Internship overview](#internship-overview)
- [Projects at a glance](#projects-at-a-glance)
- [Task 01 — Iris classification](#task-01--iris-multi-class-classification)
- [Task 02 — Housing price prediction](#task-02--california-housing-price-prediction)
- [Task 03 — Customer churn prediction](#task-03--telecom-customer-churn-prediction)
- [Task 04 — NLP sentiment classification](#task-04--nlp-sentiment-classification)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Reproducibility and data management](#reproducibility-and-data-management)
- [Skills and lessons learned](#skills-and-lessons-learned)
- [Future improvements](#future-improvements)
- [Author](#author)

## Internship overview

| Field | Details |
|---|---|
| Organization | Datheon |
| Role | Machine Learning & Data Science Intern |
| Period | November–December 2026 |
| Development | Python, Jupyter Notebook, VS Code |
| Focus | Supervised learning, regression, classification, NLP, model evaluation |
| Deliverables | Four documented notebooks, supporting code, metrics and saved models where applicable |
| Progress | **4 of 4 project implementations completed** |

## Projects at a glance

| Task | Problem | Algorithms | Highlighted test result | Status |
|---|---|---|---|---|
| [01](#task-01--iris-multi-class-classification) | Iris species classification | Logistic Regression, Decision Tree | **93.33% accuracy** (both) | Completed |
| [02](#task-02--california-housing-price-prediction) | California housing value regression | OLS, Ridge, Lasso | **OLS RMSE $68,917.32; R² 0.6375** | Completed |
| [03](#task-03--telecom-customer-churn-prediction) | Telecom churn risk | Random Forest, XGBoost | **Tuned XGBoost ROC-AUC 0.8454** | Completed |
| [04](#task-04--nlp-sentiment-classification) | Review sentiment analysis | MultinomialNB, Logistic Regression | **Logistic Regression accuracy 88.96%** | Completed |

The tasks represent different problem types and **their metrics are not directly comparable**: accuracy for balanced classification, error and R² for regression, and ROC-AUC/recall for imbalanced classification.

---

## Task 01 — Iris Multi-Class Classification

### Objective and data

Classify **Setosa, Versicolor and Virginica** using four flower measurements: sepal length, sepal width, petal length and petal width. The Iris dataset has **150 observations** and three classes.

### Methodology

1. Inspect class balance, descriptive statistics and feature distributions.
2. Explore relationships with scatter plots and correlation visualizations.
3. Create a stratified **80/20 training/test split**.
4. Apply scaling where appropriate, fit **Logistic Regression** and **Decision Tree** classifiers.
5. Evaluate held-out predictions using accuracy, precision, recall, F1-score and confusion matrices.
6. Illustrate decision boundaries using **separately fitted two-feature models**; these are explanatory plots, not the full four-feature models.

### Results

| Model | Test accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | **93.33%** | 93.33% | 93.33% | 93.33% |
| Decision Tree | **93.33%** | 93.33% | 93.33% | 93.33% |

Both models correctly classified **28 of 30** held-out flowers. The precision, recall and F1 figures above are weighted averages. Setosa was correctly classified in the observed test set; errors occurred primarily between Versicolor and Virginica.

**Notebook:** [`Task-01-Iris-Classification/iris_classification.ipynb`](Task-01-Iris-Classification/iris_classification.ipynb)  
**Dataset:** `Task-01-Iris-Classification/Iris.csv`

### Key takeaway

Even simple supervised models can perform well on a small, structured dataset. Visualizing feature relationships and confusion matrices makes the classification errors easier to interpret. With only 30 test examples, the reported accuracy should not be interpreted as a highly precise estimate of future performance.

---

## Task 02 — California Housing Price Prediction

### Objective and data

Predict California **median house value (USD)** from housing, income, demographic and geographic attributes. The dataset contains **20,640 observations** and includes the categorical feature `ocean_proximity`. The `total_bedrooms` column has **207 missing values**.

### Methodology

- Inspect distributions, outliers, missingness, correlations and relationships with location and income.
- Split into **16,512 training** and **4,128 testing** observations.
- Apply median imputation, numerical scaling and categorical one-hot encoding in scikit-learn pipelines.
- Engineer `rooms_per_household`, `bedrooms_per_room`, `population_per_household` and `income_rooms_interaction`.
- Examine multicollinearity using correlation analysis and **variance inflation factors (VIF)**.
- Fit **Ordinary Least Squares (OLS)**, **Ridge (L2)** and **Lasso (L1)** regression.
- Tune regularization strength using **5-fold cross-validation**, then compare held-out MAE, RMSE, R² and residuals.

### Hyperparameter search

| Model | Selected alpha | Cross-validation RMSE (USD) |
|---|---:|---:|
| Ridge | 10 | 67,768.06 |
| Lasso | 100 | 67,768.24 |

### Held-out test performance

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| **OLS** | 49,709.54 | **68,917.32** | **0.6375** |
| Ridge | 49,727.57 | 68,968.30 | 0.6370 |
| Lasso | **49,691.69** | 69,003.13 | 0.6366 |

**Interpretation:** OLS achieved the lowest test RMSE and highest test R², while Lasso had the lowest test MAE. The differences were small. Multicollinearity was observed in several housing features, and errors were larger for some high-value properties. Capped target values near $500,001 and unconstrained negative predictions are limitations of this regression setup.

**Notebook:** [`Task-02-Housing-Price-Prediction/housing_price_prediction.ipynb`](Task-02-Housing-Price-Prediction/housing_price_prediction.ipynb)  
**Dataset:** `Task-02-Housing-Price-Prediction/housing.csv`

### Key takeaway

Regularization does not necessarily outperform an unregularized baseline on every held-out metric. Feature engineering, leakage-aware preprocessing and residual diagnostics are as important as model selection.

---

## Task 03 — Telecom Customer Churn Prediction

### Objective and data

Predict whether a telecom customer will churn using **7,043 customer records**. The dataset contains customer tenure, contract, service and billing features. Churn is imbalanced: **5,174 no-churn (73.5%)** versus **1,869 churn (26.5%)**.

### Methodology

1. Convert `TotalCharges` to numeric (11 missing values after conversion), remove `customerID` and encode the target (`No=0`, `Yes=1`).
2. Use a stratified **80/20 split**: **5,634 training**, **1,409 testing** records.
3. Build numerical/categorical preprocessing pipelines with imputation and one-hot encoding.
4. Train **Random Forest** with `class_weight="balanced"` and **XGBoost** with training-set-derived `scale_pos_weight`.
5. Evaluate models using stratified **5-fold cross-validation**.
6. Tune XGBoost with **RandomizedSearchCV** (15 parameter combinations × 5 folds = 75 fits), optimizing ROC-AUC.
7. Analyze ROC curves, confusion matrices, churn recall and feature importance; save the selected pipeline with Joblib.

### Cross-validation comparison

| Model | Mean ROC-AUC | Mean F1 | Mean precision | Mean recall |
|---|---:|---:|---:|---:|
| Random Forest | 82.24% | 60.40% | 56.82% | 64.48% |
| XGBoost | **84.77%** | **63.25%** | 53.05% | **78.33%** |

The tuned XGBoost configuration achieved **84.99% best cross-validation ROC-AUC**.

### Held-out test performance

| Metric | Random Forest | Tuned XGBoost |
|---|---:|---:|
| Accuracy | **76.37%** | 74.17% |
| Precision (churn) | **54.82%** | 50.85% |
| Recall (churn) | 62.30% | **80.21%** |
| F1 (churn) | 58.32% | **62.24%** |
| ROC-AUC | 82.30% | **84.54%** |

### Error trade-off

| Confusion matrix count | Random Forest | Tuned XGBoost |
|---|---:|---:|
| True negatives | 843 | 745 |
| False positives | 192 | 290 |
| False negatives | 141 | **74** |
| True positives | 233 | **300** |

Tuned XGBoost identified **300 of 374** actual churn customers, at the cost of more false-positive alerts. That trade-off may be preferable when missing a likely churn customer is more costly than contacting a customer who would not churn; the appropriate decision threshold depends on business costs.

The model's most influential features included **month-to-month contracts**, absence of technical support and online security, fiber-optic internet service and electronic-check payments. Feature importance describes association with predictions, **not causation**.

**Notebook:** [`Task-03-Customer-Churn-Prediction/customer_churn_prediction.ipynb`](Task-03-Customer-Churn-Prediction/customer_churn_prediction.ipynb)  
**Dataset:** `Task-03-Customer-Churn-Prediction/Telco-Customer-Churn.csv`  
**Saved model:** `Task-03-Customer-Churn-Prediction/models/customer_churn_xgboost.joblib`

### Key takeaway

For imbalanced classification, **accuracy alone is insufficient**. ROC-AUC, recall, precision and false-positive/false-negative costs should inform model selection. The current cross-validation implementation calculates XGBoost's class weight from the full training partition rather than recalculating it separately within each fold; this is a documented evaluation limitation.

---

## Task 04 — NLP Sentiment Classification

### Objective and datasets

Classify English-language reviews as **positive (1)** or **negative (0)**. The task includes the small supplied `sentiment_reviews.csv` starter dataset and a more substantial experiment with the **Stanford IMDb Large Movie Review Dataset**.

- **Original IMDb corpus:** 50,000 labeled movie reviews, balanced between the two classes.
- **Training partition after duplicate removal:** 24,904 reviews (12,472 positive; 12,432 negative).
- **Test partition after duplicate removal and cross-split overlap exclusion:** 24,678 reviews (12,412 positive; 12,266 negative).
- **Remaining exact duplicates within each partition:** 0.
- **Exact train/test text overlap:** 0.

**Sources:** [Stanford IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/) · [Hugging Face mirror (`stanfordnlp/imdb`)](https://huggingface.co/datasets/stanfordnlp/imdb)

### NLP workflow

1. Inspect review labels, nulls, duplicates and length distributions.
2. Normalize review text with HTML cleanup, punctuation handling and lowercasing.
3. Tokenize text, remove NLTK stopwords while retaining selected negations, and lemmatize using WordNet.
4. Fit **TF-IDF on the training partition only**, using `max_features=5000`, unigrams/bigrams, `min_df=2` and sublinear term frequency.
5. Train **Multinomial Naive Bayes** and **Logistic Regression** pipelines.
6. Compare accuracy, positive-class precision, recall and F1 on the test partition.
7. Produce classification reports and confusion matrices.
8. Implement `predict_sentiment(review)` returning a label and model-estimated probability.
9. Save the selected model with Joblib, extract reusable preprocessing into `text_preprocessing.py`, and verify reloading in a **fresh Python process**.

### Held-out test performance

| Model | Accuracy | Positive precision | Positive recall | Positive F1 |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 85.65% | 85.79% | 85.66% | 85.72% |
| **Logistic Regression** | **88.96%** | **88.68%** | **89.46%** | **89.07%** |

### Confusion matrices

| Count | MultinomialNB | Logistic Regression |
|---|---:|---:|
| True negatives | 10,505 | **10,849** |
| False positives | 1,761 | **1,417** |
| False negatives | 1,780 | **1,308** |
| True positives | 10,632 | **11,104** |

Logistic Regression correctly classified **21,953 of 24,678** held-out reviews. Example predictions included clearly positive and negative movie reviews as well as product-style phrases. **Product-review performance has not been separately validated.** The displayed confidence values are uncalibrated model probabilities, not guarantees of correctness.

**Notebook:** [`Task-04-NLP-Sentiment-Classification/sentiment_classification.ipynb`](Task-04-NLP-Sentiment-Classification/sentiment_classification.ipynb)  
**Starter dataset:** `Task-04-NLP-Sentiment-Classification/sentiment_reviews.csv`  
**Reusable preprocessing:** `Task-04-NLP-Sentiment-Classification/text_preprocessing.py`  
**Saved model:** `Task-04-NLP-Sentiment-Classification/models/imdb_sentiment_logistic_regression.joblib`

### Dataset and evaluation caveats

- The full IMDb Parquet/CSV files are **excluded from Git**. Run the notebook's download step, or obtain the dataset from the linked source.
- The notebook should be run from `Task-04-NLP-Sentiment-Classification/` so relative `dataset/` and `models/` paths resolve correctly.
- The model comparison **selected the better model using test results**. These scores are descriptive comparisons, not a strictly unbiased final estimate after selection. A stronger protocol would select models with training-only cross-validation and reserve a fresh untouched test set for final reporting.
- The trained pipeline depends on the included `text_preprocessing.py` module when loaded in a separate process.
- Generalization to other languages, review domains, sarcasm or shifting review distributions is untested.

### Key takeaway

A larger, balanced corpus and a well-defined TF-IDF pipeline improved the usefulness of the sentiment experiment compared with the tiny starter dataset. **Deployment reliability also requires reproducible preprocessing and serialization**, not only a good test score.

---

## Repository structure

```text
Datheon-ML-DataScience-Internship/
├── Task-01-Iris-Classification/
│   ├── iris_classification.ipynb
│   └── Iris.csv
├── Task-02-Housing-Price-Prediction/
│   ├── housing_price_prediction.ipynb
│   └── housing.csv
├── Task-03-Customer-Churn-Prediction/
│   ├── customer_churn_prediction.ipynb
│   ├── Telco-Customer-Churn.csv
│   └── models/
├── Task-04-NLP-Sentiment-Classification/
│   ├── sentiment_classification.ipynb
│   ├── sentiment_reviews.csv
│   ├── text_preprocessing.py
│   ├── models/
│   └── dataset/              # Local only; ignored by Git
├── .gitignore
├── requirements.txt
└── README.md
```

*The tree shows the principal project files. Generated caches, the virtual environment and the full IMDb dataset are intentionally omitted from version control.*

## Getting started

### 1. Clone and create a virtual environment

From **Windows PowerShell**:

```powershell
git clone https://github.com/sudeeraattanayake/Datheon-ML-DataScience-Internship.git
cd Datheon-ML-DataScience-Internship
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements should include `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `nltk`, `pyarrow`, `joblib`, `ipykernel` and `notebook`.

### 3. Download NLTK resources (one-time)

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

If a notebook reports another missing NLTK resource, install the named resource before continuing.

### 4. Open notebooks

Open the cloned folder in **VS Code**, select the `.venv` interpreter as the Jupyter kernel, and open the notebook for the task you want to reproduce. **Run cells in order** and use that task's folder as the working directory. Some notebooks expect local dataset files at specific relative paths.

| Task | Notebook |
|---|---|
| 01 | `Task-01-Iris-Classification/iris_classification.ipynb` |
| 02 | `Task-02-Housing-Price-Prediction/housing_price_prediction.ipynb` |
| 03 | `Task-03-Customer-Churn-Prediction/customer_churn_prediction.ipynb` |
| 04 | `Task-04-NLP-Sentiment-Classification/sentiment_classification.ipynb` |

### 5. IMDb data for Task 04

The Task 04 notebook includes a Python download step for the IMDb train and test Parquet files from the Hugging Face mirror. Run that step before the large-corpus experiment. The dataset files are kept in the ignored local `dataset/` directory. `pyarrow` is required for `pandas.read_parquet()`.

### 6. Saved models

The trained Joblib files can be loaded in compatible Python environments. **Only unpickle or load model artifacts from sources you trust.** For Task 04, retain `text_preprocessing.py` alongside the project so the serialized pipeline can resolve its preprocessing function.

## Reproducibility and data management

- **Split integrity:** preprocessing should be fitted on training data only. Task 04 removes exact review overlap across the official train/test partitions.
- **Metrics:** each notebook reports held-out metrics and relevant plots; numerical results can change if the split, package versions or data preprocessing change.
- **Environment:** `requirements.txt` lists dependencies without exact pins. For strict reproducibility, capture the tested package versions or use a lockfile.
- **Data provenance:** source datasets remain attributable to their respective providers. The IMDb corpus is not redistributed; review its terms and citation requirements before reuse.
- **Version control:** `.gitignore` excludes `.venv`, Python caches, temporary files and downloaded IMDb data. Check large Joblib file sizes before pushing; GitHub rejects regular files larger than 100 MB.
- **Security:** do not commit credentials, secrets or private customer records. Saved models are not inherently safe to deserialize from unknown sources.
- **Evaluation integrity:** avoid repeated tuning on the test set. The Task 04 test-set-based model selection is explicitly disclosed rather than presented as a fully independent final evaluation.

## Skills and lessons learned

| Area | Techniques and tools |
|---|---|
| Data analysis | Pandas, NumPy, descriptive statistics, EDA |
| Visualization | Matplotlib, Seaborn, distributions, heatmaps, residuals |
| Preprocessing | Missing-value imputation, scaling, one-hot encoding, text normalization |
| Classification | Logistic Regression, Decision Tree, Random Forest, XGBoost, MultinomialNB |
| Regression | OLS, Ridge, Lasso, engineered features, VIF |
| NLP | NLTK, WordNet lemmatization, TF-IDF, n-grams |
| Validation | Stratified splits, 5-fold CV, GridSearchCV, RandomizedSearchCV |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE, R² |
| Model management | Scikit-learn Pipelines, Joblib, reload verification |
| Development | Python, Jupyter Notebook, VS Code, Git and GitHub |

**Cross-project insights:** The right metric depends on the problem; simple baselines deserve serious consideration; preprocessing must respect data splits; model interpretability has limits; and reliable model artifacts require stable code dependencies.

## Future improvements

1. **Stronger evaluation:** introduce training-only model selection for Task 04, and use an untouched final benchmark where feasible.
2. **Reproducible environments:** pin tested versions, automate environment setup and add a smoke-test script.
3. **Automated testing:** validate data schemas, preprocessing functions, prediction shape and saved-model reloads.
4. **Error analysis:** investigate misclassified Iris samples, housing residuals, churn false positives and negative sentiment predictions.
5. **Decision support:** study cost-sensitive churn thresholds and calibrate classification probabilities where appropriate.
6. **NLP robustness:** benchmark additional review domains and negation/sarcasm examples without claiming out-of-domain performance from IMDb alone.
7. **Presentation:** add selected notebook figures or screenshots to the README after confirming they are accurate and useful.

## Completion summary

- [x] Task 01 — Iris Multi-Class Classification
- [x] Task 02 — California Housing Price Prediction
- [x] Task 03 — Telecom Customer Churn Prediction
- [x] Task 04 — NLP Sentiment Classification
- [x] Model evaluation and visualizations
- [x] Saved-model verification where implemented
- [x] Public repository documentation

**Implementation status: 4/4 tasks completed.** Administrative review or credential verification by Datheon is a separate process.

## Author

**Sudeera Attanayake**  
Machine Learning & Data Science Intern

- [GitHub](https://github.com/sudeeraattanayake)
- [LinkedIn](https://www.linkedin.com/in/sudeera-attanayake-9631b03b0)
- [Repository](https://github.com/sudeeraattanayake/Datheon-ML-DataScience-Internship)

## Disclaimer

These projects were developed for educational and internship purposes. Results are specific to the stated data and evaluation procedures and do not guarantee performance on new populations or in production systems. The repository does not claim that the models have been deployed or independently audited.
