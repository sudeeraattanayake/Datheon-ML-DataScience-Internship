# Datheon — Machine Learning & Data Science Internship

Four hands-on machine learning projects developed during a project-based internship with **Datheon**. The repository covers exploratory data analysis, preprocessing, feature engineering, model selection, evaluation, visualization, and model persistence.

**Role:** Machine Learning & Data Science Intern  
**Period:** November–December 2026  
**Environment:** Python, Jupyter Notebook, VS Code

## Projects at a glance

| Task | Project | Models | Selected result | Status |
|---|---|---|---|---|
| 01 | Iris multi-class classification | Logistic Regression, Decision Tree | Both: **93.33% test accuracy** | Completed |
| 02 | California housing price prediction | OLS, Ridge, Lasso | OLS: **RMSE $68,917.32**, **R² 0.6375** | Completed |
| 03 | Telecom customer churn prediction | Random Forest, tuned XGBoost | XGBoost: **ROC-AUC 0.8454**, churn recall **80.21%** | Completed |
| 04 | IMDb sentiment classification | Multinomial Naive Bayes, Logistic Regression | Logistic Regression: **88.96% test accuracy**, positive-class F1 **0.8907** | Completed |

## Task 01 — Iris Multi-Class Classification

**Goal:** Predict Iris flower species (Setosa, Versicolor, Virginica) from four measurements across 150 samples.

- Exploratory analysis, visualization, stratified 80/20 split, scaling, and model evaluation.
- Compared Logistic Regression and Decision Tree classifiers.
- Both models correctly classified **28 of 30** held-out examples (**93.33% accuracy**).
- Confusion matrices and illustrative two-feature decision boundaries are included.

**Notebook:** `Task-01-Iris-Classification/iris_classification.ipynb`

## Task 02 — California Housing Price Prediction

**Goal:** Predict median California house values using regression and engineered housing features.

- Dataset: 20,640 records; missing-value imputation, scaling, categorical one-hot encoding.
- Engineered household ratios and an income–rooms interaction feature.
- Compared Ordinary Least Squares, Ridge, and Lasso; tuned regularization using 5-fold cross-validation.
- Evaluated MAE, RMSE, R², coefficient behavior, and residual plots.

| Model | Test MAE (USD) | Test RMSE (USD) | Test R² |
|---|---:|---:|---:|
| OLS | 49,709.54 | **68,917.32** | **0.6375** |
| Ridge | 49,727.57 | 68,968.30 | 0.6370 |
| Lasso | **49,691.69** | 69,003.13 | 0.6366 |

**Notebook:** `Task-02-Housing-Price-Prediction/housing_price_prediction.ipynb`

## Task 03 — Customer Churn Prediction

**Goal:** Predict telecom customer churn using an imbalanced dataset of 7,043 customer records.

- Numeric/categorical preprocessing, imputation, one-hot encoding, and stratified 80/20 split.
- Addressed class imbalance with class weights and XGBoost `scale_pos_weight`.
- Compared Random Forest and XGBoost; tuned XGBoost with randomized search and 5-fold cross-validation.
- Evaluated ROC-AUC, precision, recall, F1, confusion matrices, and feature importance.

| Metric | Random Forest | Tuned XGBoost |
|---|---:|---:|
| Test accuracy | **76.37%** | 74.17% |
| Test precision | **54.82%** | 50.85% |
| Churn recall | 62.30% | **80.21%** |
| Test F1 | 58.32% | **62.24%** |
| Test ROC-AUC | 82.30% | **84.54%** |

Tuned XGBoost prioritized identifying more churn cases, at the cost of more false positives.

**Notebook:** `Task-03-Customer-Churn-Prediction/customer_churn_prediction.ipynb`  
**Saved model:** `Task-03-Customer-Churn-Prediction/models/customer_churn_xgboost.joblib`

## Task 04 — NLP Sentiment Classification

**Goal:** Classify English reviews as positive or negative using TF-IDF and traditional machine learning.

- Includes the small provided `sentiment_reviews.csv` starter dataset and an extended IMDb movie-review experiment.
- Source: [Stanford IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), accessed via [Hugging Face `stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb).
- Original IMDb corpus: **50,000 labeled reviews**, balanced across positive and negative sentiment.
- After duplicate removal and exclusion of train–test review overlap: **24,904 training** and **24,678 testing** reviews.
- Preprocessing: HTML/punctuation cleanup, lowercasing, tokenization, stopword removal (preserving key negations), and WordNet lemmatization.
- Feature extraction: `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))`.
- Models: Multinomial Naive Bayes and Logistic Regression.

| Model | Test accuracy | Positive precision | Positive recall | Positive F1 |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 85.65% | 85.79% | 85.66% | 85.72% |
| Logistic Regression | **88.96%** | **88.68%** | **89.46%** | **89.07%** |

The notebook includes classification reports, confusion matrices, and a `predict_sentiment()` helper. The selected Logistic Regression pipeline was saved with Joblib and **successfully loaded and tested in a fresh Python process**.

**Notebook:** `Task-04-NLP-Sentiment-Classification/sentiment_classification.ipynb`  
**Preprocessing module:** `Task-04-NLP-Sentiment-Classification/text_preprocessing.py`  
**Saved model:** `Task-04-NLP-Sentiment-Classification/models/imdb_sentiment_logistic_regression.joblib`

### IMDb dataset download

The full IMDb CSV and Parquet files are intentionally **excluded from Git**. The notebook includes a dataset download step; run it before the extended IMDb experiment. The source is also available at the Hugging Face dataset link above. For Parquet reading, install `pyarrow` via `requirements.txt`.

Run the notebook from `Task-04-NLP-Sentiment-Classification/` so its relative `dataset/` and `models/` paths resolve correctly.

**Evaluation note:** Task 04's model comparison and selection used the held-out test scores. These results are descriptive rather than a strictly unbiased post-selection estimate. For future tuning, select models using training-only validation/cross-validation and reserve a separate untouched test set for the final estimate. The model was trained on movie reviews; accuracy on product reviews or other domains is not established.

## Repository structure

```text
Datheon-ML-DataScience-Internship/
├── Task-01-Iris-Classification/
│   └── iris_classification.ipynb
├── Task-02-Housing-Price-Prediction/
│   └── housing_price_prediction.ipynb
├── Task-03-Customer-Churn-Prediction/
│   ├── customer_churn_prediction.ipynb
│   └── models/
├── Task-04-NLP-Sentiment-Classification/
│   ├── sentiment_classification.ipynb
│   ├── sentiment_reviews.csv
│   ├── text_preprocessing.py
│   ├── models/
│   └── dataset/                 # Local only; ignored by Git
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup and execution (Windows PowerShell)

```powershell
git clone https://github.com/sudeeraattanayake/Datheon-ML-DataScience-Internship.git
cd Datheon-ML-DataScience-Internship
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open the repository in VS Code and select `.venv` as the Jupyter kernel. Open a task notebook and run its cells in order, using the task folder as the working directory. Download any datasets excluded from Git before running data-loading cells. For NLTK, the notebooks may also require one-time downloads of `stopwords` and `wordnet` corpora.

### Notes on reproducibility

- Dependencies are listed by package name; for exact reproducibility, record tested versions in a lockfile or pinned environment export.
- Saved Joblib models should only be loaded from trusted sources and may require compatible Python/library versions.
- GitHub blocks individual files larger than 100 MB; check model artifact sizes before committing.
- The full IMDb dataset is not redistributed in this repository; consult the source dataset's terms and citation information.

## Skills demonstrated

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · NLTK · TF-IDF · Classification · Regression · Cross-validation · Hyperparameter tuning · Model evaluation · Joblib · Jupyter Notebook · Git/GitHub

## Author

**Sudeera Attanayake** — Machine Learning & Data Science Intern  
[GitHub](https://github.com/sudeeraattanayake) · [LinkedIn](https://www.linkedin.com/in/sudeera-attanayake-9631b03b0)

## Disclaimer

These are educational internship projects. Reported metrics are specific to the described datasets and evaluation procedures and do not establish production readiness or guaranteed real-world performance.
