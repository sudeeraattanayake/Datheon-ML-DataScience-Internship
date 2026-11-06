# Datheon — Machine Learning & Data Science Internship

A collection of practical Machine Learning and Data Science projects developed during my project-based internship at **Datheon**.

This repository demonstrates end-to-end machine learning workflows, including data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter optimization, evaluation, visualization, and model persistence.

## Internship Overview

- **Organization:** Datheon
- **Role:** Machine Learning & Data Science Intern
- **Duration:** November–December 2026
- **Development Environment:** Python, Jupyter Notebook, VS Code
- **Focus:** Machine Learning, Data Science, Predictive Modeling, Model Evaluation, and Visualization

## Project Tasks

| Task | Project | Status |
|---|---|---|
| 01 | Iris Multi-Class Classification | Completed |
| 02 | California Housing Price Prediction (OLS, Ridge & Lasso) | Completed |
| 03 | Customer Churn Prediction (Random Forest & XGBoost) | Completed |
| 04 | NLP Sentiment Classification (TF-IDF) | Planned |

---

## Task 01 — Iris Multi-Class Classification

### Objective

Build and evaluate an end-to-end supervised multi-class classification pipeline using the Iris dataset.

### Dataset

The Iris dataset contains **150 samples** representing three flower species:

- Setosa
- Versicolor
- Virginica

Each sample contains four numerical features: sepal length, sepal width, petal length, and petal width.

### Machine Learning Workflow

1. Load and inspect the dataset using Pandas.
2. Perform exploratory data analysis using Seaborn and Matplotlib.
3. Split the dataset into training and testing sets (80/20).
4. Apply StandardScaler for feature normalization.
5. Train Logistic Regression and Decision Tree classifiers.
6. Evaluate accuracy, precision, recall, and F1-score.
7. Visualize confusion matrices.
8. Plot two-dimensional decision boundaries.

### Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 93.33% | 93.33% | 93.33% | 93.33% |
| Decision Tree | 93.33% | 93.33% | 93.33% | 93.33% |

Both models correctly classified **28 out of 30 test samples**.

Precision, recall, and F1-score are weighted averages.

### Key Findings

- Setosa was correctly classified in all test cases.
- Most classification errors occurred between Versicolor and Virginica.
- Petal length and petal width showed strong positive correlation.
- Logistic Regression generated linear decision boundaries.
- Decision Tree generated axis-aligned decision regions.

Decision-boundary visualizations used separately trained two-feature models.

### Project Files

- `Task-01-Iris-Classification/iris_classification.ipynb`
- `Task-01-Iris-Classification/Iris.csv`

**Status:** Completed

---

## Task 02 — California Housing Price Prediction

### Objective

Develop an end-to-end regression pipeline to predict California median housing prices using **Ordinary Least Squares (OLS), Ridge Regression (L2), and Lasso Regression (L1)**.

The project focuses on feature engineering, missing-value handling, multicollinearity analysis, hyperparameter optimization, and regression model evaluation.

### Dataset

The California Housing dataset contains:

- **20,640 observations**
- **10 original columns**
- **9 input features**
- **1 target variable:** `median_house_value`

The dataset includes geographical, demographic, housing, and income-related information.

### Data Preprocessing

The preprocessing workflow includes:

1. Dataset inspection and statistical analysis.
2. Identification of missing values and duplicate records.
3. Median imputation for missing numerical values.
4. StandardScaler normalization of numerical features.
5. One-Hot Encoding for the `ocean_proximity` categorical feature.
6. Training and testing split (80/20).
7. Scikit-learn Pipelines to prevent preprocessing data leakage.

**Dataset Quality:**

- Missing values: 207, all in `total_bedrooms`.
- Duplicate rows: 0.
- Training samples: 16,512.
- Testing samples: 4,128.

### Feature Engineering

Four additional features were created:

| Engineered Feature | Description |
|---|---|
| `rooms_per_household` | Average number of rooms per household |
| `bedrooms_per_room` | Ratio of bedrooms to total rooms |
| `population_per_household` | Average population per household |
| `income_rooms_interaction` | Interaction between median income and rooms per household |

Feature engineering increased the number of input features from **9 to 13**, before categorical encoding.

### Multicollinearity Analysis

Feature correlations and Variance Inflation Factor (VIF) values were examined to identify multicollinearity.

Selected VIF results:

| Feature | VIF |
|---|---:|
| total_bedrooms | 49.94 |
| households | 47.03 |
| total_rooms | 18.82 |
| income_rooms_interaction | 17.80 |
| median_income | 13.69 |

Strong multicollinearity was observed among several housing-related variables, motivating the comparison of OLS with Ridge and Lasso regularization.

### Regression Models

Three regression algorithms were implemented:

- **OLS:** Ordinary Least Squares Linear Regression.
- **Ridge:** L2-regularized linear regression.
- **Lasso:** L1-regularized linear regression.

### Hyperparameter Optimization

Ridge and Lasso were optimized using **GridSearchCV with 5-fold cross-validation**.

| Model | Best Alpha | Cross-Validation RMSE |
|---|---:|---:|
| Ridge | 10 | 67,768.06 |
| Lasso | 100 | 67,768.24 |

Ridge achieved a marginally lower cross-validation RMSE.

### Final Model Performance

Models were evaluated on the held-out test dataset using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

| Model | MAE (USD) | RMSE (USD) | R² Score |
|---|---:|---:|---:|
| **OLS** | 49,709.54 | **68,917.32** | **0.6375** |
| Ridge | 49,727.57 | 68,968.30 | 0.6370 |
| Lasso | **49,691.69** | 69,003.13 | 0.6366 |

**Lowest Test RMSE:** OLS Regression

**Lowest Test MAE:** Lasso Regression

**Highest Test R²:** OLS Regression

### Data Visualization

The notebook includes:

- Housing price distribution and boxplots.
- Feature correlation heatmaps.
- Housing price comparisons by ocean proximity.
- Engineered-feature correlation analysis.
- Model performance comparison charts.
- Actual vs. predicted housing price plots.
- Residual scatter plots.
- Residual distribution histograms.
- OLS, Ridge, and Lasso coefficient comparisons.

### Key Findings

- Median income showed a strong positive correlation with median housing value (approximately 0.69).
- Several housing features exhibited substantial multicollinearity.
- OLS achieved the lowest test RMSE and highest test R² among the evaluated models.
- Lasso achieved the lowest test MAE.
- Ridge and Lasso produced coefficient regularization effects while maintaining similar predictive performance.
- Lasso reduced some categorical coefficients to zero.
- Prediction errors increased for some high-value properties.
- The dataset contains capped housing values around $500,001, affecting regression analysis.
- Some linear regression predictions were negative, demonstrating a limitation of unconstrained linear models.

### Conclusion

The project successfully implemented a complete machine learning regression workflow using the California Housing dataset.

OLS achieved the lowest test RMSE of **$68,917.32** and an R² score of **0.6375**, while Ridge and Lasso demonstrated comparable predictive performance.

The results highlight the importance of data preprocessing, feature engineering, regularization, cross-validation, and residual analysis in regression modeling.

### Project Files

- `Task-02-Housing-Price-Prediction/housing_price_prediction.ipynb`
- `Task-02-Housing-Price-Prediction/housing.csv`

**Status:** Completed

---

## Task 03 — Predictive Customer Churn Classification

### Objective

Develop and evaluate a machine learning classification system to predict telecom customer churn using **Random Forest and XGBoost**.

The project focuses on class imbalance handling, preprocessing pipelines, cross-validation, hyperparameter tuning, ROC-AUC evaluation, feature importance analysis, and model persistence.

### Dataset

The Telco Customer Churn dataset contains:

- **7,043 customer records**
- **21 original columns**
- **19 input features** after removing `customerID`
- **1 target variable:** `Churn`

The dataset includes customer demographics, subscription information, internet services, contract types, payment methods, and billing details.

### Dataset Quality

- Missing `TotalCharges` values after numeric conversion: 11.
- Duplicate customer IDs: 0.
- Training samples: 5,634.
- Testing samples: 1,409.
- Training/testing split: 80/20 with stratification.

### Data Preprocessing

The preprocessing workflow includes:

1. Dataset inspection and exploratory data analysis.
2. Conversion of `TotalCharges` to numeric values.
3. Removal of the `customerID` identifier.
4. Binary target encoding (`No = 0`, `Yes = 1`).
5. Stratified train-test splitting.
6. Median imputation for numerical features.
7. Most-frequent imputation for categorical features.
8. One-Hot Encoding using `OneHotEncoder`.
9. Integration of preprocessing and classifiers using Scikit-learn Pipelines.

Preprocessing is fitted within the training process to prevent validation and test feature leakage.

### Exploratory Data Analysis

The notebook includes:

- Customer churn distribution.
- Churn percentage visualization.
- Customer tenure distribution by churn.
- Monthly charges comparison.
- Contract type vs. churn.
- Internet service vs. churn.

### Class Imbalance Handling

The dataset contains an imbalanced binary target:

| Class | Customers | Percentage |
|---|---:|---:|
| No Churn | 5,174 | 73.5% |
| Churn | 1,869 | 26.5% |

Class weighting was used to address imbalance.

**Random Forest:**

`class_weight="balanced"`

**XGBoost:**

`scale_pos_weight=2.7686`

The XGBoost class weight was calculated from the training partition.

### Machine Learning Models

Two ensemble learning classifiers were developed:

- **Random Forest Classifier**
- **XGBoost Classifier**

Both were evaluated using stratified 5-fold cross-validation.

### Cross-Validation Performance

| Model | Mean ROC-AUC | Mean F1-Score | Mean Precision | Mean Recall |
|---|---:|---:|---:|---:|
| Random Forest | 82.24% | 60.40% | 56.82% | 64.48% |
| XGBoost | **84.77%** | **63.25%** | 53.05% | **78.33%** |

XGBoost achieved the highest baseline cross-validation ROC-AUC.

### Hyperparameter Optimization

XGBoost was optimized using **RandomizedSearchCV**.

Configuration:

- **Cross-validation:** Stratified 5-fold
- **Parameter combinations:** 15
- **Total fits:** 75
- **Optimization metric:** ROC-AUC
- **Random state:** 42

Best hyperparameters:

| Parameter | Value |
|---|---:|
| `n_estimators` | 285 |
| `max_depth` | 2 |
| `learning_rate` | 0.02969 |
| `subsample` | 0.94464 |
| `colsample_bytree` | 0.94066 |
| `min_child_weight` | 1 |
| `reg_lambda` | 0.52485 |

**Best Cross-Validation ROC-AUC:** 84.99%

### Final Model Performance

Both models were evaluated on the held-out test dataset containing 1,409 customers.

| Metric | Random Forest | Tuned XGBoost |
|---|---:|---:|
| Accuracy | **76.37%** | 74.17% |
| Precision | **54.82%** | 50.85% |
| Recall | 62.30% | **80.21%** |
| F1-Score | 58.32% | **62.24%** |
| ROC-AUC | 82.30% | **84.54%** |

**Best Test ROC-AUC:** Tuned XGBoost — 84.54%

Both classifiers exceeded the internship project's target ROC-AUC of 80%.

### Confusion Matrix Analysis

**Random Forest**

| Metric | Count |
|---|---:|
| True Negatives | 843 |
| False Positives | 192 |
| False Negatives | 141 |
| True Positives | 233 |

**Tuned XGBoost**

| Metric | Count |
|---|---:|
| True Negatives | 745 |
| False Positives | 290 |
| False Negatives | 74 |
| True Positives | 300 |

Tuned XGBoost correctly identified **300 out of 374 actual churn customers**.

It achieved higher churn recall than Random Forest, while producing more false-positive predictions.

### Feature Importance Analysis

The tuned XGBoost model identified the following top 10 predictive features:

| Feature | Importance |
|---|---:|
| Contract — Month-to-month | 0.422337 |
| Tech Support — No | 0.091224 |
| Online Security — No | 0.080787 |
| Internet Service — Fiber optic | 0.070935 |
| Payment Method — Electronic check | 0.049086 |
| Contract — Two year | 0.041017 |
| Internet Service — No | 0.032095 |
| Contract — One year | 0.029332 |
| Tenure | 0.022069 |
| Paperless Billing — Yes | 0.017587 |

Month-to-month contract status was the most influential feature according to the trained model's feature-importance measure.

These values represent predictive importance rather than causal effects.

### Model Visualization

The notebook includes:

- Customer churn distribution charts.
- Feature analysis plots.
- ROC curve comparison.
- Confusion matrix heatmaps.
- Classification reports.
- Top 10 feature importance bar chart.

### Model Persistence

The selected XGBoost pipeline was saved using Joblib.

**Model artifact:**

`Task-03-Customer-Churn-Prediction/models/customer_churn_xgboost.joblib`

The serialized pipeline contains the fitted preprocessing and classification components.

### Inference Testing

The saved model was successfully reloaded and used to predict churn for a customer from the held-out test dataset.

Example inference result:

- **Predicted Status:** No Churn
- **Churn Probability:** 7.71%
- **Actual Status:** No Churn

Prediction consistency was verified between the original and reloaded models using NumPy numerical assertions.

### Key Findings

- Both classifiers achieved ROC-AUC scores above 80%.
- Tuned XGBoost achieved the highest test ROC-AUC of 84.54%.
- XGBoost achieved 80.21% recall for the churn class.
- Random Forest achieved higher overall accuracy and precision.
- Contract type, technical support, online security, and internet service were influential predictive features.
- The results demonstrate a trade-off between identifying churn customers and generating false-positive alerts.

### Limitations

- The dataset is a historical telecom customer sample and may not represent current customer behavior.
- Class weighting improves attention to the minority class but does not eliminate prediction errors.
- XGBoost's `scale_pos_weight` was calculated from the complete training partition rather than separately within each cross-validation training fold.
- Hyperparameter selection was based on cross-validation, so the selected cross-validation score may be optimistic.
- Model feature importance does not establish causal relationships.
- Further testing, monitoring, security validation, and deployment infrastructure would be required for real-world production use.

### Conclusion

The project successfully developed and evaluated Random Forest and XGBoost classifiers for customer churn prediction.

Tuned XGBoost was selected as the preferred model based on its superior ROC-AUC, F1-score, and churn recall.

The final XGBoost model achieved a **test ROC-AUC of 84.54%**, exceeding the project's target of 80%.

The project demonstrates a reproducible, production-oriented machine learning workflow incorporating data preprocessing, class imbalance handling, cross-validation, hyperparameter optimization, model evaluation, feature importance analysis, and model persistence.

### Project Files

- `Task-03-Customer-Churn-Prediction/customer_churn_prediction.ipynb`
- `Task-03-Customer-Churn-Prediction/Telco-Customer-Churn.csv`
- `Task-03-Customer-Churn-Prediction/models/customer_churn_xgboost.joblib`

**Status:** Completed

---

## Task 04 — NLP Sentiment Classification

### Objective

Develop a Natural Language Processing classification system to identify positive and negative sentiment in customer or movie reviews.

### Planned Workflow

1. Load and inspect the text dataset.
2. Clean and normalize review text.
3. Apply stopword removal and lemmatization.
4. Extract TF-IDF features.
5. Train Multinomial Naive Bayes and Logistic Regression classifiers.
6. Evaluate classification performance.
7. Analyze confusion matrices and classification reports.
8. Implement a reusable sentiment prediction function.
9. Save and validate the selected NLP pipeline.

**Status:** Planned

---

## Technologies

### Programming and Data Analysis

- Python
- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost
- Random Forest
- Logistic Regression
- Decision Tree
- Ridge Regression
- Lasso Regression

### Data Preprocessing and Evaluation

- One-Hot Encoding
- StandardScaler
- SimpleImputer
- Scikit-learn Pipelines
- Stratified K-Fold Cross-Validation
- GridSearchCV
- RandomizedSearchCV
- ROC-AUC
- Classification Metrics
- Regression Metrics

### Visualization

- Matplotlib
- Seaborn

### Model Persistence

- Joblib

### Development Tools

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sudeeraattanayake/Datheon-ML-DataScience-Internship.git
cd Datheon-ML-DataScience-Internship
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Ensure the dependency list includes the libraries required by the completed notebooks, including `xgboost` and `joblib`.

### 5. Run the Notebooks

Open the repository in VS Code and select the Python virtual environment as the Jupyter kernel.

Available notebooks:

- `Task-01-Iris-Classification/iris_classification.ipynb`
- `Task-02-Housing-Price-Prediction/housing_price_prediction.ipynb`
- `Task-03-Customer-Churn-Prediction/customer_churn_prediction.ipynb`

Run each notebook sequentially from its corresponding project directory so that relative dataset paths resolve correctly.

### 6. Model Artifacts

Task 03 generates a serialized XGBoost pipeline at:

`Task-03-Customer-Churn-Prediction/models/customer_churn_xgboost.joblib`

Only load serialized models from trusted sources and use compatible dependency versions.

---

## Project Progress

- [x] Task 01 — Iris Multi-Class Classification
- [x] Task 02 — California Housing Price Prediction
- [x] Task 03 — Customer Churn Prediction
- [ ] Task 04 — NLP Sentiment Classification

**Overall Progress: 3 of 4 Tasks Completed (75%)**

---

## Author

**Sudeera Attanayake**

Machine Learning & Data Science Intern

- **GitHub:** https://github.com/sudeeraattanayake
- **LinkedIn:** https://www.linkedin.com/in/sudeera-attanayake-9631b03b0

---

## Disclaimer

This repository contains independently developed educational projects completed as part of a project-based internship program.

Results are based on the datasets, preprocessing methods, and experimental configurations described in the notebooks.

Model performance is not a guarantee of real-world predictive accuracy. The projects demonstrate machine learning development practices and are not presented as fully validated production deployments.