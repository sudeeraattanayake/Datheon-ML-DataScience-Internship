# Datheon — Machine Learning & Data Science Internship

A collection of practical Machine Learning and Data Science projects developed during my project-based internship at **Datheon**.

This repository demonstrates end-to-end machine learning workflows, including data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter optimization, evaluation, and visualization.

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
| 03 | Customer Churn Prediction (Random Forest & XGBoost) | Planned |
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
|---|---|---|---|---|
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

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- VS Code
- Git and GitHub

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

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Notebooks

Open the repository in VS Code and select the Python virtual environment as the Jupyter kernel.

Available notebooks:

- `Task-01-Iris-Classification/iris_classification.ipynb`
- `Task-02-Housing-Price-Prediction/housing_price_prediction.ipynb`

Run the notebook cells sequentially.

## Project Progress

- [x] Task 01 — Iris Multi-Class Classification
- [x] Task 02 — California Housing Price Prediction
- [ ] Task 03 — Customer Churn Prediction
- [ ] Task 04 — NLP Sentiment Classification

## Author

**Sudeera Attanayake**

Machine Learning & Data Science Intern

- GitHub: https://github.com/sudeeraattanayake
- LinkedIn: https://www.linkedin.com/in/sudeera-attanayake-9631b03b0

## Disclaimer

This repository contains independently developed educational projects completed as part of a project-based internship program. Results are based on the datasets, preprocessing methods, and experimental configurations described in the notebooks. Model performance is not a guarantee of real-world predictive accuracy.