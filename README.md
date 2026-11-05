
# Datheon — Machine Learning & Data Science Internship

A collection of practical Machine Learning and Data Science projects developed during my project-based internship at Datheon.

## Internship Overview

- **Organization:** Datheon
- **Role:** Machine Learning & Data Science Intern
- **Duration:** November–December 2026
- **Development Environment:** Python, Jupyter Notebook, VS Code
- **Focus:** Machine Learning, Data Analysis, Model Evaluation and Visualization

## Project Tasks

| Task | Project | Status |
|---|---|---|
| 01 | Iris Multi-Class Classification | Completed |
| 02 | Housing Price Prediction (Ridge & Lasso) | Planned |
| 03 | Customer Churn Prediction (Random Forest & XGBoost) | Planned |
| 04 | NLP Sentiment Classification (TF-IDF) | Planned |

## Task 01 — Iris Multi-Class Classification

### Objective

Build and evaluate a supervised multi-class classification pipeline using the Iris dataset.

### Dataset

The Iris dataset contains 150 samples representing three flower species:

- Setosa
- Versicolor
- Virginica

Each sample contains four numerical features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Machine Learning Workflow

1. Load and inspect the dataset using Pandas.
2. Perform exploratory data analysis with Seaborn.
3. Split the dataset into training and testing sets (80/20).
4. Apply StandardScaler for feature normalization.
5. Train Logistic Regression and Decision Tree classifiers.
6. Evaluate accuracy, precision, recall and F1-score.
7. Visualize confusion matrices.
8. Plot two-dimensional decision boundaries.

### Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 93.33% | 93.33% | 93.33% | 93.33% |
| Decision Tree | 93.33% | 93.33% | 93.33% | 93.33% |

Both models correctly classified 28 out of 30 test samples.

Precision, recall and F1-score are weighted averages.

### Key Findings

- Setosa was classified correctly in all test cases.
- Most errors occurred between Versicolor and Virginica.
- Petal length and petal width are highly correlated.
- Logistic Regression produced linear decision boundaries.
- Decision Tree produced axis-aligned decision regions.

The decision-boundary visualizations used separately trained two-feature models.

### Project Files

- `Task-01-Iris-Classification/iris_classification.ipynb`
- `Task-01-Iris-Classification/Iris.csv`

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/sudeeraattanayake/Datheon-ML-DataScience-Internship.git
cd Datheon-ML-DataScience-Internship
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open `Task-01-Iris-Classification/iris_classification.ipynb` in VS Code or Jupyter Notebook and run the cells in order.

## Author

**Sudeera Attanayake**

Machine Learning & Data Science Intern

- GitHub: https://github.com/sudeeraattanayake
- LinkedIn: https://www.linkedin.com/in/sudeera-attanayake-9631b03b0

## Disclaimer

This repository contains independently developed educational projects completed as part of a project-based internship program. Results are based on the datasets and experimental configurations described in the notebooks.