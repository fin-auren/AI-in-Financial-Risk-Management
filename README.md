# AI-in-Financial-Risk-Management
We will build an AI model that predicts whether a credit card customer may default next month. Then we will convert model probabilities into risk bands and simple stress-scenario losses.
### Executive Summary of the Notebook

#### 1. Project Overview & Business Goal
* **Objective**: Build an AI model to predict whether a credit card customer will default on their payment next month (`default_next_month`).
* **Downstream Goal**: Translate model probabilities into operational risk bands and evaluate potential credit losses under simple stress scenarios.

#### 2. Data Acquisition & Pre-processing
* **Dataset**: UCI Credit Card Default dataset (ID: 350) containing 30,000 clients and 24 features.
* **Data Cleaning**:
  * Standardized column names to lowercase format.
  * Excluded undocumented/anomalous categorical values (`marriage == 0`, `education` values of `0`, `5`, and `6`), resulting in a cleaned dataset of **29,601 records**.

#### 3. Exploratory Data Analysis (EDA) Insights
* **Class Imbalance**: Approximately **77.69%** of customers are non-defaulters, while **22.31%** default, indicating a moderate class imbalance.
* **Key Predictor Trends**:
  * **Credit Limit (`limit_bal`)**: Defaulters consistently show a distribution skewed toward lower credit limits compared to non-defaulters.
  * **Age**: Both populations share a similar peak near late 20s/early 30s, though risk margins slightly shift across demographic distributions.
  * **Demographics**: Visualization of categorical columns (`sex`, `education`, `marriage`) highlights baseline distributions across high school, university, and graduate-level client segments.

#### 4. Modeling & Performance Metrics
Two benchmark Logistic Regression configurations have been evaluated using a stratified 80-20 train-test split:

| Model Configuration | Accuracy | Precision (Default) | Recall (Default) | F1-Score (Default) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Unbalanced)** | **81.05%** | **72.36%** | 24.38% | 36.47% | **0.725** |
| **Logistic Regression (Balanced with ROS)** | 69.04% | 38.58% | **65.48%** | **48.55%** | **0.725** |

* **Key Takeaway**: Balancing the dataset using a `RandomOverSampler` drastically improves the model's sensitivity (Recall) for identifying defaults from **24.38%** to **65.48%**, which is critical for proactive risk mitigation.
