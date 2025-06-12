#  Credit Card Fraud Detection using LightGBM

Fraudulent credit card transactions are a serious issue for financial institutions and customers alike. This project aims to build a robust fraud detection system using **LightGBM**, with thoughtful feature transformations, class imbalance strategies, and metric-based threshold tuning for optimal fraud capture.

🔗 **Kaggle Notebook:** [View Full Project Here](https://www.kaggle.com/code/khwaishsaxena/credit-card-fraud-detection)

---

##  Project Overview

-  Goal: Accurately detect fraudulent transactions from anonymized credit card data
-  Model: LightGBM (with Optuna hyperparameter tuning)
-  Class Imbalance: Handled using `class_weight`
-  Evaluation: F1-Score, Precision, Recall, Confusion matrix, Classification report
-  Environment: kaggle Notebook, Python 

---

##  Motivation

Credit card fraud data is rare, highly imbalanced, and sensitive. Detecting fraud is not just about accuracy — it's about:
- Minimizing false negatives (missing a fraud)
- Managing false positives (flagging genuine transactions)
- Building interpretable and scalable ML pipelines

---

##  Dataset Details

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Records: 284,807 transactions
- Fraud cases: 492 (0.172%)
- Features: 
  - `V1` to `V28`: PCA-transformed features  
  - `Time` and `Amount`: Original numeric features  
  - `Class`: Target (0 = Non-Fraud, 1 = Fraud)

---

##  Workflow Summary

###  Data Preprocessing
- Copied dataset for safe manipulation
- Applied transformations to improve feature distribution:
  - `np.sqrt()`
  - `np.log()`

>  **No outliers were removed** — preserving rare fraud signals was critical

---

###  Exploratory Data Analysis
- KDE plots to visualize feature-wise fraud vs non-fraud distributions
- Class distribution imbalance checked
- Identified features showing potential for separating classes

---

###  Modeling with LightGBM
- Used `class_weight='balanced'` for class imbalance
- **Optuna** used to tune:
  - `num_leaves`
  - `max_depth`
  - `learning rate`
  - `subsample`
  - `colsample_bytree`
  - `n_estimators`


>  **XGBoost and Random Forest were NOT used** due to compatibility issues with transformed data (`NaN`, `inf` errors)

---

###  Evaluation Metrics
- F1-Score  
- Precision
- Recall
- Confusion matrix
- Classification report 

---

##  Results

| Metric        | Value    |
|---------------|----------|
| F1-Score      | 90%      |
| Precision     | 93.5%    |
| Recall        | 86.9%    |


 Final model: **LightGBM** with tuned parameters and threshold optimization

---

##  Visualizations

- KDE plots of transformed features
- Confusion matrix of final model
- Feature impact analysis

---

##  How to Run This Project

```bash
# Step 1: Clone the repo
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Step 2: Run the Jupyter Notebook
jupyter notebook lightgbm_fraud_detection.ipynb
