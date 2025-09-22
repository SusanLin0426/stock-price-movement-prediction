# Stock Price Movement Prediction

This repository contains the implementation and results of our **Big Data and Business Analytics project**. The project focuses on predicting stock price up/down movements using **textual features from news and forum articles**.

---

## Project Structure
- `Preprocessing_Part2.ipynb` : Data preprocessing pipeline (word segmentation, feature mapping)
- `Preprocessing_Part3.ipynb` : Extended preprocessing for rolling backtesting
- `NaiveBayes_Part2.ipynb` : Naive Bayes model training and testing
- `NaiveBayes_Part3.ipynb` : Naive Bayes with rolling backtesting
- `KNN_Part2.ipynb` : KNN training and testing
- `KNN_Part3.ipynb` : KNN with rolling backtesting
- `SVM.ipynb` : SVM implementation and tuning
- `XGBoost.ipynb` : XGBoost implementation and tuning
- `slides.pdf` : Report slides (in Chinese)

---

## Research Overview

### Goals
- Extract and label stock-related news/forum articles for **Yuanta Taiwan 50 ETF**.
- Predict stock price movement (up or down, threshold ±0.4%).
- Evaluate models using **accuracy, confusion matrix, and trading decision rate**.

### Methodology
1. **Preprocessing**
   - Word segmentation with **monpa**.
   - Feature extraction with Chi-Square filtering + CountVectorizer.
2. **Models**
   - Naive Bayes (Gaussian, Complement, Bernoulli, Multinomial)
   - KNN (hyperparameter tuning with GridSearchCV)
   - SVM (linear, poly, RBF kernels)
   - XGBoost (max_depth, gamma, learning_rate tuning)
3. **Evaluation**
   - Train-test split (80/20)
   - Rolling 3-month backtesting
   - Metrics: accuracy, confusion matrix, decision rate

---

## Key Findings
- **Naive Bayes** achieved ~67% accuracy in static testing; ~59% in rolling backtesting.
- **KNN, SVM, XGBoost** ranged between 53%–62.5% accuracy depending on period.
- Rolling backtesting showed **trade-off between accuracy and decision frequency**.
- **Preprocessing choice matters**: using monpa before CountVectorizer reduced sparsity and improved performance.

---

## Environment
- Python 3.9+
- Jupyter Notebook
- Libraries:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `xgboost`
  - `monpa`
  - `matplotlib`, `seaborn`
