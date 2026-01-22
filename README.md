# Titanic Survival Prediction Pipeline

An end-to-end machine learning project that predicts passenger survival on the Titanic using structured data, feature engineering, ensemble learning, and robust evaluation metrics.

---

## 📌 Project Overview

This project demonstrates a complete machine learning workflow, from raw data preprocessing and feature engineering to model training, hyperparameter tuning, and evaluation.  
The goal is to build a reliable classification model that predicts whether a passenger survived the Titanic disaster.

---

## 🧠 Key Features

- Feature engineering using domain knowledge (Deck, Title, FamilySize, IsAlone)
- Data preprocessing with imputation, scaling, and one-hot encoding
- Ensemble learning using soft voting
- Hyperparameter optimization with RandomizedSearchCV
- Comprehensive evaluation using multiple performance metrics

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Models:** Logistic Regression, Random Forest, Support Vector Machine  

---

## ⚙️ Pipeline Overview

1. **Data Loading**
   - Titanic training dataset
   - Target variable: `Survived`

2. **Feature Engineering**
   - Extracted passenger deck from cabin information
   - Derived titles from passenger names
   - Created family-based features (FamilySize, IsAlone)

3. **Preprocessing**
   - Numerical features: mean imputation + standard scaling
   - Categorical features: one-hot encoding
   - Column-wise transformations using `ColumnTransformer`

4. **Modeling**
   - Trained Logistic Regression, Random Forest, and SVM
   - Combined models using a soft voting ensemble

5. **Hyperparameter Tuning**
   - Used `RandomizedSearchCV` to optimize ensemble parameters

6. **Evaluation**
   - Hold-out validation set
   - 5-fold cross-validation

---

## 📊 Evaluation Metrics

The model was evaluated using multiple metrics to ensure balanced and robust performance:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Cross-validation accuracy

