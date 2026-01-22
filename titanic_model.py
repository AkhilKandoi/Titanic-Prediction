import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

from scipy.stats import uniform, randint
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

train = pd.read_csv(r"D:\code\ML\Datasets\titanic_train.csv")

y = train["Survived"]
X = train.drop("Survived", axis=1)

# Feature Engineering
X["Deck"] = X["Cabin"].str[0]

X.loc[X["Deck"].isnull() & (X["Pclass"] == 1), "Deck"] = "B"
X.loc[X["Deck"].isnull() & (X["Pclass"] == 2), "Deck"] = "D"
X.loc[X["Deck"].isnull() & (X["Pclass"] == 3), "Deck"] = "F"

X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

X["Title"] = X["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
X["Title"] = X["Title"].replace(['Mlle', 'Ms'], 'Miss')
X["Title"] = X["Title"].replace('Mme', 'Mrs')
X["Title"] = X["Title"].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer')
X["Title"] = X["Title"].replace(
    ['Don', 'Dona', 'Lady', 'Countess', 'Jonkheer', 'Sir'], 'Noble'
)

X = X.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Preprocessing
cat_features = X.select_dtypes(include=["object"]).columns
num_features = X.select_dtypes(include=["int64", "float64"]).columns

num_pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler()
)

preprocessor = make_column_transformer(
    (num_pipeline, num_features),
    (OneHotEncoder(handle_unknown="ignore"), cat_features)
)

X_processed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)
# Models
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(random_state=42)
svc = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('svc', svc)
    ],
    voting='soft'
)

# Hyperparameter Tuning
param_distributions = {
    'lr__C': uniform(0.01, 10),
    'rf__n_estimators': randint(50, 200),
    'rf__max_depth': [None, 5, 10, 15],
    'svc__C': uniform(0.1, 5)
}


random_search = RandomizedSearchCV(
    estimator=voting_clf,
    param_distributions=param_distributions,
    n_iter=30,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# Evaluation on Validation Set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Cross-validation Score
cv_scores = cross_val_score(
    best_model, X_train, y_train, cv=5, scoring="accuracy"
)

print("Cross-validation Accuracy:", cv_scores.mean())

