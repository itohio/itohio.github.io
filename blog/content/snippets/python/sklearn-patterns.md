---
title: "Scikit-learn Common Patterns"
date: 2024-12-13
draft: false
category: "python"
tags: ["scikit-learn", "machine-learning", "python", "ml"]
---

Common patterns and workflows for scikit-learn: preprocessing, model training, evaluation, and pipelines.

## Installation

```bash
pip install scikit-learn numpy pandas matplotlib
```

## Basic Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
X, y = load_data()  # Features and target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

---

## Data Preprocessing

### Scaling and Normalization

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    Normalizer, MaxAbsScaler
)

# StandardScaler: mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# MinMaxScaler: scale to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

# MinMaxScaler: custom range
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X_train)

# RobustScaler: robust to outliers (uses median and IQR)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)

# Normalizer: scale samples to unit norm
normalizer = Normalizer(norm='l2')
X_normalized = normalizer.fit_transform(X_train)

# MaxAbsScaler: scale by maximum absolute value
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_train)
```

### Encoding Categorical Variables

```python
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)

# LabelEncoder: for target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # ['cat', 'dog'] -> [0, 1]
y_decoded = le.inverse_transform(y_encoded)

# OneHotEncoder: for features
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = encoder.fit_transform(X[['category']])

# OrdinalEncoder: for ordinal features
encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_encoded = encoder.fit_transform(X[['priority']])

# pd.get_dummies (Pandas alternative)
import pandas as pd
X_encoded = pd.get_dummies(X, columns=['category'], drop_first=True)
```

### Handling Missing Values

```python
from sklearn.impute import SimpleImputer, KNNImputer

# SimpleImputer: mean, median, most_frequent, constant
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)

imputer = SimpleImputer(strategy='constant', fill_value=0)
X_imputed = imputer.fit_transform(X)

# KNNImputer: impute using k-nearest neighbors
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

### Feature Engineering

```python
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Power transformation (make data more Gaussian)
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)

# Box-Cox (requires positive data)
pt = PowerTransformer(method='box-cox')
X_transformed = pt.fit_transform(X)
```

---

## Feature Selection

```python
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    SelectFromModel, VarianceThreshold, f_classif, mutual_info_classif
)

# SelectKBest: select top k features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# SelectPercentile: select top percentile
selector = SelectPercentile(score_func=mutual_info_classif, percentile=20)
X_selected = selector.fit_transform(X, y)

# RFE: Recursive Feature Elimination
from sklearn.ensemble import RandomForestClassifier
estimator = RandomForestClassifier(n_estimators=100)
selector = RFE(estimator, n_features_to_select=10)
X_selected = selector.fit_transform(X, y)

# RFECV: RFE with cross-validation
selector = RFECV(estimator, step=1, cv=5)
X_selected = selector.fit_transform(X, y)
print(f"Optimal features: {selector.n_features_}")

# SelectFromModel: based on feature importances
selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
X_selected = selector.fit_transform(X, y)

# VarianceThreshold: remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

---

## Model Training

### Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SVM
model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_train, y_train)

# K-Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)
```

### Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Ridge Regression (L2 regularization)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Lasso Regression (L1 regularization)
model = Lasso(alpha=1.0)
model.fit(X_train, y_train)

# ElasticNet (L1 + L2)
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
model.fit(X_train, y_train)

# Decision Tree Regressor
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SVR
model = SVR(kernel='rbf', C=1.0)
model.fit(X_train, y_train)
```

---

## Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # For binary classification

# Basic metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print(classification_report(y_test, y_pred))

# ROC-AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {auc:.3f}")

# Plot ROC curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)
plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
print(f"MAPE: {mape:.3f}")

# Plot predictions vs actual
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predictions vs Actual')
plt.show()
```

---

## Cross-Validation

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate, KFold, StratifiedKFold,
    TimeSeriesSplit, LeaveOneOut
)

# Simple cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Multiple metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']
scores = cross_validate(model, X, y, cv=5, scoring=scoring)
for metric in scoring:
    print(f"{metric}: {scores[f'test_{metric}'].mean():.3f}")

# K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)

# Stratified K-Fold (preserves class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)

# Time Series Split (no shuffle)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)

# Leave-One-Out
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

---

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# Random search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

---

## Pipelines

### Basic Pipeline

```python
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Cross-validation on pipeline
scores = cross_val_score(pipeline, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f}")
```

### Pipeline with Feature Selection

```python
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(k=10)),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

### Column Transformer

```python
from sklearn.compose import ColumnTransformer

# Define transformers for different column types
numeric_features = ['age', 'income', 'score']
categorical_features = ['city', 'category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### Pipeline with Grid Search

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
```

---

## Model Persistence

```python
import joblib
import pickle

# Save model with joblib (recommended)
joblib.dump(model, 'model.joblib')
loaded_model = joblib.load('model.joblib')

# Save model with pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Save pipeline
joblib.dump(pipeline, 'pipeline.joblib')
loaded_pipeline = joblib.load('pipeline.joblib')
```

---

## Common Patterns

### Handling Imbalanced Data

```python
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
model = RandomForestClassifier(class_weight='balanced')

# SMOTE (oversampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.show()
```

### Feature Importances

```python
# For tree-based models
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()
```

---

## Further Reading

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Scikit-learn Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
- [Scikit-learn Examples](https://scikit-learn.org/stable/auto_examples/index.html)

