---
title: "Random Forests in Depth"
date: 2024-12-13
draft: false
category: "python"
tags: ["random-forest", "machine-learning", "ensemble", "python"]
---

Comprehensive guide to Random Forests: theory, implementation, tuning, and interpretation.

## What are Random Forests?

**Random Forest** is an ensemble learning method that constructs multiple decision trees and combines their predictions.

**Key Concepts:**
- **Bagging**: Bootstrap Aggregating - train each tree on random subset of data
- **Feature Randomness**: Each split considers random subset of features
- **Ensemble**: Combine predictions by voting (classification) or averaging (regression)

---

## Basic Usage

### Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Maximum depth (None = unlimited)
    min_samples_split=2,   # Minimum samples to split node
    min_samples_leaf=1,    # Minimum samples in leaf
    max_features='sqrt',   # Features to consider for split
    bootstrap=True,        # Use bootstrap samples
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)

rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Regression

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")
```

---

## Hyperparameter Tuning

### Key Hyperparameters

```python
# Number of trees
n_estimators = [50, 100, 200, 500]

# Maximum depth of trees
max_depth = [10, 20, 30, None]

# Minimum samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum samples required at leaf node
min_samples_leaf = [1, 2, 4]

# Number of features to consider at each split
max_features = ['sqrt', 'log2', None]  # sqrt = sqrt(n_features)

# Bootstrap samples
bootstrap = [True, False]

# Criterion
criterion = ['gini', 'entropy']  # For classification
criterion = ['squared_error', 'absolute_error']  # For regression
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use best model
best_rf = grid_search.best_estimator_
```

### Random Search (Faster)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

---

## Feature Importance

### Basic Feature Importance

```python
import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for i, idx in enumerate(indices):
    print(f"{i+1}. Feature {idx} ({importances[idx]:.4f})")

# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), indices)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# With feature names
if isinstance(X_train, pd.DataFrame):
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(feature_importance_df)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'][:20], 
             feature_importance_df['importance'][:20])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()
```

### Permutation Importance

More reliable than default feature importances.

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Sort by importance
sorted_idx = perm_importance.importances_mean.argsort()[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.boxplot(perm_importance.importances[sorted_idx].T,
            labels=np.array(range(X_test.shape[1]))[sorted_idx],
            vert=False)
plt.xlabel("Permutation Importance")
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.show()
```

---

## Model Interpretation

### Partial Dependence Plots

```python
from sklearn.inspection import PartialDependenceDisplay

# Plot partial dependence for top features
features = [0, 1, (0, 1)]  # Single features and interaction
fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(
    rf, X_train, features, ax=ax
)
plt.tight_layout()
plt.show()
```

### SHAP Values

```python
import shap

# Create explainer
explainer = shap.TreeExplainer(rf)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Force plot for single prediction
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_test[0]
)

# Dependence plot
shap.dependence_plot(0, shap_values[1], X_test)
```

---

## Out-of-Bag (OOB) Score

Random Forests can estimate test error without cross-validation using OOB samples.

```python
# Enable OOB score
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # Enable OOB scoring
    bootstrap=True,  # Required for OOB
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# OOB score (similar to cross-validation score)
print(f"OOB Score: {rf.oob_score_:.3f}")

# OOB predictions
oob_pred = rf.oob_decision_function_
```

---

## Handling Imbalanced Data

### Class Weights

```python
from sklearn.utils import class_weight

# Compute class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Use in model
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # or dict with custom weights
    random_state=42
)

rf.fit(X_train, y_train)
```

### Balanced Random Forest

```python
from imblearn.ensemble import BalancedRandomForestClassifier

# Automatically balances classes
brf = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy='auto',  # Balance all classes
    replacement=True,
    random_state=42,
    n_jobs=-1
)

brf.fit(X_train, y_train)
```

---

## Ensemble Methods with Random Forests

### Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Stacking with logistic regression as meta-model
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
```

### Voting

```python
from sklearn.ensemble import VotingClassifier

# Create voting classifier
voting = VotingClassifier(
    estimators=[
        ('rf1', RandomForestClassifier(n_estimators=100, max_depth=10)),
        ('rf2', RandomForestClassifier(n_estimators=200, max_depth=20)),
        ('rf3', RandomForestClassifier(n_estimators=300, max_depth=None))
    ],
    voting='soft'  # 'hard' for majority vote, 'soft' for probability average
)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
```

---

## Advanced Techniques

### Extremely Randomized Trees (Extra Trees)

Faster training, sometimes better performance.

```python
from sklearn.ensemble import ExtraTreesClassifier

# Extra Trees: more randomness in splits
et = ExtraTreesClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

et.fit(X_train, y_train)
```

### Isolation Forest (Anomaly Detection)

```python
from sklearn.ensemble import IsolationForest

# Detect anomalies
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # Expected proportion of outliers
    random_state=42
)

# Fit and predict (-1 for outliers, 1 for inliers)
predictions = iso_forest.fit_predict(X)
anomalies = X[predictions == -1]
```

### Quantile Regression Forest

Estimate prediction intervals.

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Train multiple trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get predictions from all trees
all_predictions = np.array([tree.predict(X_test) for tree in rf.estimators_])

# Calculate quantiles
lower_bound = np.percentile(all_predictions, 5, axis=0)
upper_bound = np.percentile(all_predictions, 95, axis=0)
median = np.percentile(all_predictions, 50, axis=0)

# Plot prediction intervals
plt.figure(figsize=(10, 6))
plt.scatter(y_test, median, alpha=0.5, label='Predictions')
plt.fill_between(range(len(y_test)), lower_bound, upper_bound, 
                 alpha=0.2, label='90% Prediction Interval')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', label='Perfect Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()
```

---

## Optimization Tips

### Memory Optimization

```python
# Reduce memory usage
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,           # Limit tree depth
    min_samples_leaf=5,     # Increase minimum leaf size
    max_features='sqrt',    # Limit features per split
    warm_start=False,       # Don't keep trees in memory
    n_jobs=-1
)
```

### Speed Optimization

```python
# Faster training
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,           # Limit depth
    min_samples_split=10,   # Larger minimum split
    max_features='log2',    # Fewer features
    bootstrap=True,
    n_jobs=-1,              # Parallel processing
    random_state=42
)

# Use warm_start for incremental training
rf = RandomForestClassifier(n_estimators=50, warm_start=True)
rf.fit(X_train, y_train)

# Add more trees
rf.n_estimators = 100
rf.fit(X_train, y_train)  # Only trains 50 new trees
```

---

## Common Pitfalls

### Overfitting

```python
# ❌ BAD: Overfitting
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,         # Unlimited depth
    min_samples_split=2,    # Split on 2 samples
    min_samples_leaf=1      # Leaf can have 1 sample
)

# ✅ GOOD: Regularization
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,           # Limit depth
    min_samples_split=10,   # Require more samples to split
    min_samples_leaf=5,     # Require more samples in leaf
    max_features='sqrt'     # Limit features
)
```

### Class Imbalance

```python
# ❌ BAD: Ignoring imbalance
rf = RandomForestClassifier()

# ✅ GOOD: Handle imbalance
rf = RandomForestClassifier(class_weight='balanced')

# Or use sampling
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
rf.fit(X_resampled, y_resampled)
```

---

## Comparison with Other Algorithms

| Algorithm | Pros | Cons |
|-----------|------|------|
| **Random Forest** | Robust, handles non-linear, feature importance | Slow prediction, black box, large memory |
| **Gradient Boosting** | Higher accuracy, handles missing values | Slower training, prone to overfitting |
| **Logistic Regression** | Fast, interpretable, probabilistic | Linear only, requires feature engineering |
| **SVM** | Effective in high dimensions, kernel trick | Slow on large data, hard to interpret |
| **Neural Networks** | Handles complex patterns, flexible | Requires lots of data, hard to tune |

---

## Production Deployment

```python
import joblib

# Save model
joblib.dump(rf, 'random_forest_model.joblib')

# Load model
loaded_rf = joblib.load('random_forest_model.joblib')

# Predict
predictions = loaded_rf.predict(new_data)

# Model size optimization
from sklearn.tree import _tree

def get_model_size(model):
    """Estimate model size in MB"""
    size = 0
    for tree in model.estimators_:
        size += tree.tree_.__sizeof__()
    return size / (1024 * 1024)

print(f"Model size: {get_model_size(rf):.2f} MB")
```

---

## Further Reading

- [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Random Forests Paper (Breiman, 2001)](https://link.springer.com/article/10.1023/A:1010933404324)
- [Understanding Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [Feature Importances with Random Forests](https://explained.ai/rf-importance/)

