#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Machine Learning - Air System Problem
# Goal: Predict air system failures in trucks to optimize
#       preventive maintenance and reduce costs.
#
# Strategy: train on historical data, evaluate on current year
#           to simulate real production deployment.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ============================================================
# CONSTANTS
# ============================================================

COST_CORRECTIVE        = 500   # $ per missed defect (False Negative)
COST_FALSE_ALARM       = 10    # $ per unnecessary maintenance (False Positive)
COST_TRUE_PREVENTIVE   = 25    # $ per correctly flagged defect (True Positive)

LABELS = ['neg', 'pos']


# ============================================================
# 1. LOADING DATA
# ============================================================

previous_years_data = pd.read_csv('air_system_previous_years.csv', skiprows=20, na_values='na')
present_year_data   = pd.read_csv('air_system_present_year.csv',   skiprows=20, na_values='na')

print(f'Data shape (previous): {previous_years_data.shape}')
print(f'Data shape (current):  {present_year_data.shape}')


# ============================================================
# 2. DATA CLEANING
# ============================================================

# Impute missing values with median (single strategy, robust to outliers)
imputer = SimpleImputer(strategy='median')

numeric_cols_prev    = previous_years_data.select_dtypes(include=[np.number]).columns
numeric_cols_present = present_year_data.select_dtypes(include=[np.number]).columns

previous_years_data[numeric_cols_prev] = imputer.fit_transform(
    previous_years_data[numeric_cols_prev]
)
# Use the same imputer fitted on historical data to avoid leakage
present_year_data[numeric_cols_present] = imputer.transform(
    present_year_data[numeric_cols_present]
)

print(f'\nRemaining NaN (previous): {previous_years_data.isna().sum().sum()}')
print(f'Remaining NaN (current):  {present_year_data.isna().sum().sum()}')


# ============================================================
# 3. FEATURE / TARGET SPLIT AND SCALING
# ============================================================

X_prev    = previous_years_data.drop(columns=['class'])
y_prev    = previous_years_data['class']

X_present = present_year_data.drop(columns=['class'])
y_present = present_year_data['class']

# stratify ensures both splits keep the same class proportion (important for imbalanced data)
X_train, X_test_prev, y_train, y_test_prev = train_test_split(
    X_prev, y_prev, test_size=0.3, random_state=42, stratify=y_prev
)
X_test_present, y_test_present = X_present, y_present

# Fit scaler only on training data to avoid leakage into test/present sets
scaler = StandardScaler()
X_train_scaled        = scaler.fit_transform(X_train)
X_test_prev_scaled    = scaler.transform(X_test_prev)
X_test_present_scaled = scaler.transform(X_test_present)


# ============================================================
# 4. MODEL TRAINING AND EVALUATION
# ============================================================

def evaluate_model(name, model, X_train, y_train, test_sets):
    """
    Train model and evaluate on multiple test sets.

    Parameters
    ----------
    name      : display name for the model
    model     : sklearn estimator
    X_train   : training features (scaled)
    y_train   : training labels
    test_sets : list of (label, X_test, y_test) tuples

    Returns
    -------
    model     : fitted model
    results   : dict with accuracy and f1 per test set
    """
    model.fit(X_train, y_train)
    results = {}

    for label, X_test, y_test in test_sets:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, pos_label='pos')

        print(f'\n--- {name} | {label} data ---')
        print(f'Accuracy: {acc:.2f}  |  F1 (pos): {f1:.2f}')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred, labels=LABELS))

        results[label] = {'accuracy': acc, 'f1': f1}

    return model, results


test_sets = [
    ('previous (hold-out)', X_test_prev_scaled,    y_test_prev),
    ('current (present)',   X_test_present_scaled, y_test_present),
]

rf_model,  rf_results  = evaluate_model('RandomForest',     RandomForestClassifier(random_state=42),     X_train_scaled, y_train, test_sets)
svc_model, svc_results = evaluate_model('SVC',              SVC(random_state=42),                        X_train_scaled, y_train, test_sets)
gb_model,  gb_results  = evaluate_model('GradientBoosting', GradientBoostingClassifier(random_state=42), X_train_scaled, y_train, test_sets)

# Summary table
print('\n--- Model Comparison (current year) ---')
for name, results in [('RandomForest', rf_results), ('SVC', svc_results), ('GradientBoosting', gb_results)]:
    r = results['current (present)']
    print(f'{name:<22} Accuracy: {r["accuracy"]:.2f}  F1 (pos): {r["f1"]:.2f}')


# ============================================================
# 5. CONFUSION MATRIX VISUALIZATION (RandomForest / current)
# ============================================================

y_pred_rf_present = rf_model.predict(X_test_present_scaled)
cm = confusion_matrix(y_test_present, y_pred_rf_present, labels=LABELS)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - RandomForest (current data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# ============================================================
# 6. COST ANALYSIS
# ============================================================

# cm was built with labels=LABELS (['neg','pos']), so ravel() is always [tn, fp, fn, tp]
tn, fp, fn, tp = cm.ravel()

cost_corrective_total     = fn * COST_CORRECTIVE
cost_false_alarm_total    = fp * COST_FALSE_ALARM
cost_true_preventive_total = tp * COST_TRUE_PREVENTIVE
total_cost = cost_corrective_total + cost_false_alarm_total + cost_true_preventive_total

print('\n--- Cost Analysis (RandomForest / current data) ---')
print(f'Corrective maintenance   (FN={fn} x ${COST_CORRECTIVE}):  ${cost_corrective_total}')
print(f'Unnecessary preventive   (FP={fp} x ${COST_FALSE_ALARM}):   ${cost_false_alarm_total}')
print(f'Correct preventive       (TP={tp} x ${COST_TRUE_PREVENTIVE}):   ${cost_true_preventive_total}')
print(f'Total estimated cost:                          ${total_cost}')


# ============================================================
# 7. HYPERPARAMETER TUNING (GradientBoosting)
# ============================================================

# Reduced grid for practical runtime on large datasets (60k rows x 171 features)
param_grid = {
    'learning_rate': [0.1, 0.05],
    'n_estimators':  [50, 100],
    'max_depth':     [3, 5],
}

# Use f1 scoring (pos class) — more meaningful than accuracy for imbalanced data
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='f1_macro',
    verbose=1,
    n_jobs=-1,
)

# Sample 20% of training data to speed up GridSearch
X_gs, _, y_gs, _ = train_test_split(X_train_scaled, y_train, train_size=0.2, random_state=42, stratify=y_train)
grid_search.fit(X_gs, y_gs)

print('\n--- GridSearch Results (GradientBoosting) ---')
print('Best parameters:', grid_search.best_params_)
print('Best CV F1 (macro): {:.2f}'.format(grid_search.best_score_))

# Evaluate tuned model on current data
best_gb = grid_search.best_estimator_
y_pred_best_gb = best_gb.predict(X_test_present_scaled)
print('\nTuned GradientBoosting on current data:')
print(classification_report(y_test_present, y_pred_best_gb))
