#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Machine Learning - Air System Problem
# Goal: Predict air system failures in trucks to optimize
#       preventive maintenance and reduce costs.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# 1. LOADING DATA
# ============================================================

previous_years_data = pd.read_csv('air_system_previous_years.csv')
present_year_data = pd.read_csv('air_system_present_year.csv')

print(f'Data shape before cleaning (previous): {previous_years_data.shape}')
print(f'Data shape before cleaning (current):  {present_year_data.shape}')


# ============================================================
# 2. DATA CLEANING
# ============================================================

# Replace string 'na' with NaN
previous_years_data.replace('na', np.nan, inplace=True)
present_year_data.replace('na', np.nan, inplace=True)

# Fill missing values with column mean (numeric columns only)
numeric_cols_prev = previous_years_data.select_dtypes(include=[np.number]).columns
numeric_cols_present = present_year_data.select_dtypes(include=[np.number]).columns

previous_years_data[numeric_cols_prev] = previous_years_data[numeric_cols_prev].fillna(
    previous_years_data[numeric_cols_prev].mean()
)
present_year_data[numeric_cols_present] = present_year_data[numeric_cols_present].fillna(
    present_year_data[numeric_cols_present].mean()
)

# Fallback: fill any remaining NaN with median
previous_years_data.fillna(previous_years_data.median(numeric_only=True), inplace=True)
present_year_data.fillna(present_year_data.median(numeric_only=True), inplace=True)

print(f'\nRemaining NaN (previous): {previous_years_data.isna().sum().sum()}')
print(f'Remaining NaN (current):  {present_year_data.isna().sum().sum()}')


# ============================================================
# 3. FEATURE / TARGET SPLIT AND SCALING
# ============================================================

X_prev = previous_years_data.drop(columns=['class'])
y_prev = previous_years_data['class']

X_present = present_year_data.drop(columns=['class'])
y_present = present_year_data['class']

X_train_prev, X_test_prev, y_train_prev, y_test_prev = train_test_split(
    X_prev, y_prev, test_size=0.3, random_state=42
)
X_train_present, X_test_present, y_train_present, y_test_present = train_test_split(
    X_present, y_present, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_prev_scaled    = scaler.fit_transform(X_train_prev)
X_test_prev_scaled     = scaler.transform(X_test_prev)
X_train_present_scaled = scaler.transform(X_train_present)
X_test_present_scaled  = scaler.transform(X_test_present)


# ============================================================
# 4. MODEL TRAINING AND EVALUATION
# ============================================================

def evaluate_model(name, model, X_train, y_train, X_test_prev, y_test_prev, X_test_present, y_test_present):
    model.fit(X_train, y_train)

    for label, X_test, y_test in [('previous', X_test_prev, y_test_prev),
                                   ('current',  X_test_present, y_test_present)]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f'\n--- {name} | {label} data ---')
        print(f'Accuracy: {acc:.2f}')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))

    return model


rf_model  = evaluate_model('RandomForest',          RandomForestClassifier(random_state=42),
                            X_train_prev_scaled, y_train_prev,
                            X_test_prev_scaled, y_test_prev,
                            X_test_present_scaled, y_test_present)

svc_model = evaluate_model('SVC',                   SVC(random_state=42),
                            X_train_prev_scaled, y_train_prev,
                            X_test_prev_scaled, y_test_prev,
                            X_test_present_scaled, y_test_present)

gb_model  = evaluate_model('GradientBoosting',      GradientBoostingClassifier(random_state=42),
                            X_train_prev_scaled, y_train_prev,
                            X_test_prev_scaled, y_test_prev,
                            X_test_present_scaled, y_test_present)


# ============================================================
# 5. CONFUSION MATRIX VISUALIZATION (RandomForest / current)
# ============================================================

y_pred_rf_present = rf_model.predict(X_test_present_scaled)
cm = confusion_matrix(y_test_present, y_pred_rf_present, labels=['neg', 'pos'])

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

tn, fp, fn, tp = cm.ravel()

cost_corrective              = fn * 500   # missed defects -> corrective maintenance
cost_preventive_no_defect    = fp * 10    # false alarms   -> unnecessary preventive maintenance
cost_preventive_with_defect  = tp * 25    # true positives -> scheduled preventive maintenance

total_cost = cost_corrective + cost_preventive_no_defect + cost_preventive_with_defect

print('\n--- Cost Analysis ---')
print(f'Corrective maintenance (FN x $500):          ${cost_corrective}')
print(f'Preventive - no defect (FP x $10):           ${cost_preventive_no_defect}')
print(f'Preventive - with defect (TP x $25):         ${cost_preventive_with_defect}')
print(f'Total estimated cost:                        ${total_cost}')


# ============================================================
# 7. HYPERPARAMETER TUNING (GradientBoosting)
# ============================================================

param_grid = {
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators':  [50, 100, 200],
    'max_depth':     [3, 5, 7],
}

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
)
grid_search.fit(X_train_prev_scaled, y_train_prev)

print('\n--- GridSearch Results ---')
print('Best parameters:', grid_search.best_params_)
print('Best CV accuracy: {:.2f}'.format(grid_search.best_score_))
