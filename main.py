#!/usr/bin/env python
# coding: utf-8

# #  Machine Learning Air System Problem

# In[1]:


# Importing pandas and numpy
import pandas as pd
import numpy as np


# In[2]:


# Loading the data
previous_years_data = pd.read_csv('air_system_previous_years.csv')
present_year_data = pd.read_csv('air_system_present_year.csv')


# In[3]:


# Replacing 'na' with np.nan
previous_years_data.replace('na', np.nan, inplace=True)
present_year_data.replace('na', np.nan, inplace=True)


# In[51]:


# Checking the amount of data before cleaning
print(f'Amount of data before cleaning (previous data): {previous_years_data.shape}')
print(f'Amount of data before cleaning (current data): {present_year_data.shape}')


# In[5]:


# Filling missing values with column average, for numeric columns only
numeric_columns_prev = previous_years_data.select_dtypes(include=[np.number]).columns
numeric_columns_present = present_year_data.select_dtypes(include=[np.number]).columns


# In[6]:


previous_years_data[numeric_columns_prev] = previous_years_data[numeric_columns_prev].fillna(previous_years_data[numeric_columns_prev].mean())
present_year_data[numeric_columns_present] = present_year_data[numeric_columns_present].fillna(present_year_data[numeric_columns_present].mean())


# In[53]:


# Checking the amount of remaining missing values
print(f'Remaining missing values (previous data):\n{previous_years_data.isna().sum()}')
print(f'Remaining missing values (current data):\n{present_year_data.isna().sum()}')


# In[8]:


#Imputing missing values in a more sophisticated way (example with median)
previous_years_data.fillna(previous_years_data.median(), inplace=True)

present_year_data.fillna(present_year_data.median(), inplace=True)


# In[54]:


# Checking the amount of missing values remaining after filling
print(f'Missing values remaining after filling (previous data):\n{previous_years_data.isna().sum()}')
print(f'Missing values remaining after filling (current data):\n{present_year_data.isna().sum()}')


# # Deploying our machine learning model

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[11]:


X_prev = previous_years_data.drop(columns=['class'])
y_prev = previous_years_data['class']

X_present = present_year_data.drop(columns=['class'])
y_present = present_year_data['class']


# In[12]:


# Splitting data into training and testing
X_train_prev, X_test_prev, y_train_prev, y_test_prev = train_test_split(X_prev, y_prev, test_size=0.3, random_state=42)
X_train_present, X_test_present, y_train_present, y_test_present = train_test_split(X_present, y_present, test_size=0.3, random_state=42)


# In[13]:


# Normalizing the data
scaler = StandardScaler()
X_train_prev_scaled = scaler.fit_transform(X_train_prev)
X_test_prev_scaled = scaler.transform(X_test_prev)


# In[14]:


X_train_present_scaled = scaler.transform(X_train_present)  
X_test_present_scaled = scaler.transform(X_test_present)    


# In[15]:


# Training the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_prev_scaled, y_train_prev)


# In[55]:


# Evaluating the model
y_pred_prev = model.predict(X_test_prev_scaled)
accuracy_prev = accuracy_score(y_test_prev, y_pred_prev)
print(f'Model accuracy on previous data: {accuracy_prev:.2f}')

y_pred_present = model.predict(X_test_present_scaled)
accuracy_present = accuracy_score(y_test_present, y_pred_present)
print(f'Model accuracy on current data: {accuracy_present:.2f}')


# In[56]:


# Other evaluation metrics
print('\nRanking Report (Past Data):\n', classification_report(y_test_prev, y_pred_prev))
print('Ranking report (current data):\n', classification_report(y_test_present, y_pred_present))


# Rating reports provide a detailed look at how your rating model performed on past and current truck air system data. Here are the main points of analysis:
# 
# Previous Data:
# Precision: For the "pos" class (air system defect), the model correctly predicts 91% of the instances that are actually positive. This means that, of the times the model predicts that a truck has an air system defect, it is correct in 91% of cases.
# Recall: The 70% recall for the "pos" class indicates that the model correctly identifies 70% of the positive instances (trucks with a defect in the air system) in relation to the total number of cases that are actually positive.
# F1-score: Harmonic combination of precision and recall. A value of 0.79 for the "pos" class indicates a good balance between precision and recall for that class.
# Accuracy: Generally, the percentage of correct predictions made by the model. In this case, the overall accuracy is 99%, which is very good.
# A value of 0.79 for the "pos" class indicates a good balance between precision and recall for that class.
# 
# 
# Current Data:
# Precision, Recall and F1-score: The values are similar to previous data for the "neg" class (no defect in the air system), indicating that the model maintains consistent performance for this class.
# For the "pos" class (air system defect), the accuracy is a little lower (93%) compared to previous data, but it is still quite high. However, the recall is significantly lower (63%), which suggests that the model is identifying fewer air system defect cases in the current data compared to previous data.
# Accuracy: The overall accuracy is 99%, again indicating that the model is performing well in predicting both classes.
# Analysis: The model appears to be highly accurate in predicting the "neg" class (no air system defect) in both data sets.
# For the "pos" class (air system defect), although the precision is high, the recall is lower in the current data, which may indicate that the model needs adjustments to improve the identification of these specific cases in the current situation.
# 
# 

# In[19]:


from sklearn.metrics import confusion_matrix


# In[20]:


# Calculating the confusion matrix for the current data
cm = confusion_matrix(y_test_present, y_pred_present, labels=['neg', 'pos'])
tn, fp, fn, tp = cm.ravel()


# In[57]:


# Printing confusion matrix
print('Confusion matrix:')
print(cm)


# In[61]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[62]:


# Setting the confusion matrix
cm = np.array([[4695, 5], [37, 63]])


# In[65]:


# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion matrix')
plt.xlabel('Forecast')
plt.ylabel('Real')
plt.show()


# From the confusion matrix, we can observe that:
# 
# The model has a relatively low false positive rate (5 cases), which is positive as it means it is not making many errors in predicting an air system defect where there is none.However, the false negative rate is higher (37 cases), which indicates that the model is missing the identification of some real cases of air system defects.
# 
# This analysis helps to better understand the strengths and weaknesses of the model in relation to detecting defects in the air system. 

# In[58]:


# Confusion matrix values
tn = 4695  # True Negatives
fp = 5   # False Positives
fn = 37    # False Negatives
tp = 63    # True Positives

# Calculating the costs
cost_corrective = fn * 500  
cost_preventive_no_defect = tn * 10  
cost_preventive_with_defect = tp * 25  

# Displaying calculated costs
print(f'Total corrective maintenance cost: ${cost_corrective}')
print(f'Total cost of preventive maintenance without defect in the air system: ${cost_preventive_no_defect}')
print(f'Total cost of preventive maintenance for air system defects: ${cost_preventive_with_defect}')


# Corrective maintenance cost: False Negatives (FN) represent cases where the model did not detect real defects. Therefore, the total cost for corrective maintenance is:
# $500 * 37 = $18500
# 
# Defect-free preventative maintenance cost: False Positives (FP) indicate cases where the model predicted a defect that did not exist. Thus, the total cost for defect-free preventive maintenance is:
# $10 * 5 = $50
# 
# Defect preventive maintenance cost: True Positives (TP) are cases where the model correctly predicted a defect. Therefore, the total cost for defective preventive maintenance is:
# $25 * 63 = $1,575
# 
# Adding these costs, the total maintenance cost using the model is:
# $18500 (corrective maintenance) + $50 (fault-free preventive maintenance) + $1,575 (faulty preventive maintenance) = $20,125
# This is the estimated total cost based on the model's performance in detecting defects in the air system using the provided confusion matrix values.
# This indicates that, based on the model's performance in detecting defects in the vehicles' air system, the estimated costs using the model are lower than the amount spent last year on maintenance. This suggests that the model can potentially help reduce maintenance costs, especially compared to the historical amount spent.

# # Testing other methods
# 

# In[26]:


from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[27]:


# Training the SVC model
svc_model = SVC(random_state=42)
svc_model.fit(X_train_prev_scaled, y_train_prev)


# In[59]:


# Evaluating the SVC model
y_pred_svc_prev = svc_model.predict(X_test_prev_scaled)
print('Assessment of SVC on past data:')
print(classification_report(y_test_prev, y_pred_svc_prev))
print('Confusion Matrix:')
print(confusion_matrix(y_test_prev, y_pred_svc_prev))
accuracy_svc_prev = accuracy_score(y_test_prev, y_pred_svc_prev)
print(f'Accuracy of SVC on previous data: {accuracy_svc_prev:.2f}')

y_pred_svc_present = svc_model.predict(X_test_present_scaled)
print('Assessment of SVC on current data:')
print(classification_report(y_test_present, y_pred_svc_present))
print('Confusion Matrix:')
print(confusion_matrix(y_test_present, y_pred_svc_present))
accuracy_svc_present = accuracy_score(y_test_present, y_pred_svc_present)
print(f'SVC accuracy on current data: {accuracy_svc_present:.2f}')


# Although the SVC has a high overall accuracy, it is important to note that the recall for the 'pos' class is relatively low, especially in the current data, indicating that the model has difficulty identifying all cases of defects in the vehicles' air system. This can affect the model's ability to correctly predict these cases, which is crucial for optimizing preventive maintenance.

# In[30]:


# Training the GradientBoostingClassifier model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_prev_scaled, y_train_prev)


# In[60]:


# Evaluating the GradientBoostingClassifier model
y_pred_gb_prev = gb_model.predict(X_test_prev_scaled)
print('GradientBoostingClassifier evaluation on past data:')
print(classification_report(y_test_prev, y_pred_gb_prev))
print('Confusion Matrix:')
print(confusion_matrix(y_test_prev, y_pred_gb_prev))
accuracy_gb_prev = accuracy_score(y_test_prev, y_pred_gb_prev)
print(f'Accuracy of GradientBoostingClassifier on previous data: {accuracy_gb_prev:.2f}')

y_pred_gb_present = gb_model.predict(X_test_present_scaled)
print('Evaluation of GradientBoostingClassifier on current data:')
print(classification_report(y_test_present, y_pred_gb_present))
print('Confusion Matrix:')
print(confusion_matrix(y_test_present, y_pred_gb_present))
accuracy_gb_present = accuracy_score(y_test_present, y_pred_gb_present)
print(f'Accuracy of GradientBoostingClassifier on current data: {accuracy_gb_present:.2f}')


# GradientBoostingClassifier presents similar results to SVC in terms of overall accuracy, but offers slightly better precision and recall for the 'pos' class in the previous data. In current data, the model also shows solid performance, with an accuracy of 85% for detecting defects in the vehicles' air system.

# # Comparison:
# General Accuracy: All models have high general accuracy (close to 0.99), which indicates that the majority of predictions are correct.
# Precision and Recall for 'pos' (air system defects): RandomForestClassifier had the best precision on previous data (0.91) and current data (0.93) for the 'pos' class. The GradientBoostingClassifier had the best recall on previous data (0.67) and current data (0.57) for the 'pos' class.
# 
# If the main objective is to maximize the correct identification of vehicle air system defects (prioritizing recall), the GradientBoostingClassifier may be preferable due to its higher recall for the 'pos' class.
# If accuracy in identifying defects is crucial (prioritizing precision), then RandomForestClassifier may be more suitable, especially in the current data where it obtained an accuracy of 0.93 for the 'pos' class.

# In[34]:


#perform the hyperparameter optimization 
from sklearn.model_selection import GridSearchCV


# In[35]:


# Define the parameter grid
param_grid = {
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}


# In[36]:


# Initialize the GridSearchCV
grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # Number of cross-validation folds
                           scoring='accuracy',  # Evaluation metric
                           verbose=1,  # Print progress
                           n_jobs=-1)  # Use all available CPU cores


# In[37]:


# Fit the grid search to the data
grid_search.fit(X_train_prev_scaled, y_train_prev)


# In[38]:


# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))


# These results indicate that the GradientBoostingClassifier model with adjusted hyperparameters is capable of achieving high accuracy in cross-validation with training data, which is a good indication that it can perform well when generalizing to new data.
# 

# # Scratchs:
# 
# Model performance in production may vary from the test environment due to changes in operational data or new patterns not seen during training.
# 
# There is a risk of overfitting during training, which can result in poor performance on new data and reduce the reliability of predictions.
# 
# Inconsistent quality or lack of complete data can lead to biased or less accurate predictions.
# Significant resources may be required to implement and maintain the model, including computational resources, ongoing monitoring, and integration with existing systems.
# 
# 
# # Precautions:
# 
# Conduct extensive testing of the model on diverse datasets to ensure robustness and reliability before implementation.
# 
# Implement monitoring mechanisms to track model performance in real time, establishing performance metrics and alerts for deviations.
# 
# Maintain human oversight to review model outputs and intervene when necessary, especially on critical decisions or when model confidence is low.
# 
# Plan periodic updates and re-training of the model with new data to adapt to changing conditions and improve performance over time.
# 
# Develop contingency plans and strategies to mitigate potential failures or unexpected results from model predictions.
# Adotando estas precauções e considerando cuidadosamente os riscos envolvidos, é possível maximizar os benefícios do modelo na otimização do planejamento de manutenção, enquanto se protege contra possíveis desafios e cenários adversos.
# 
# By adopting these precautions and carefully considering the risks involved, you can maximize the benefits of the model in optimizing maintenance planning, while protecting yourself against potential challenges and adverse scenarios.

# # Conclusão
# To maximize defect identification (prioritizing recall), the GradientBoostingClassifier may be preferable due to its higher recall for the 'pos' class.
# If accuracy in identifying defects is crucial (prioritizing accuracy), then RandomForestClassifier may be more suitable, especially given the consistent results in both metrics.
# These results indicate that the models have the potential to optimize maintenance planning, providing a solid basis for informed decisions. The choice of the final model must consider not only performance, but also interpretability, implementation cost and the specific needs of each application within the company's operational context.
