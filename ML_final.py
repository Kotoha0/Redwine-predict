#!/usr/bin/env python
# coding: utf-8

# In[36]:


##https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Read Red Wine Quality dataset
wine = pd.read_csv('winequality-red.csv', sep=',')

# Calculate correlation matrix
#calculated to see how different features in the dataset are correlated with each other.
matrix = wine.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(matrix, dtype=bool))

# Create heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(matrix, mask=mask, annot=True, fmt='.2f', annot_kws={"size": 8}, cmap='bwr')
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Define quality categories
def cate(x):
    if x < 5:
        return "low"
    elif x > 6:
        return "high"
    else:
        return "average"

wine['category'] = wine['quality'].apply(cate)

# Visualize histograms by catecolize
#Easy simplifies the creation of faceted plots, making it easier to visualize and compare the distributions 
frame = wine.melt(id_vars=['category'], value_vars=wine.columns[:-2], var_name='var', value_name='value')
face = sns.FacetGrid(frame, col='var', col_wrap=4, hue="category", palette="muted", height=2.0, sharex=False, sharey=False)
face = (face.map(sns.kdeplot, "value"))
plt.legend(loc='upper right', bbox_to_anchor=(0, 1))
plt.show()

# Split data into training and testing sets
target = ['quality', 'category']
X = wine.drop(target, axis=1)
y = wine['category']
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
#25% of the dataset will be used as the test set and 75% will be used as the training set
#Ensures you get the same data split every time you run the code, which helps in comparing results and debugging.

# Encode target labels
#convert the categorical values into numerical values suitable for machine learning models.
label = LabelEncoder()
y_train = label.fit_transform(y_train)
y_test = label.transform(y_test)


# Random Forest Classifier without GridSearchCV
#The parameter grid defined for GridSearchCV might be too limited, or the actual optimal parameters are outside the tested range??
model_rf_no_grid = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
model_rf_no_grid.fit(x_train, y_train)
rf_pred_no_grid = model_rf_no_grid.predict(x_test)
rf_acc_no_grid = accuracy_score(y_test, rf_pred_no_grid)

# Random Forest Classifier with GridSearchCV
#For smaller datasets and parameter grids, this method is feasible and often effective in finding optimal hyperparameters.
#GridSearchCV tests different combinations to find the best settings for highest accuracy.
model_rf_grid = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 300, 400, 500],# The number of trees in the forest. More trees generally improve accuracy
    'max_depth': [None, 10, 20, 30, 40, 50]#Limitation of tree depth. Prevents overfitting or underfitting.
}
grid_rf = GridSearchCV(estimator=model_rf_grid, param_grid=param_grid_rf, cv=5, verbose=2, n_jobs=-1)
result_rf = grid_rf.fit(x_train, y_train)
rf_pred_grid = result_rf.predict(x_test)
rf_acc_grid = accuracy_score(y_test, rf_pred_grid)

# Store results for later display
best_rf_params = result_rf.best_params_
best_rf_score = result_rf.best_score_

# Visualize feature importances
plt.figure(figsize=(8, 6))
plt.bar(X.columns, result_rf.best_estimator_.feature_importances_, color='red')
plt.xticks(rotation=45, fontsize=10, ha='right')
plt.title('Feature Importances (Random Forest)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Print Random Forest results
print("\nRandom Forest model:")
print(f"Best Parameters : {best_rf_params}")
print(f"Best Accuracy with GridSearchCV: {best_rf_score * 100:.2f}%")
print(f"Best Accuracy without GridSearchCV: {rf_acc_no_grid * 100:.2f}%")

# Decision Tree Classifier without GridSearchCV
model_dt_no_grid = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
model_dt_no_grid.fit(x_train, y_train)
dt_pred_no_grid = model_dt_no_grid.predict(x_test)
dt_acc_no_grid = accuracy_score(y_test, dt_pred_no_grid)

# Decision Tree Classifier with GridSearchCV
model_dt_grid = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'criterion': ['entropy', 'gini'],#Chose spliting method
    'max_depth': [None, 5, 10, 15, 20, 25, 30]#Limitation of tree depth. Prevents overfitting or underfitting.
}
grid_dt = GridSearchCV(model_dt_grid, param_grid=param_grid_dt, cv=5, verbose=2, n_jobs=-1)
grid_dt.fit(x_train, y_train)
dt_pred_grid = grid_dt.predict(x_test)
dt_acc_grid = accuracy_score(y_test, dt_pred_grid)

# Store results for later display
best_dt_params = grid_dt.best_params_
best_dt_score = grid_dt.best_score_

# Print Decision Tree results
print("\nDecision Tree model:")
print(f"Best Parameters : {best_dt_params}")
print(f"Best Accuracy with GridSearchCV: {best_dt_score * 100:.2f}%")
print(f"Best Accuracy without GridSearchCV: {dt_acc_no_grid * 100:.2f}%")

# Plotting the comparison graph
accuracy_data = {
    'Model': ['Random Forest (No GridSearch)', 'Random Forest (GridSearch)', 
              'Decision Tree (No GridSearch)', 'Decision Tree (GridSearch)'],
    'Accuracy': [rf_acc_no_grid * 100, rf_acc_grid * 100, dt_acc_no_grid * 100, dt_acc_grid * 100]
}

accuracy_df = pd.DataFrame(accuracy_data)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=accuracy_df, palette='viridis')
plt.title('Comparison of Model Accuracies with and without GridSearchCV')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.ylim(75, 90)  # Adjust the y-axis limit based on your accuracy values
plt.show()

# Random Forest Classifier with Grid Search Results
print("Random Forest Results:")
rf_results_df = pd.DataFrame(grid_rf.cv_results_)
plt.figure(figsize=(10, 6))
rf_results_pivot = rf_results_df.pivot(index='param_n_estimators', columns='param_max_depth', values='mean_test_score')
sns.heatmap(rf_results_pivot, annot=True, cmap='YlGnBu', fmt='.3f', cbar=True)
plt.title('Random Forest Accuracy Heatmap')
plt.xlabel('Max Depth')
plt.ylabel('Number of Estimators')
plt.xticks(np.arange(len(param_grid_rf['max_depth'])), param_grid_rf['max_depth'])
plt.yticks(np.arange(len(param_grid_rf['n_estimators'])), param_grid_rf['n_estimators'])
plt.show()

# Decision Tree Classifier with Grid Search Results
print("\nDecision Tree Results:")
dt_results_df = pd.DataFrame(grid_dt.cv_results_)
plt.figure(figsize=(12, 8))
dt_results_pivot = dt_results_df.pivot(index='param_criterion', columns='param_max_depth', values='mean_test_score')
sns.heatmap(dt_results_pivot, annot=True, cmap='YlGnBu', fmt='.3f', cbar=True)
plt.title('Decision Tree Accuracy Heatmap')
plt.xlabel('Max Depth')
plt.ylabel('Criterion')
plt.xticks(np.arange(len(param_grid_dt['max_depth'])), param_grid_dt['max_depth'])
plt.yticks(np.arange(len(param_grid_dt['criterion'])), param_grid_dt['criterion'])
plt.show()
'''
# Visualize decision trees from the Random Forest Classifier
plt.figure(figsize=(38, 17))
plot_tree(result_rf.best_estimator_.estimators_[0], feature_names=X.columns.tolist(), class_names=label.classes_.tolist(), filled=True)
plt.title('Decision Tree from Random Forest Classifier', fontsize=50)
plt.show()

# Visualize the decision tree
plt.figure(figsize=(18, 10))
plot_tree(new_dc, feature_names=X.columns.tolist(), class_names=label.classes_.tolist(), filled=True)
plt.title('Decision Tree ', fontsize=20)
plt.show()

'''


# In[ ]:




