# Insurance-Claims-Fraud-Prevention
Insurance fraud is a significant issue within the insurance industry, leading to billions of dollars in losses each year. The Insurance Claims Fraud Prevention project aims to develop a comprehensive system to identify and prevent fraudulent insurance claims. 
Here's a comprehensive `README` file for your heart attack prediction project. This file provides an overview, instructions for setup and usage, and details on the project's code and functionality.

---

# Heart Attack Prediction Project

## Overview

The Heart Attack Prediction Project aims to develop a machine learning model to predict the likelihood of heart attacks based on insurance claims data. The project involves data loading, exploration, preprocessing, and training various classification models to evaluate their performance.

## Project Structure

The project consists of several stages:

1. **Data Loading and Exploration**: Load the data, explore its structure, and understand its contents.
2. **Data Preprocessing**: Handle missing values, encode categorical variables, and remove less correlated columns.
3. **Model Training and Evaluation**: Train various machine learning models, evaluate their performance, and compare their results.

## Requirements

Ensure you have the following packages installed:
- `pandas`
- `matplotlib`
- `seaborn`
- `numpy`
- `scikit-learn`
- `xgboost`
- `imbalanced-learn` (for BalancedRandomForestClassifier)

You can install the required packages using pip:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn xgboost imbalanced-learn
```

## Usage

### 1. Data Loading

Load the dataset from a CSV file:

```python
import pandas as pd

# Data Loading
datapath = "insurance_claims.csv"
data = pd.read_csv(datapath)
original_data = data.copy()
data.head()
```

### 2. Data Exploration

Explore the dataset to understand its structure:

```python
# Data Exploration
data.columns
data.shape
print("Null Values: " + str(data.isnull().any().sum()))
```

### 3. Data Visualization

Visualize fraud reported statistics and the annual premium based on education level and occupation:

```python
import matplotlib.pyplot as plt

# Fraud Reported Stats
df_count_fraud = data.groupby(['fraud_reported']).count()
df_fraud = df_count_fraud['policy_number']
df_fraud.plot.bar(x='Fraud Reported', y='Count')

# Annual Premium by Education Level
fig, ax = plt.subplots(figsize=(15,7))
df_avg_prem = data.groupby(['insured_education_level', 'fraud_reported']).mean()['policy_annual_premium']
df_avg_prem.unstack().plot(ax=ax)

# Annual Premium by Occupation
fig, ax = plt.subplots(figsize=(15,7))
data.groupby(['insured_occupation', 'fraud_reported']).mean()['total_claim_amount'].unstack().plot(ax=ax)
```

### 4. Data Preprocessing

Handle categorical data and encode it:

```python
# Handle Categorical Data
data.dtypes

# One-hot encoding categorical columns
list_hot_encoded = []
for column in data.columns:
    if data[column].dtypes == object and column != 'fraud_reported':
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        list_hot_encoded.append(column)

# Drop hot-encoded columns
data = data.drop(list_hot_encoded, axis=1)

# Binary encoder for output column
data['fraud_reported'] = data['fraud_reported'].map({'Y': 1, 'N': 0})
```

### 5. Model Training and Evaluation

Train and evaluate different classification models:

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import neighbors, tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Prepare data
y = data['fraud_reported']
X = data.drop(['fraud_reported'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# K-Nearest Neighbors
KNNClassifier = neighbors.KNeighborsClassifier(n_neighbors=12, weights='distance')
KNNClassifier.fit(X=X_train, y=y_train)
print("KNN Score :", KNNClassifier.score(X_test, y_test))

# Decision Tree
DTClassifier = tree.DecisionTreeClassifier()
DTClassifier.fit(X_train, y_train)
print("Decision Tree Score :", DTClassifier.score(X_test, y_test))

# Support Vector Machine
SVMClassifier = SVC(kernel='rbf', probability=True, random_state=42, gamma='auto')
SVMClassifier.fit(X_train, y_train)
print("SVM Score :", SVMClassifier.score(X_test, y_test))

# Random Forest
RFClassifier = RandomForestClassifier()
RFClassifier.fit(X_train, y_train)
print("Random Forest Score :", RFClassifier.score(X_test, y_test))

# Balanced Random Forest
BRFClassifier = BalancedRandomForestClassifier()
BRFClassifier.fit(X_train, y_train)
print("Balanced Random Forest Score :", BRFClassifier.score(X_test, y_test))

# Linear Discriminant Analysis
lda = LDA()
lda.fit(X_train, y_train)
print("Linear Discriminant Analysis Score :", lda.score(X_test, y_test))

# Naive Bayes
NBClassifier = BernoulliNB()
NBClassifier.fit(X_train, y_train)
print("Naive Bayes Classifier Score :", NBClassifier.score(X_test, y_test))

# XGBoost
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train, verbose=False)
print("XGBClassifier Score :", model_xgb.score(X_test, y_test))

# Neural Network
clf_MLP = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(64))
clf_MLP.fit(X_train, y_train)
print("MLPClassifier Score :", clf_MLP.score(X_test, y_test))
```

### 6. Model Comparison

Compare the ROC curves of all models:

```python
# Compare ROC Curves
# (code for plotting ROC curves of all models)

print("The predictive power of each model expressed by ROC curves. For instance, Linear Discriminant Analysis and XGBOOST model has\
        higher probability of accurate prediction of correct class member, and gaining high level of accuracy prediction probability\
        as compared to Random Forest, KNN, Naive Bayes, Neural Network and SVM models.")
```

### 7. Feature Importance (for XGBoost)

Plot feature importance using XGBoost:

```python
def rf_feat_importance(m, df):
    return pd.DataFrame({'feature': df.columns, 'imp': m.feature_importances_}
                        ).sort_values('imp', ascending=False)
def plot_fi(fi): 
    return fi.plot('feature', 'imp', 'barh', figsize=(15,7), legend=False)

fi = rf_feat_importance(model_xgb, X_train)
plot_fi(fi[:30])
```

### 8. Prediction and Ranking

Generate and rank predictions:

```python
test_target = y_test.copy()
test_target.reset_index(drop=True, inplace=True)
test_target = test_target.replace({1: 'Y', 0: 'N'})

predicted_target = model_xgb.predict(X_test)
predicted_target = pd.Series(predicted_target).replace({1: 'Y', 0: 'N'})

ranks = pd.DataFrame({
    'RealClass': test_target,
    'PredictedClass': predicted_target,
    'rank': xgboost_pred_prob
})
ranks.sort_values(by=['rank'], ascending=False, inplace=True)
print(ranks.head())

top = ranks.where(ranks['rank'] > 0.5).dropna()
print(top.head())
```

## Conclusion

The XGBoost model demonstrated the highest performance with the best ROC curve and cross-validated accuracy. Feature importance analysis suggests which features are most significant in predicting heart attack risk.

