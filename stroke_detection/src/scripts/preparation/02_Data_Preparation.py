# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/preparation//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: PyCaret_Env
#     language: python
#     name: py_caret_env
# ---

# ## Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datasist.project as dp
import datasist as ds
from pandas_profiling import ProfileReport
import sweetviz as sv
from boruta import BorutaPy
from BorutaShap import BorutaShap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score

# + [markdown] heading_collapsed=true
# ## Load Data

# + hidden=true
#df = dp.get_data('stroke_data.csv', loc='raw', method='csv')

df = pd.read_csv('../../data/raw/stroke_data.csv')

# + hidden=true
ds.structdata.describe(df)

# + hidden=true
Numerical_Features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']

# + hidden=true
Categorical_Features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# + [markdown] heading_collapsed=true
# ## Summary Of EDA

# + hidden=true
df.columns

# + hidden=true
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', \
                        'Residence_type', 'smoking_status']

# + hidden=true
normally_distributed_features = ['age','bmi']

# + hidden=true
skewed_distributed_features = ['avg_glucose_level']

# + [markdown] heading_collapsed=true
# ## Normality Test
#
# * NN Rule: Normalize Non-Gaussian

# + [markdown] hidden=true
# * Standardize Only Gaussian-Like Input Variables
# * Normalize Only Non-Gaussian Input Variables

# + hidden=true
from scipy.stats import shapiro, normaltest


# + hidden=true
def is_normal(data):
    alpha = 0.05
    stat, p = normaltest(data)
    if p > alpha:
        normalTest = True
    else:
        normalTest = False
    
    stat, p = shapiro(data)
    if p > alpha:
        shapiroTest = True
    else:
        shapiroTest = False
        
    return normalTest and shapiroTest


# + hidden=true
numeic_features = ['age','bmi', 'avg_glucose_level']

# + hidden=true
Gaussian_Like = []
Non_Gaussian = []

for i, name in enumerate (numeic_features):
    if is_normal(X[name]):
        Gaussian_Like.append(name)
    else:
        Non_Gaussian.append(name)
        
print (f"Gaussian Like columns: {Gaussian_Like}")
print (f"Non-Gaussian Like columns: {Non_Gaussian}")

# + [markdown] heading_collapsed=true
# ## Data Preprocessing

# + [markdown] heading_collapsed=true hidden=true
# ### Remove Duplicate Data

# + [markdown] hidden=true
# There are no duplicate rows

# + [markdown] heading_collapsed=true hidden=true
# ### Identify and Remove column variables that only have a single value.

# + [markdown] hidden=true
# There is no such column

# + [markdown] heading_collapsed=true hidden=true
# ### Missing Data Imputation

# + [markdown] hidden=true
# There is missing values

# + [markdown] heading_collapsed=true hidden=true
# ### Scaling Data

# + [markdown] hidden=true
# df['norm_amount'] = StandardScaler().fit_transform(np.array(df['Amount']).reshape(-1,1))

# + [markdown] hidden=true
# df['norm_time'] = StandardScaler().fit_transform(np.array(df['Time']).reshape(-1,1))
# df.drop(['Time','Amount'], axis=1, inplace=True)

# + hidden=true
df.head()

# + hidden=true
df.describe()

# + hidden=true
corr = df.corr()

# + hidden=true
#cmap=sns.diverging_palette(220, 20, as_cmap=True)
plt.figure(figsize=(14,8))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, 
            cmap='RdBu'); #cmap='RdBu'

# + hidden=true
sns.distplot(df['bmi']);

# + hidden=true
#sns.set_theme(style="whitegrid")
sns.boxplot(y=df['age']);

# + hidden=true
sns.boxplot(y=df['avg_glucose_level']);

# + hidden=true
sns.boxplot(y=df['bmi']);

# + [markdown] heading_collapsed=true hidden=true
# ### Handling Categorical Columns

# + hidden=true
df_one_hot= pd.get_dummies(df, columns=categorical_features, drop_first=True)


# + hidden=true
df_one_hot.columns

# + [markdown] heading_collapsed=true hidden=true
# ### Handling Target Value

# + hidden=true
X_one_hot = df_one_hot.drop(['stroke'],axis=1)
y = df_one_hot['stroke']

# + hidden=true
y = LabelEncoder().fit_transform(y)

# + [markdown] heading_collapsed=true
# ## Feature Selection

# + hidden=true
X_one_hot.columns

# + hidden=true
X_one_hot.shape

# + [markdown] heading_collapsed=true hidden=true
# ### Boruta

# + hidden=true
forest = RandomForestClassifier(n_jobs = -1, max_depth = 5) 
boruta = BorutaPy(
   estimator = forest, 
   n_estimators = 'auto',
   max_iter = 100 #number of trials to perform
)

# + hidden=true
#fit Boruta (it accepts np.array, not pd.DataFrame)
boruta.fit(np.array(X_one_hot), np.array(y))

# + hidden=true
X_filtered = boruta.transform(np.array(X_one_hot))
X_filtered.shape

# + hidden=true
#print results
green_area = X_one_hot.columns[boruta.support_].to_list()
blue_area = X_one_hot.columns[boruta.support_weak_].to_list()
print('features in the green area:', green_area)
print('features in the blue area:', blue_area)

# + hidden=true
X_filtered = pd.DataFrame(X_filtered)
X_filtered.columns = ['age', 'avg_glucose_level', 'heart_disease_1']

# + hidden=true
print (X_filtered.shape)
X_filtered.head()

# + hidden=true
X_new = X_one_hot[['age', 'avg_glucose_level', 'heart_disease_1']]
X_new.head()

# + [markdown] heading_collapsed=true hidden=true
# ### BorutaShap

# + [markdown] hidden=true
# This initialization takes a maximum of 5 parameters including a tree based model of your choice example a **“Decision Tree” or “XGBoost” or "CatBoost" default is a “Random Forest”**. Which importance metric you would like to evaluate the features importance with either **Shapley values (default) or Gini importance**, A flag to specify if the problem is either classification or regression, a percentile parameter which will take a percentage of the max shadow feature thus making the selector less strict and finally a p-value or significance level which a after a feature will be either rejected or accepted.

# + hidden=true
model_xgb = XGBClassifier(objective='binary:logistic')
Feature_Selector_xgb = BorutaShap(model=model_xgb,
                              importance_measure='shap',
                              classification=True)
Feature_Selector_xgb.fit(X=X_one_hot, y=y, n_trials=25, random_state=0)

# + hidden=true
#Returns a subset of the original data with the selected features
X_subset_xgb = Feature_Selector_xgb.Subset()
print (X_subset_xgb.shape)
X_subset_xgb.head()

# + hidden=true
model_cat = CatBoostClassifier()
Feature_Selector_cat = BorutaShap(model=model_cat,
                              importance_measure='shap',
                              classification=True)
Feature_Selector_cat.fit(X=X_one_hot, y=y, n_trials=25, random_state=0)

# + hidden=true
#Returns a subset of the original data with the selected features
X_subset_cat = Feature_Selector_cat.Subset()
print (X_subset_cat.shape)
X_subset_cat.head()

# + hidden=true
X_cat = X_one_hot[['age', 'ever_married_Yes', 'avg_glucose_level', 'bmi']]

# + [markdown] heading_collapsed=true hidden=true
# ### Custom Data based on Boruta Results

# + hidden=true
X_new = X_one_hot[['age', 'avg_glucose_level', 'heart_disease_1', 'bmi', 'ever_married_Yes']]

# + [markdown] heading_collapsed=true
# ## Train Test Split

# + hidden=true
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.30, stratify=y, random_state=42)


# + [markdown] heading_collapsed=true
# ## Class Ratio Calculation

# + hidden=true
def get_weight():
    from collections import Counter
    # summarize the shape of the dataset
    print(X_subset_xgb.shape)
    counter = Counter(y)
    for k,v in counter.items():
        per = v / len(y) * 100
        print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
    weight = counter[0] / counter[1]
    return (int(weight))


# + hidden=true
weights = get_weight()
print(weights)

# + [markdown] heading_collapsed=true
# ## Data Evaluation

# + [markdown] hidden=true
# define the data preparation for the columns
# t = [('cat', OneHotEncoder(drop='first', sparse=False), categorical_features), \
#      ('min_max', MinMaxScaler(), skewed_distributed_features),\
#      ('scaler', StandardScaler(), normally_distributed_features)]
#     #('power', PowerTransformer(), normally_distributed_features)]
# col_transform = ColumnTransformer(transformers=t)

# + hidden=true
#define the model evaluation the metric
metric = make_scorer(geometric_mean_score)

# + hidden=true
#define the model cross-validation configuration
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# + hidden=true
model = LogisticRegression(solver='lbfgs', class_weight='balanced')
pipeline = Pipeline(steps=[('norm', MinMaxScaler()), ('m', model)])
scores = cross_val_score(pipeline, X_new, y, scoring=metric, cv=cv, n_jobs=-1)
print('Geometric mean score: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# + hidden=true
pipeline.fit(X_train, y_train)

# + hidden=true
print(classification_report(y_test, pipeline.predict(X_test)))

# + hidden=true
plot_confusion_matrix(pipeline, X_test, y_test, values_format='d', display_labels=["Stroke","Not Stroke"]);

# + [markdown] heading_collapsed=true
# ## Save Clean Data

# + hidden=true
df_one_hot.shape

# + hidden=true
df_one_hot['stroke'].head()

# + hidden=true
df_one_hot['stroke'] = y

# + hidden=true
df_one_hot['stroke'].head()

# + hidden=true
df_one_hot.to_csv("../../data/processed/stroke_data_processed_all_features.csv") 

# + hidden=true
df_boruta = df_one_hot[['age', 'avg_glucose_level', 'heart_disease_1', 'bmi', 'ever_married_Yes', 'stroke']]

# + hidden=true
df_boruta.to_csv("../../data/processed/stroke_data_processed_important_features.csv") 

# + hidden=true

