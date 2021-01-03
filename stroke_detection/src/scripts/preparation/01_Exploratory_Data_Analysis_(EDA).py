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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Exploratory Data Analysis (EDA)

# + [markdown] heading_collapsed=true
# ## Import Library

# + hidden=true
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import matplotlib.pylab as plt
import seaborn as sns

# + [markdown] heading_collapsed=true
# ## Environment Variable Setting

# + hidden=true
#pd.set_option('display.width', 100)
pd.set_option('precision', 4)

# + [markdown] heading_collapsed=true
# ## Load Dataset

# + hidden=true
missing_values = ["n/a", "na", "--", "?"]

# + hidden=true
df = pd.read_csv('../../data/raw/stroke_data.csv', na_values = missing_values)

# + [markdown] heading_collapsed=true
# ## Descriptive Statistics 

# + [markdown] heading_collapsed=true hidden=true
# ### Peek at Your Data

# + hidden=true
df.head(5)

# + hidden=true
df.tail(5)

# + hidden=true
df.sample(20)

# + [markdown] heading_collapsed=true hidden=true
# ### Dimensions of Your Data

# + hidden=true
print(df.shape)

# + [markdown] heading_collapsed=true hidden=true
# ### Data Type For Each Attribute

# + hidden=true
df.dtypes

# + hidden=true
df.info()

# + hidden=true
bool_col = df.select_dtypes(['bool']).columns
bool_col

# + hidden=true
object_col = df.select_dtypes(['object']).columns
object_col

# + [markdown] heading_collapsed=true hidden=true
# ### Basic Statistics

# + hidden=true
df.describe()

# + hidden=true
df.describe(include="all").transpose()

# + [markdown] hidden=true
# * Look at the unique values (for non categorical this will show up as NaN). 
# * If a feature has only 1 unique value it will not help to model, so discard it.
# * Look at the ranges of the values. If the max or min of a feature is significantly different from the mean and from the 75% / 25%, then look into this further to understand if these values make sense in their context.

# + hidden=true
#data_fifa.describe(include="all").transpose()
numeric_columns = df.describe(percentiles=[.1, .25, .5, .75, .9], exclude='O').transpose()
numeric_columns

# + hidden=true
df.describe()["bmi"]

# + hidden=true
df.describe()["bmi"]["75%"]

# + hidden=true
numeric_features = numeric_columns.index.to_list()
numeric_features

# + hidden=true
categorical_columns = df.describe(include='O').transpose()
categorical_columns

# + [markdown] hidden=true
# * unique: If a feature has only 1 unique value it will not help to model, so discard it.
# * top: Most commonly occuring value among all values in a column. 
# * freq: Frequency (or count of occurance) of most commonly occuring value among all values in a column. 

# + hidden=true
categorical_features = categorical_columns.index.to_list()
categorical_features

# + [markdown] heading_collapsed=true hidden=true
# ### Duplicate Rows

# + hidden=true
df.duplicated().sum()

# + [markdown] hidden=true
# No duplicate rows are present in the data

# + [markdown] heading_collapsed=true hidden=true
# ### Missing Values 

# + hidden=true
df.isna().sum().sum()

# + hidden=true
df.isnull().sum().sum()

# + hidden=true
null_columns = {}

all_columns = df.isnull().sum().sort_values(ascending=False)
for item in all_columns.index:
    if all_columns[item] > 0:
        null_columns[item] = 100* all_columns[item]/len(df)
        
null_columns

# + [markdown] heading_collapsed=true hidden=true
# ### Minimum = 0, in Numeric Column

# + hidden=true
statistics = df.describe()
min_value_zero_columns = [item for item in statistics if statistics[item]['min'] == 0]
min_value_zero_columns

# + [markdown] heading_collapsed=true hidden=true
# ### Class Distribution (Classification Only)

# + hidden=true
sns.countplot(x='stroke',data=df);

# + hidden=true
df['stroke'].value_counts().size

# + hidden=true
df['stroke'].value_counts()


# + [markdown] hidden=true
# * Not Stroke ->   28517  [Majority/Negative Class]
# * Stroke ->   548  [Minority/Positive Class]

# + [markdown] heading_collapsed=true hidden=true
# #### Weight Calculation

# + [markdown] hidden=true
# weight = total negative examples/total positive examples

# + hidden=true
def calculate_weight():
    y = df['stroke']
    counter = Counter(y)
    for k,v in counter.items():
        per = v / len(y) * 100
        print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
    weight = counter[0] / counter[1]
    return weight


# + hidden=true
weight = calculate_weight()
print(weight)

# + [markdown] hidden=true
# * Most of the contemporary works in class imbalance concentrate on imbalance ratios ranging from 1:4 up to 1:100.

# + [markdown] heading_collapsed=true
# ## Distributions

# + hidden=true
#plt.figure(figsize=(12,6))
sns.pairplot(df);

# + hidden=true
df.hist(bins=50, figsize=(20,15));

# + hidden=true
df.plot(kind='density', subplots=True, layout=(4,3), sharex=False, figsize=(20,15));

# + hidden=true
sns.distplot(df['bmi']);

# + hidden=true
sns.countplot(x='work_type', data=df);

# + [markdown] hidden=true
# #### Skew of Univariate Distributions

# + [markdown] hidden=true
# Skew refers to a distribution that is assumed Gaussian (normal or bell curve) that is shifted or squashed in one direction or another. Many machine learning algorithms assume a Gaussian distribution. Knowing that an attribute has a skew may allow you to perform data preparation to correct the skew and later improve the accuracy of your models.

# + hidden=true
skew = df.skew()
print(skew)

# + [markdown] hidden=true
# The skew results show a positive (right) or negative (left) skew. Values closer to zero show less skew.

# + [markdown] heading_collapsed=true
# ## Outliers

# + hidden=true
df.plot(kind='box', subplots=True, layout=(4,3), sharex=False, sharey=False, figsize=(20,15));

# + hidden=true
plt.figure(figsize=(12,8));
sns.boxplot(x='bmi', data=df);

# + hidden=true
sns.boxplot(y="bmi", x='stroke', data=df, palette='winter');

# + [markdown] heading_collapsed=true
# ## Correlation

# + [markdown] hidden=true
# **Degree of correlation:**
# 1.	**Perfect:** If the value is near ± 1, then it said to be a perfect correlation: as one variable increases, the other variable tends to also increase (if positive) or decrease (if negative).
# 2.	**High degree:** If the coefficient value lies between ± 0.50 and ± 1, then it is said to be a strong correlation.
# 3.	**Moderate degree:** If the value lies between ± 0.30 and ± 0.49, then it is said to be a medium correlation.
# 4.	**Low degree:** When the value lies below + .29, then it is said to be a small correlation.
# 5.	**No correlation:** When the value is zero.

# + hidden=true
# calculate correlation matrix
corr = df.corr()

# + hidden=true
# plot the heatmap
plt.figure(figsize=(15,8))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, 
            cmap=sns.diverging_palette(220, 20, as_cmap=True)); #cmap='viridis'

# + hidden=true
sns.regplot(x="heart_disease", y="stroke", data=df);

# + hidden=true
corr

# + hidden=true
df.corr()['stroke'].sort_values(ascending=False).drop('stroke')

# + hidden=true
pearson_coef, p_value = stats.pearsonr(df['heart_disease'], df['hypertension'])
print(pearson_coef)
print(p_value)

# + hidden=true
pearson_coef, p_value = stats.pearsonr(df['hypertension'], df['avg_glucose_level'])
print(pearson_coef)
print(p_value)

# + [markdown] heading_collapsed=true
# ## Pandas Profiling

# + hidden=true
from pandas_profiling import ProfileReport

# + hidden=true
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

# + hidden=true
profile.to_file("../../outputs/profiling/Stroke_Detection_pandas_profiling.html")

# + hidden=true
profile.to_notebook_iframe()

# + [markdown] heading_collapsed=true
# ## Sweetviz Profiling

# + hidden=true
import sweetviz as sv

# + hidden=true
my_report = sv.analyze(df)
my_report.show_html('../../outputs/profiling/Stroke_Detection_sv_profiling.html') # Default arguments will generate to "SWEETVIZ_REPORT.html"

# + hidden=true
my_report.show_notebook(  w='100%', 
                h='Full', 
                scale=0.8,
                layout='widescreen')

# + hidden=true

