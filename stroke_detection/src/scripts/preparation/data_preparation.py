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
#     display_name: Stroke_Detection
#     language: python
#     name: stroke_detection
# ---

# ## Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import datasist.project as dp
import datasist as ds
from pandas_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold

# ## Load Data

# +
#df = dp.get_data('stroke_data.csv', loc='raw', method='csv')

df = pd.read_csv('../../data/raw/stroke_data.csv')
# -

ds.structdata.describe(df)

Numerical_Features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']

Categorical_Features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# ## Exploratory Data Analysis (EDA)

profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

profile.to_file("../../outputs/profiling/Stroke_Detection_pandas_profiling.html")

df.columns

categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', \
                        'Residence_type', 'smoking_status']

normally_distributed_features = ['age','bmi']

skewed_distributed_features = ['avg_glucose_level']

# ## Data Preprocessing

# + [markdown] heading_collapsed=true
# ### Remove Duplicate Data

# + [markdown] hidden=true
# There are no duplicate rows

# + [markdown] heading_collapsed=true
# ### Identify and Remove column variables that only have a single value.

# + [markdown] hidden=true
# There is no such column

# + [markdown] heading_collapsed=true
# ### Missing Data Imputation

# + [markdown] hidden=true
# There is missing values
# -

# ### Handling Categorical Columns



# ## Train Test Split

X = df.drop(['stroke'],axis=1)
y = df['stroke']

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

# ## Evaluate

# define the data preparation for the columns
t = [('cat', OneHotEncoder(drop='first', sparse=False), categorical_features), \
     ('min_max', MinMaxScaler(), skewed_distributed_features),\
     ('scaler', StandardScaler(), normally_distributed_features)]
col_transform = ColumnTransformer(transformers=t)

# define the model
model = RandomForestClassifier(n_estimators=1000, class_weight='balanced')

# define the data preparation and modeling pipeline
pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])

# define the model cross-validation configuration
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate the pipeline using cross validation and calculate MAE
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)

print('F Score: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


