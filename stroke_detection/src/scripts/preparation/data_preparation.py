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

# + [markdown] heading_collapsed=true
# ## Import Libraries

# + hidden=true
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
# ## Exploratory Data Analysis (EDA)

# + hidden=true
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

# + hidden=true
profile.to_file("../../outputs/profiling/Stroke_Detection_pandas_profiling.html")

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

# + [markdown] hidden=true
# ### Handling Categorical Columns

# + hidden=true


# + [markdown] heading_collapsed=true
# ## Train Test Split

# + hidden=true
X = df.drop(['stroke'],axis=1)
y = df['stroke']

# + hidden=true
y = LabelEncoder().fit_transform(y)

# + hidden=true
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

# + [markdown] heading_collapsed=true
# ## Pipeline Preparation and Evaluation

# + hidden=true
# define the data preparation for the columns
t = [('cat', OneHotEncoder(drop='first', sparse=False), categorical_features), \
     ('min_max', MinMaxScaler(), skewed_distributed_features),\
     ('scaler', StandardScaler(), normally_distributed_features),\
    ('power', PowerTransformer(), normally_distributed_features)]
col_transform = ColumnTransformer(transformers=t)

# + hidden=true
# define the model
model = RandomForestClassifier(n_estimators=1000, class_weight='balanced')

# + hidden=true
# define the data preparation and modeling pipeline
pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])

# + hidden=true
# define the model cross-validation configuration
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# + hidden=true
# evaluate the pipeline using cross validation and calculate MAE
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)

# + hidden=true
print('F Score: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# + hidden=true

