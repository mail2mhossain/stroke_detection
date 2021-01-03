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

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from tpot import TPOTClassifier
from hpsklearn import HyperoptEstimator, min_max_scaler, standard_scaler, one_hot_encoder, pca
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hyperopt import tpe, hp
from pycaret.classification import * 
from pycaret.datasets import get_data
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import ADASYN  #Its a improved version of Smote
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# ## Load Clean Data

df = pd.read_csv("../../data/processed/stroke_data_processed_important_features.csv")

df.shape

X = df.drop(['stroke'],axis=1)
y = df['stroke']

y = LabelEncoder().fit_transform(y)

# ## Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

print (f"Train Data (X): {X_train.shape}")
print (f"Test Data (X): {X_test.shape}")
print (f"Validation Data (X): {X_valid.shape}")

print (f"Train Data (y): {y_train.shape}")
print (f"Test Data (y): {y_test.shape}")
print (f"Validation Data (y): {y_valid.shape}")


# ## Shape of the Dataset

def get_weight():
    from collections import Counter
    # summarize the shape of the dataset
    print(X.shape)
    counter = Counter(y)
    for k,v in counter.items():
        per = v / len(y) * 100
        print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
    weight = counter[0] / counter[1]
    return (int(weight))


weights = get_weight()
print(weights)


# ## Evaluation Method

# evaluate a model
def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define the model evaluation the metric
    metric = make_scorer(geometric_mean_score)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
    return scores


# ## Find Best Algorithm using AutoML

# + [markdown] heading_collapsed=true
# ### Find Best Algorithm with Best Params Using PyCaret

# + hidden=true
clf = setup(data = df, target = 'stroke', session_id=123,
        train_size = 0.7,
        data_split_stratify = True,
        #sampling = False,
        #categorical_features = cat_features,
        normalize = True, 
        normalize_method = 'minmax',
        #transformation = True, 
        #transformation_method = 'quantile',
        #pca = True,
        #pca_method = 'incremental',
        #feature_selection = True,
        #feature_selection_threshold = 0.7,
        #feature_selection_method = 'boruta',
        #remove_outliers = True,
        #outliers_threshold = 0.05,
        fix_imbalance = True,
        fix_imbalance_method = ADASYN() 
        #profile = True
       )

# + hidden=true
#with mlflow.start_run():
best = compare_models(sort='AUC')

#mlflow.log_metric('accuracy', predictions)
#mlflow.sklearn.log_model(tuned_model, 'model')

# + hidden=true
best_model = finalize_model(best)

# + hidden=true
metric = make_scorer(geometric_mean_score)
tuned_model = tune_model(best_model, optimize = 'AUC') #Accuracy, AUC, Recall, Precision, F1, Kappa, MCC

# + hidden=true
predictions = predict_model(tuned_model)

# + hidden=true
plot_model(tuned_model, plot = 'confusion_matrix')

# + hidden=true
plot_model(tuned_model, plot='feature')

# + hidden=true
# define the data preparation and modeling pipeline
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('m', tuned_model)])

# + hidden=true
tuned_model.fit(X_train, y_train)

# + hidden=true
print(classification_report(y_test, tuned_model.predict(X_test)))

# + hidden=true
plot_confusion_matrix(pipeline, X_test, y_test, values_format='d', display_labels=["Stroke","Not Stroke"]);

# + hidden=true
lr = create_model('lr')

# + hidden=true
tuned_lr = tune_model(lr)

# + hidden=true
tuned_model.fit(X_train, y_train)
print(classification_report(y_test, tuned_model.predict(X_test)))

# + hidden=true
plot_confusion_matrix(tuned_model, X_test, y_test, values_format='d', display_labels=["Stroke","Not Stroke"]);

# + hidden=true


# + [markdown] heading_collapsed=true
# ### Find Best Algorithm with Best Params Using HyperoptEstimator AutoML

# + hidden=true
preproc = hp.choice('myprepros_name', 
                    [
                        [min_max_scaler('myprepros_name.norm')],
                        [standard_scaler('myprepros_name.std_scaler')],
                        [min_max_scaler('myprepros_name.norm2'), standard_scaler('myprepros_name.std_scaler2')]
                    ])

# + hidden=true
#with mlflow.start_run():
model = HyperoptEstimator(  classifier= any_classifier('cla'), 
                            preprocessing= preproc, #any_preprocessing('pre'), 
                            algo=tpe.suggest, 
                            max_evals=50, 
                            trial_timeout=5000)
# perform the search
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

#mlflow.log_params(params)
#mlflow.log_metric('accuracy', accuracy)

# Logging training data
#mlflow.log_artifact(local_path = '../Data/higgs_boson_training.csv')

# Logging training code
#mlflow.log_artifact(local_path = './Base_Model_Selection.ipynb')

# Logging model to MLFlow
#mlflow.sklearn.log_model(model.best_model(), 'model')
#summarize the best model
print("Accuracy: %.3f" % accuracy)
print(model.best_model())
# + hidden=true

# -


# ### Find Best Algorithm with Best Params Using TPOT

# +
#with mlflow.start_run():
# define the model evaluation the metric
metric = make_scorer(geometric_mean_score)
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search
#model = TPOTClassifier(generations=5, population_size=50, cv=cv, scoring='accuracy', 
                       #verbosity=2, random_state=1, n_jobs=-1)
tpot = TPOTClassifier(cv=cv, scoring=metric, 
                       verbosity=2, random_state=1, n_jobs=-1) #max_time_mins=720, 
# perform the search
tpot.fit(X_train, y_train)

predictions = tpot.score(X_test, y_test)

model = tpot.fitted_pipeline_

#mlflow.log_metric('accuracy', predictions)
#mlflow.sklearn.log_model(model, 'model')
# -

#export the best model
tpot.export('tpot_stroke_best_model.py')

print(predictions)

print(model)

print(classification_report(y_test, model.predict(X_test)))

plot_confusion_matrix(model, X_test, y_test, values_format='d', display_labels=["Stroke","Not Stroke"]);


# + [markdown] heading_collapsed=true
# ## Spot Check Balanced Model Algorithms

# + hidden=true
# define models to test
def get_balanced_models():
    models = list()   
    #LR
    models.append(('LR_Bal', LogisticRegression(solver='lbfgs', class_weight='balanced'))) 
    # LDA
    models.append(('LDA', LinearDiscriminantAnalysis()))
    #KNN
    models.append(('KNN', KNeighborsClassifier()))
    #NB
    models.append(('NB', GaussianNB()))
    #MNB
    #models.append(('MNB', MultinomialNB()))
    #GPC
    #models.append(('GPC', GaussianProcessClassifier()))
    if X.shape[0] < 100000:
        #SVM Balanced
        models.append(('SVM_Bal', SVC(gamma='scale', class_weight='balanced')))    
        #SVM Weight
        models.append(('SVM_W', SVC(gamma='scale', class_weight=weights)))    
    #Balanced RF
    models.append(('Bal_RF', BalancedRandomForestClassifier(n_estimators=1000)))      
    #RF
    models.append(('RF_Bal', RandomForestClassifier(n_estimators=1000, class_weight='balanced')))
    #DT
    models.append(('DT_Bal', DecisionTreeClassifier(class_weight='balanced')))
    #Bag
    models.append(('BAG', BaggingClassifier(n_estimators=1000)))
    #XGB
    models.append(('XGB_W', XGBClassifier(scale_pos_weight=weights))) 
    return models
# + hidden=true
init_time = datetime.now()
models = get_balanced_models()
for name, model in models:
    #pipeline = Pipeline(steps=[('prep', col_transform), ('m', model)])
    pipeline = Pipeline(steps=[('norm', MinMaxScaler()), ('m', model)])
    scores = evaluate_model(X, y, pipeline)
    print(f"Geometric mean score of {name}: {np.mean(scores):.3f} ({np.std(scores):.3f})")
fin_time = datetime.now()
print("Execution time : ", (fin_time-init_time))


# + [markdown] heading_collapsed=true
# ## Spot Check Ensemble Algorithms with Over Sampling

# + hidden=true
# define models to test
def get_ensemble_models():
    models = list()
    #LR
    models.append(('LR', LogisticRegression(solver='lbfgs',class_weight='balanced'))) 
    # LDA
    models.append(('LDA', LinearDiscriminantAnalysis()))
    #KNN
    models.append(('KNN', KNeighborsClassifier()))
    #NB
    models.append(('NB', GaussianNB()))
    #MNB
    #models.append(('MNB', MultinomialNB()))
    #GPC
    #models.append(('GPC', GaussianProcessClassifier()))
    #Balanced RF
    models.append(('Bal_RF', BalancedRandomForestClassifier(n_estimators=1000)))
    #DT
    models.append(('DT', DecisionTreeClassifier()))
    #Bag
    models.append(('BAG', BaggingClassifier()))
    #SGD
    models.append(('SGD', SGDClassifier()))
    #ADA
    models.append(('AD', AdaBoostClassifier())) 
    #ET 
    models.append(('ET', ExtraTreesClassifier()))
    #RF
    models.append(('RF', RandomForestClassifier(n_estimators=1000)))
    #GBM
    models.append(('GBM', GradientBoostingClassifier(n_estimators=1000)))
    #XGB
    models.append(('XGB', XGBClassifier()))
    #LGB
    models.append(('LGB', LGBMClassifier()))
    #CAT
    models.append(('CAT', CatBoostClassifier()))
    return models


# + hidden=true
init_time = datetime.now()
models = get_ensemble_models() 
for name, model in models:
    #pipeline = Pipeline(steps=[('prep', col_transform), ('over', RandomOverSampler()), ('m', model)])
    pipeline = Pipeline(steps=[('norm', MinMaxScaler()), ('over', RandomOverSampler()),  ('m', model)])
    scores = evaluate_model(X, y, pipeline)
    print(f"Geometric mean score of {name}: {np.mean(scores):.3f} ({np.std(scores):.3f})")
fin_time = datetime.now()
print("Execution time : ", (fin_time-init_time))


# + [markdown] heading_collapsed=true
# ## Over Sampling

# + hidden=true
# define oversampling models to test
def get_oversampling_models():
    models, names = list(), list()
    # RandomOverSampler
    models.append(RandomOverSampler())
    names.append('ROS')
    # SMOTE
    models.append(SMOTE())
    names.append('SMOTE')
    # BorderlineSMOTE
    models.append(BorderlineSMOTE())
    names.append('BLSMOTE')
    # SVMSMOTE
    models.append(SVMSMOTE())
    names.append('SVMSMOTE')
    # ADASYN
    models.append(ADASYN())
    names.append('ADASYN')
    return models, names


# + hidden=true
# define models
oversamples, names = get_oversampling_models()
# evaluate each model
for i in range(len(oversamples)):
    # define the model
    model = LogisticRegression(solver='lbfgs',class_weight='balanced')
    # define the pipeline steps
    steps = [('prep', col_transform), ('over', oversamples[i]), ('m', model)]
    # define the pipeline
    pipeline = Pipeline(steps=steps)
    # evaluate the model and store results
    scores = evaluate_model(X, y, pipeline)
    # summarize and store
    print(f"Geometric mean score of {names[i]}: {np.mean(scores):.3f} ({np.std(scores):.3f})")


# + [markdown] heading_collapsed=true
# ## Best Model


# + hidden=true
# define the model
model = LogisticRegression(solver='lbfgs', class_weight='balanced')
# define the pipeline steps
steps = [('prep', col_transform), ('norm', MinMaxScaler()), ('power', PowerTransformer()), \
         ('over', RandomOverSampler()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)

# + hidden=true
print (classification_report(y_test, prediction))

# + hidden=true
plot_confusion_matrix(pipeline, X_test, y_test)

# + [markdown] heading_collapsed=true
# ## Parameter Optimization

# + hidden=true



# + hidden=true

