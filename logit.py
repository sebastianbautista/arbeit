"""
Logit

0. Load data
1. Separately scale binary and continuous vars
2. Instantiate SGDClassifier, fit and predict
3. Score and output
"""

import pandas as pd
import time
import datetime as dt
import os; path = '/sasdata/ra/user/sebastian.bautista/angelina/py'
import sys; sys.path.append(path)
import util_functions as uf
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, \
    RandomizedSearchCV
from sklearn.metrics import confusion_matrix, \
    classification_report, precision_recall_curve, average_precision_score, \
    roc_curve, roc_auc_score

start_time = time.perf_counter()
today = dt.datetime.today().strftime('%Y%m%d_%H%M%S')
os.chdir(path)

dataset = 'select'
#dataset = 'full'

file = dataset + '_logit'

class Columns(BaseEstimator, TransformerMixin):
    """ 
    This is a custom transformer for splitting the data into subsets for FeatureUnion.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X[self.columns]

# Data load
Xtrain_raw, Xtest_raw, ytrain, ytest = uf.data_load(dataset=dataset)

# Defining binary and continuous variables for scaling
binary = [col for col in Xtrain_raw.columns if Xtrain_raw[col].nunique() == 2]
cont = [col for col in Xtrain_raw.columns if col not in binary]

# FeatureUnion shifts the ordering so we need to save it here
cols = binary + cont

# Create pipeline
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('binary', Pipeline([
            ('bincols', Columns(columns=binary)),
            ('minmax', MinMaxScaler())
        ])),
        ('continuous', Pipeline([
            ('contcols', Columns(columns=cont)),
            ('scaler', StandardScaler())
        ]))
    ]))   
])

# Fit and transform to create our final Xtrain and Xtest
pipeline.fit(Xtrain_raw)
Xtrain_scaled = pipeline.transform(Xtrain_raw)
Xtest_scaled = pipeline.transform(Xtest_raw)

# Put everything back into dfs
Xtrain = pd.DataFrame(Xtrain_scaled, columns=cols)
Xtest = pd.DataFrame(Xtest_scaled, columns=cols)
print(Xtrain.shape, Xtest.shape)

''' RanSearch '''
l1_ratio = [0, .15, .25, .50, .75, .85, 1]
alpha = [0.001, 0.01, 0.1, 1]
max_iter = [100, 1000, 10000]
learning_rate = ['constant', 'optimal', 'invscaling']
eta0 = [0.01, 0.1, 0.5, 1]

param_distributions = {
    'l1_ratio': l1_ratio,
    'alpha': alpha,
    'max_iter': max_iter,
    'learning_rate': learning_rate,
    'eta0': eta0
    }

sgdc = SGDClassifier(class_weight='balanced', loss='log', penalty='elasticnet')
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
ransearch = RandomizedSearchCV(
    estimator=sgdc,
    param_distributions=param_distributions,
    scoring='f1',
    n_iter=600,
    cv=cv,
    verbose=0,
    n_jobs=15
    )

ransearch.fit(Xtrain, ytrain)
print("Parameters chosen by {}".format(file))
print(ransearch.best_params_)

print("CV score: ", ransearch.best_score_)

best_estimator = ransearch.best_estimator_
joblib.dump(best_estimator, '../models/{}_{}.joblib'.format(file, today))

# Diagnostics
yhat = best_estimator.predict(Xtest)
yprob = best_estimator.predict_proba(Xtest)
yscore = yprob[:,1]

# Classification report
print('Classification report for {}: \n'.format(file))
print(classification_report(ytest, yhat))

# Confusion matrix
cm = confusion_matrix(ytest, yhat)
print('Confusion matrix for {}: \n'.format(file))
print(cm)

# Precision-Recall curve
average_precision = average_precision_score(ytest, yhat)
precision, recall, thresholds = precision_recall_curve(ytest, yhat)
print("Average Precision: ", average_precision)
print("Precision: ", precision)
print("Recall: ", recall)
print("Thresholds: ", thresholds)

# ROC Curve
auroc = roc_auc_score(ytest, yscore)
fpr, tpr, thresholds = roc_curve(ytest, yscore)
print("AUC: ", auroc)
print("FPR: ", fpr)
print("TPR: ", tpr)
print("Thresholds: ", thresholds)

# Feature importances/coefficients
features = Xtest.columns
coef_df = pd.DataFrame(
    data=best_estimator.coef_[0,:],
    index=features
    )
coef_df.to_csv('../output/{}_logit_coefs_{}.csv'.format(file, today))

stop_time = time.perf_counter()
print('Total runtime: {} mins'.format((stop_time - start_time)/60))
