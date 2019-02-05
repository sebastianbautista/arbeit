"""
BANK FAILURES MODEL TUNING

0. Read in data 
1. Fit model using RandomizedSearchCV
2. Print chosen hyperparameters and pickle model
3. Print diagnostics
4. Generate feature importances csv

Input = final bf csv
Output = pickled model after RandomizedSearchCV, 
    diagnostics, and feature importances csv
"""

import numpy as np
import pandas as pd
import time
import datetime as dt
import os; path = '/sasdata/ra/user/sebastian.bautista/angelina/py'
import sys; sys.path.append(path)
import util_functions as uf
from sklearn.model_selection import StratifiedKFold, \
    GridSearchCV
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, \
    classification_report, precision_recall_curve, \
    average_precision_score, roc_curve, fbeta_score, make_scorer

def main():
    my_script = sys.argv[0]
    print('Printing from', my_script)

    start_time = time.perf_counter()
    today = dt.datetime.today().strftime('%Y%m%d_%H%M%S')
    os.chdir(path)

    model = str(sys.argv[1])
    file = 'full_' + model

    # Read in data   
    Xtrain, Xtest, ytrain, ytest = uf.data_load(dataset='full')

    # Parameter distributions for searching over
    rf_params = {
            'n_estimators': [int(x) for x in np.linspace(1200, 1800, 5)],
            'max_depth': [40, 50, 60, 70],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', 'balanced_subsample'],
            } # 80
        
    gb_params = {
            'learning_rate': [.01, .1, 1.0, 5.0],
            'n_estimators': [50, 150, 250, 400],
            'subsample': [.1, 1],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_depth': [2, 5],
            'max_features': ['sqrt', None],
            } # 512
        
    ada_params = {
            'n_estimators': [int(x) for x in np.linspace(50, 200, 5)],
            'learning_rate': [.01, 1, 2],
            'base_estimator': [dtc(max_depth=1), dtc(max_depth=2)],
            } # 30 
    
    xg_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.001, 0.01, 0.1],
            'gamma': [0, 0.5, 1, 5, 10],
            'subsample': [0.5, 1],
            } # 270
    
    param_distributions = {
            'rf': rf_params,
            'gb': gb_params,
            'ada': ada_params,
            'xg': xg_params,
            }
        
    # Models to choose from
    estimator = {
        'rf': RandomForestClassifier(
                max_features='sqrt',
                criterion='gini', 
                min_samples_split=5,
                bootstrap=False,
                ),
        'gb': GradientBoostingClassifier(),
        'ada': AdaBoostClassifier(algorithm='SAMME.R'),
        'xg': XGBClassifier(
                booster='gbtree',
                scale_pos_weight=644,
                )
        }
        
    # Checking print
    print("Model: \n {}".format(estimator[model]))
    print("Hyperparameters to search over: \n {}".format(param_distributions[model]))
    
    # Make scorer - f3
    f3 = make_scorer(fbeta_score, beta=3)
    
    # Search with 3-fold cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    search = GridSearchCV(
        estimator=estimator[model],
        param_grid=param_distributions[model],
        scoring=f3,
        cv=cv,
        verbose=1,
        n_jobs=15,
        )

    # Fit the search
    search.fit(Xtrain, ytrain)

    # Print chosen hyperparameters
    print("Parameters chosen by RandomizedSearchCV for {}".format(file))
    print(search.best_params_)

    # Print CV score
    print("Cross-validation score for {}".format(file))
    print(search.best_score_)

    # Pickle the fitted model
    best_estimator = search.best_estimator_
    joblib.dump(best_estimator, '../models/{}_{}.joblib'.format(file, today))

    # Print diagnostics
    yhat = best_estimator.predict(Xtest)
    yscore = best_estimator.predict_proba(Xtest)[:,1]
    print('Classification report for {}: \n'.format(file))
    print(classification_report(ytest, yhat))
    
    cm = confusion_matrix(ytest, yhat)
    print('Confusion matrix for {}: \n'.format(file))
    print(cm)

    average_precision = average_precision_score(ytest, yscore)
    precision, recall, thresholds = precision_recall_curve(ytest, yscore)
    print("Average Precision: {}".format(average_precision))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Thresholds: {}".format(thresholds))
    
    fpr, tpr, thresholds = roc_curve(ytest, yscore)
    print('fpr ', fpr)
    print('tpr ', tpr)
    print('thresholds ', thresholds)

    # Generate feature importances csv
    coef = pd.DataFrame(
        best_estimator.feature_importances_,
        index=Xtrain.columns,
        columns=['importance'],).sort_values('importance', ascending=False)
        
    coef.to_csv('../output/{}_feature_importances_{}.csv'.format(file, today))

    stop_time = time.perf_counter()
    print('Total runtime: {} hours'.format((stop_time-start_time)/3600))

if __name__ == '__main__':
    main()