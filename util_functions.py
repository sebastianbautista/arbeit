"""
Utility functions for this project.
At this point, pretty much just data_load.
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split as tts

def data_load(target=1, dataset='full'):
    '''
    Reads in the csv, drops unneeded columns, splits based on X/y and then train/test.
    Defaults to the second target 'failed_4Q' and test size .33, and the full dataset.
    Args:
        target = 0:2 to choose from y_cols
        data = 0, 1, or 2 to choose from full, macro, and nomacro datasets
    '''
    path = '/sasdata/ra/user/sebastian.bautista/angelina/py'
    os.chdir(path)
    
    filename = sorted(glob.glob('../data/*{}*.csv'.format(dataset)), reverse=True)
    df = pd.read_csv(filename[0])
    print('Dataset {} specified. \nReading {}'.format(dataset, filename[0]))
    
    y_cols = ['FAILED', 'failed_4Q', 'Failed_flag']
    useless_cols = [col for col in df.columns if df[col].max() == df[col].min()]
    drop_cols = y_cols + useless_cols
    feature_cols = [col for col in df.columns if col not in drop_cols]
	
    X = df[feature_cols]
    y = df[y_cols[target]]
    print('Features: ', X.columns)
    print('Target variable is {}'.format(y_cols[target]))
    print('{} and {} should be the same length'.format(X.shape, y.shape))
    
    Xtrain, Xtest, ytrain, ytest = tts(X, y, test_size=0.33, random_state=7)
    print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
    return Xtrain, Xtest, ytrain, ytest

if __name__ == '__main__':
    print("Dee Dee, get out of my laboratory!")
