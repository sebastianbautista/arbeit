"""
BANK FAILURES DATA GENERATION

This script converts the Bank Failures SAS dataset
to CSV datasets for use in Python sklearn.

Input: 
    Bank Failures SAS dataset
    
Output: 
    3 Bank Failures csvs:
        Full - full dataset with dummified categorical variables
        Macro - above minus categorical variables
        Nomacro - above minus macro variables 
    3 summary statistics csvs
"""

import pandas as pd
import time
import datetime as dt
import os

start_time = time.perf_counter()
today = dt.datetime.today().strftime('%Y%m%d_%M%H%S')
path = '/sasdata/ra/user/sebastian.bautista/angelina/py'
os.chdir(path)

# Read in data
df_original = pd.read_sas('../data/bankfailures.sas7bdat')
print('Input data set has {} instances, {} features'.format(*df_original.shape))


'''
Data set 0: Selected By The Three
Wow that makes it sound way more dramatic than it is.
'''

selected_df0 = df_original.copy()

logit_selected = ['stchrtr', 'fedchrtr', 'Cap_R', 'size', 'InvSec_TA_R',
                  'Yields_loan_R', 'NII_R', 'HPA_4Q', 'NPL_TA_R', 'asset']

rf_selected = ['bank_type', 'NPL_TA_R', 'NPL_TL_R', 'NIAT_TA_R', 'Cap_R',
               'ROA_R', 'Retearnings', 'Cap_chg_R']

xg_selected = ['asset', 'Ti_ta_R', 'Cap_R', 'AGE', 'NPL_CI_R',
               'stchrtr', 'Ti', 'Cap_chg_R', 'CoreDep_ta_R', 'Retearnings']

selected = list(set(logit_selected + rf_selected + xg_selected))
selected.append('failed_4Q')

selected_df1 = selected_df0[selected]

selected_df = pd.get_dummies(selected_df1, prefix='bank_type', dummy_na=True)

selected_df = selected_df.fillna(0)

# Output to csv
selected_df.to_csv('../data/final_bf_select_{}.csv'.format(today), index=False)
print('Select dataset completed. Shape: ', selected_df.shape)


''' 
Data set 1: Macro
Should have 92 variables
'''

macro_df0 = df_original.copy()

# List variables to drop
macro_dropvars = [
    'ESTYMD', 'Fail_date', 'Failed_last', 'inssave', 'qtr_dn', 'thrift',
    'Resol_type', 'New_bkclas', 'failed_state', 'cert_4Q',
    'name', 'inscoml', 'OBA', 'flag_MSA', 'cert', 'Close_code',
    'bank_type', 'merge_code', 'Failed_yr', 'PEERCODE', 'banksize', 'STALP',
    'qtr_dt'
]

# Put everything into a final dataframe
macro_features = [var for var in macro_df0.columns if var not in macro_dropvars]
macro_df = macro_df0[macro_features]
macro_df.fillna(0, inplace=True)

# Summary statistics - macro
summstats = macro_df.describe(include='all').round().transpose()
summstats.to_csv('../output/summstats_macro_{}.csv'.format(today))

# Output to csv
macro_df.to_csv('../data/final_bf_macro_{}.csv'.format(today), index=False)
print('Macro dataset completed. Should have 92 variables \n', macro_df.shape)

'''
Data set 2: Nomacro
Should have 71 variables
'''

nomacro_df0 = df_original.copy()

# List variables to drop
nomacro_dropvars = [
    'ESTYMD', 'Fail_date', 'Failed_last', 'inssave', 'qtr_dn', 'thrift',
    'Resol_type', 'New_bkclas', 'failed_state', 'cert_4Q',
    'name', 'inscoml', 'OBA', 'flag_MSA', 'cert', 'Close_code',
    'bank_type', 'merge_code', 'Failed_yr', 'PEERCODE', 'banksize', 'STALP',
    'r_inflation', 'chg_ir', 'UR', 'chg_UR', 'index_rent', 'HPI_US',
    'HPA_1y_US', 'HPA_2y_US', 'HPA_5y_US', 'HPA_7y_US', 'HPI_ST',
    'HPA_1y_ST', 'HPA_2y_ST', 'HPA_5y_ST', 'HPA_7y_ST', 'PIPC_ST',
    'UST1yr', 'UST10yr', 'PMMS30yr', 'TED_spread', 'RECESSPROB', 'CASESHILPE_R',
    'qtr_dt'
]

# Put everything into a final dataframe
nomacro_features = [var for var in nomacro_df0.columns if var not in nomacro_dropvars]
nomacro_df = nomacro_df0[nomacro_features]
nomacro_df.fillna(0, inplace=True)

# Summary statistics - macro
summstats = nomacro_df.describe(include='all').round().transpose()
summstats.to_csv('../output/summstats_nomacro_{}.csv'.format(today))

# Output to csv
nomacro_df.to_csv('../data/final_bf_nomacro_{}.csv'.format(today), index=False)
print('Nomacro dataset completed. Should have 71 variables \n', nomacro_df.shape)

'''
Data set 3: Full
Should have 204 variables
'''

full_df0 = df_original.copy()

# Convert SAS date to datetime
import datetime as dt
import pandas as pd
full_df0['qtr_dt'] = pd.to_timedelta(full_df0['qtr_dt'], unit='D') + pd.Timestamp('1960-1-1')

# Create datetime dummies
full_df0['year'] = full_df0['qtr_dt'].dt.year
full_df0['qtr'] = full_df0['qtr_dt'].dt.quarter

# List variables to drop
full_dropvars = [
    'ESTYMD', 'Fail_date', 'Failed_last', 'inssave', 'qtr_dn', 'thrift',
    'Resol_type', 'New_bkclas', 'failed_state', 'cert_4Q',
    'name', 'inscoml', 'OBA', 'flag_MSA', 'cert', 'Close_code',
    'merge_code', 'Failed_yr', 'qtr_dt'
]

full_df0.drop(columns=full_dropvars, inplace=True)

# Create dummies from categorical variables
dummyvars = [
    'bank_type', 'banksize', 'PEERCODE', 'STALP', 'year', 'qtr'
]

# Put everything into final dataframe
full_df = pd.get_dummies(full_df0, prefix=dummyvars, columns=dummyvars, dummy_na=True)
full_df.fillna(0, inplace=True)
    
# Summary statistics - full
summstats = full_df.describe(include='all').round().transpose()
summstats.to_csv('../output/summstats_full_{}.csv'.format(today))

# Output to csv
full_df.to_csv('../data/final_bf_full_{}.csv'.format(today), index=False)
print('Full dataset completed. Should have 204 variables \n', full_df.shape)

stop_time = time.perf_counter()
print('Total runtime: {} mins'.format((stop_time - start_time)/60))
