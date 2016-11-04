"""
  **************************************
  Created by Romano Foti - rfoti
  On 10/27/2016
  **************************************
"""
#******************************************************************************
# Importing packages
#******************************************************************************
#-----------------------------
# Standard libraries
#-----------------------------
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import make_scorer
#-----------------------------
# User defined modules and functions
#-----------------------------
import utils

#******************************************************************************

#******************************************************************************
# MAIN PROGRAM
#******************************************************************************

data_path = './'
header = True

data_dc = {'train_categorical.csv': {'args':{'dtype': np.str}, 'train': True, 'header': header, 'data_path': data_path},
           'train_numeric.csv': {'args':{'dtype': np.float32}, 'train': True, 'header': header, 'data_path': data_path},
           'train_date.csv': {'args':{'dtype': np.float32}, 'train': True, 'header': header, 'data_path': data_path},
           'test_categorical.csv': {'args':{'dtype': np.str}, 'train': False, 'header': header, 'data_path': data_path},
           'test_numeric.csv': {'args':{'dtype': np.float32}, 'train': False, 'header': header, 'data_path': data_path},
           'test_date.csv': {'args':{'dtype': np.float32}, 'train': False, 'header': header, 'data_path': data_path},
            }

url_dc = {
          'test_categorical': 'https://www.kaggle.com/c/bosch-production-line-performance/download/test_categorical.csv.zip',
          'train_categorical': 'https://www.kaggle.com/c/bosch-production-line-performance/download/train_categorical.csv.zip',
          'test_numeric': 'https://www.kaggle.com/c/bosch-production-line-performance/download/test_numeric.csv.zip',
          'train_numeric': 'https://www.kaggle.com/c/bosch-production-line-performance/download/train_numeric.csv.zip',
          'train_date': 'https://www.kaggle.com/c/bosch-production-line-performance/download/train_date.csv.zip',
          'test_date': 'https://www.kaggle.com/c/bosch-production-line-performance/download/test_date.csv.zip',
          'sample_subm': 'https://www.kaggle.com/c/bosch-production-line-performance/download/sample_submission.csv.zip'
          }

download = False

train_sample = 10000
test_sample = 10000

Feats_Selector_Classifier = RFC(n_estimators=50)
n_feats_prel = 100
c_feats_prel = 10

label_id = 'Response'
score = make_scorer(MCC, greater_is_better=True)

xgb_params = {'params': {
                         'seed': 0,
                         'colsample_bytree': 0.7,
                         'silent': 1,
                         'subsample': 0.7,
                         'learning_rate': 0.1,
                         'objective': 'binary:logistic',
                         'max_depth': 4,
                         'num_parallel_tree': 1,
                         'min_child_weight': 2,
                         'scale_pos_weight': 1,
                         'feval': score,
                         'base_score': 0.05
                         },
               'cross_val': {'num_boost_round': 10, 
                           'nfold': 4, 
                           'seed': 0, 
                           'stratified': True,
                           'early_stopping_rounds': 1,
                           'verbose_eval': 1,
                           'show_stdv': True
                           }
              }

logger = utils.Logging().configure_logger('bpsp_model_logs', './bpsp_logfile.log')

#******************************************************************************