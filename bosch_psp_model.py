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
import random
import numpy as np
import pandas as pd

#-----------------------------
# Import Global Variables
#-----------------------------
from bpsp_config import data_path, data_dc, url_dc, download, logger, train_sample, test_sample, feature_ranking_sample_dc, Feats_Selector_Classifier, \
                        n_feats_prel, c_feats_prel, label_id, xgb_params

#-----------------------------
# User defined modules and functions
#-----------------------------
import utils
from bpsp_modules import DataDownloader, DataReader, FeatsManipulator, PrelFeatsSelector, Assembler, Classifier, ThresholdOptimizer, OutputHandler

#******************************************************************************
# MAIN PROGRAM
#******************************************************************************

if __name__=='__main__':

    if download:
        DataDownloader().download_from_kaggle(url_dc)
        logger.info('Data downloaded from Kaggle.')
    else:
        logger.info('Download from Kaggle skipped. Using data stored.')
    #end

    df_dc = {}
    df_dc['train_categorical_df'], df_dc['train_numeric_df'], df_dc['train_date_df'] = DataReader().read_train(data_dc, train_sample=train_sample)
    logger.info('Train data successfully read. Sample: ' + str(train_sample))
    df_dc['test_categorical_df'], df_dc['test_numeric_df'], df_dc['test_date_df'] = DataReader().read_test(data_dc, test_sample=test_sample)
    logger.info('Test data successfully read. Sample: ' + str(test_sample))

    full_n_df, full_c_df = FeatsManipulator().preliminary_manipulation(df_dc)
    full_c_df[label_id] = full_n_df[label_id]
    logger.info('Preliminary DataFrame manipulation successful.')

    Prel_Feat_Selector = PrelFeatsSelector(Feats_Selector_Classifier, num_threshold=n_feats_prel, cat_threshold=c_feats_prel, sample_dc=feature_ranking_sample_dc)

    n_feats = [col for col in full_n_df if col not in ['Id', 'is_test']]
    c_feats = [col for col in full_c_df if col not in ['Id', 'is_test']]

    n_feats_ranked = Prel_Feat_Selector.select_feats(full_n_df[n_feats][full_n_df['is_test']==0], label_id=label_id, feat_type='numeric')
    c_feats_ranked = Prel_Feat_Selector.select_feats(full_c_df[c_feats][full_c_df['is_test'].astype(int)==0], label_id=label_id, feat_type='categorical')

    n_df = full_n_df[n_feats_ranked + ['Id', 'is_test', 'Response']]
    c_df = full_c_df[c_feats_ranked + ['Id', 'is_test', 'Response']]
    assembled_train_ar, responses_ar, assembled_test_ar, test_id_ar = Assembler().assemble_train_test(n_df, c_df)

    np.save('./assembled_train_ar', assembled_train_ar)
    np.save('./responses_ar', responses_ar)
    np.save('./assembled_test_ar', assembled_test_ar)
    np.save('./test_id_ar', test_id_ar)

    pred_proba = Classifier().classify(assembled_train_ar, responses_ar, assembled_test_ar, xgb_params, cv=True)

    threshold = ThresholdOptimizer().get_threshold()

    prediction_ls = [0 if pred<threshold else 1 for pred in pred_proba]

    OutputHandler().output_writer(test_id_ar.tolist(), prediction_ls, gz=True)

#end













