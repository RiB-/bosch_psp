"""
  **************************************
  Created by Romano Foti - rfoti
  On 10/21/2016
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
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Imputer
import xgboost as xgb
#-----------------------------
# User defined modules and functions
#-----------------------------
import kaggle_utils
import utils
from bpsp_config import logger

#******************************************************************************

#******************************************************************************
# Defining functions
#******************************************************************************

class DataDownloader():
    '''
    '''

    def __init__(self):
        pass
    #end

    def download_from_kaggle(self, url_dc=None):
        '''
        Downloads and unzips datasets from Kaggle

        '''
        if url_dc==None:      
            logger.info('Dictionary of downloading URLs needs to be provided!')
        #end
        for ds, url in zip(url_dc.keys(), url_dc.values()):
            logger.info('Downloading and unzipping %s ...' %ds)
            kaggle_utils.KaggleRequest().retrieve_dataset(url)
        #end
        return
    #end

#end

class DataReader():
    '''
    '''
    def __init__(self):
        pass
    #end

    def sample_skip(self, filename, samplesize=1000, header=True):
        '''
        Reads a random sample of lines from a csv file and
        loads it into a pandas DataFrame
        '''
        num_records = sum(1 for line in open(filename)) - int(header)
        skip = sorted(random.sample(xrange(int(header), num_records+1), num_records-samplesize))
        return skip
    #end

    def read_train(self, data_dc, train_sample=None):
        if train_sample!=None:
            skip = self.sample_skip('./train_categorical.csv', train_sample, data_dc['train_categorical.csv']['header'])
            for datasetname, dataset_attr in data_dc.iteritems():
                if dataset_attr['train']:
                    dataset_attr['args']['skiprows'] = skip
                #end
            #end
        #end
        train_categorical_df = None #pd.read_csv('./train_categorical.csv', **data_dc['train_categorical.csv']['args'])
        train_numeric_df = pd.read_csv('./train_numeric.csv', **data_dc['train_numeric.csv']['args'])
        train_date_df = pd.read_csv('./train_date.csv', **data_dc['train_date.csv']['args'])
        return train_categorical_df, train_numeric_df, train_date_df
    #end

    def read_test(self, data_dc, test_sample=None):
        if test_sample!=None:
            skip = self.sample_skip('./test_categorical.csv', test_sample, data_dc['test_categorical.csv']['header'])
            for datasetname, dataset_attr in data_dc.iteritems():
                if not dataset_attr['train']:
                    dataset_attr['args']['skiprows'] = skip
                #end
            #end
        #end
        test_categorical_df = None #pd.read_csv('./test_categorical.csv', **data_dc['test_categorical.csv']['args'])
        test_numeric_df = pd.read_csv('./test_numeric.csv', **data_dc['test_numeric.csv']['args'])
        test_date_df = pd.read_csv('./test_date.csv', **data_dc['test_date.csv']['args'])
        return test_categorical_df, test_numeric_df, test_date_df
    #end

#end

class FeatsManipulator():
    '''
    '''

    def __init__(self):
        pass
    #end

    def rebalancer(self, df, label_id='Response', balance_ratio=0.05, sample_index=None):
        '''
        '''
        logger.info('Rebalancing Train Datasets...')
        logger.info('Original dataset shape %s' %str(df.shape))
        if isinstance(sample_index, pd.Int64Index):
            df = df.ix[sample_index]
        else:
            one_count = len(df[df[label_id]==1])
            sample_n = ((1 - balance_ratio) / balance_ratio) * one_count
            df = pd.concat((df[df[label_id]==1], df[df[label_id]==0].sample(n=int(sample_n))))
        #end
        ind = df.index
        logger.info('Dataset rebalanced.')
        logger.info('Rebalanced dataset shape %s' %str(df.shape))
        return df.reset_index(drop=True), ind
    #end

    def engineer_bool(self, bool_sr):
        '''
        Turne boolean series into binary
        '''
        return bool_sr.astype(int)
    #end

    def engineer_str(self, string_sr, fill_na='NaN'):
        '''
        Fills missing strings with 'NaN' in string type series
        '''
        string_sr.fillna(fill_na, inplace=True)
        return string_sr
    #end

    def engineer_float(self, float_sr, fill_na=0.0):
        '''
        Fills missing np.numbers with 0.0 in numpy type series
        '''
        float_sr.fillna(fill_na, inplace=True)
        return float_sr
    #end

    def low_st_remover(self, df, threshold=0.0):
        '''
        '''
        return df.loc[:, df.std()>threshold]
    #end

    def float_imputer(self, df, exclude_col_ls=None, Imputer_obj=None, strategy='mean'):
        '''
        '''
        logger.info('Imputing numeric columns...')
        if exclude_col_ls==None:
            exclude_col_ls = []
        #end
        col_ls = [col for col in df.columns if col not in exclude_col_ls]
        if Imputer_obj==None:
            Imputer_obj = Imputer(strategy=strategy, axis=0)
            try:
                logger.info('Imputing...')
                df[col_ls] = Imputer_obj.fit_transform(df[col_ls])
            except:
                logger.info('Something went wrong with the imputation.')
            #end
        else:
            df[col_ls] = Imputer_obj.transform(df[col_ls])
        #end
        return df, Imputer_obj
    #end

    def float_scaler(self, df, exclude_col_ls=None, Scaler_obj=None):
        '''
        '''
        logger.info('Performing scaling of numeric features...')
        if exclude_col_ls==None:
            exclude_col_ls = []
        #end
        col_ls = [col for col in df.columns if col not in exclude_col_ls]
        if Scaler_obj==None:
            Scaler_obj = StandardScaler()
            df[col_ls] = Scaler_obj.fit_transform(df[col_ls])
        else:
            df[col_ls] = Scaler_obj.transform(df[col_ls])
        #end
        return df, Scaler_obj
    #end

    def unpack_date(self, df):
        '''
        Creates new DF columns with year, month, day and weekend
        '''
        logger.info('Unpacking data column...')
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['is_weekend'] = (df['date'].dt.weekday>=5).astype(int)
        #end
        return df
    #end

    def merge_uncommon(self, string_sr, replace_with='uncommon', threshold=1, max_f=50):
        '''
        Finds all unique or uncommon strings in string series and replace them with an identifier
        with the purpose to merge all string entries that only occur few times
        '''
        if threshold==1:
            for index, el in string_sr.duplicated(keep=False).iteritems():
                if el==False:
                    string_sr.set_value(index,replace_with)
                #end
            #end
        else:
            for unique in string_sr.unique():
                if len(string_sr[string_sr==unique])<=threshold:
                    string_sr[string_sr==unique] = replace_with
                #end
            #end
        #end
        return string_sr
    #end

    def add_timestamp_cols(self, train_df, test_df, train_date_df, test_date_df, non_feats_ls):
        '''
        '''
        feats_cols = np.setdiff1d(train_date_df.columns, non_feats_ls)
        train_df['begin'] = train_date_df[feats_cols].min(axis=1)
        test_df['begin'] = test_date_df[feats_cols].min(axis=1)
        train_df['end'] = train_date_df[feats_cols].max(axis=1)
        test_df['end'] = test_date_df[feats_cols].max(axis=1)
        train_df['duration'] = train_df['end'] - train_df['begin']
        test_df['duration'] = test_df['end'] - test_df['begin']
        return train_df, test_df
    #end

    def add_test_flag_and_merge(self, train_df, test_df, flag_type='int'):
        train_df['is_test'] = 0
        test_df['is_test'] = 1
        full_df = pd.concat((train_df, test_df)).reset_index(drop=True)
        full_df['is_test'] = full_df['is_test'].astype(flag_type)
        return full_df
    #end

    def add_index(self, df):
        return df.reset_index()
    #end

    def drop_index(self, df):
        return df.sort_values(by=['index']).drop(['index'], axis=1)
    #end

    def add_leaks(self, full_n_df):
        full_n_df = self.add_index(full_n_df)
        full_n_df['leak_01'] = full_n_df['Id'].diff().fillna(9999999).astype(int)
        full_n_df['leak_02'] = full_n_df['Id'].iloc[::-1].diff().fillna(9999999).astype(int)
        full_n_df = full_n_df.sort_values(by=['begin', 'Id'], ascending=True)
        full_n_df['leak_03'] = full_n_df['Id'].diff().fillna(9999999).astype(int)
        full_n_df['leak_04'] = full_n_df['Id'].iloc[::-1].diff().fillna(9999999).astype(int)
        full_n_df = self.drop_index(full_n_df)
        return full_n_df
    #end

    def data_cleansing(self, df, exclude_ls):
        '''
        '''
        logger.info('Performing final cleaning of datasets...')
        if exclude_ls==None:
            exclude_ls = []
        #end

        cols_ls = [col for col in df.columns if col not in exclude_ls]
        for col in cols_ls:
            if df[col].dtype == 'object':
                df[col] = self.engineer_str(df[col]) # engineer string columms
                df[col] = self.merge_uncommon(df[col]) # consolidate all uncommon labels into one
            elif df[col].dtype == 'bool':
                df[col] = self.engineer_bool(df[col]) #engineer boolean columns
            elif np.issubdtype(df[col].dtype, np.number):
                df[col] = self.engineer_float(df[col])
            #end
        #end
        return df
    #end

    def preliminary_manipulation(self, df_dc, categorical=True):
        '''
        '''
        logger.info('Performing a preliminary manipulation of the datasets...')
        n_train_df, n_test_df = self.add_timestamp_cols(df_dc['train_numeric_df'], df_dc['test_numeric_df'], 
                                                        df_dc['train_date_df'], df_dc['test_date_df'],
                                                        non_feats_ls=['Id'])
        n_train_df = self.low_st_remover(n_train_df, threshold=0.0)
        n_test_df = n_test_df[[col for col in n_train_df.columns if col not in ['Response']]]
        n_train_df, balance_ind = self.rebalancer(n_train_df, label_id='Response', balance_ratio=0.01)
        #n_train_df, Imputer_obj = self.float_imputer(n_train_df, exclude_col_ls=['Id', 'Response'], Imputer_obj=None, strategy='mean')
        #n_test_df, Imputer_obj = self.float_imputer(n_test_df, exclude_col_ls=['Id', 'Response'], Imputer_obj=Imputer_obj, strategy='mean')
        #n_train_df, Scaler_obj = self.float_scaler(n_train_df, exclude_col_ls=['Id', 'Response'], Scaler_obj=None)
        #n_test_df, Scaler_obj = self.float_scaler(n_test_df, exclude_col_ls=['Id', 'Response'], Scaler_obj=Scaler_obj)

        full_n_df = self.add_test_flag_and_merge(n_train_df, n_test_df, flag_type='int')
        full_n_df = self.add_leaks(full_n_df)
        full_n_df = self.data_cleansing(full_n_df, exclude_ls=['Id', 'is_test', 'Response'])

        if categorical:
            c_train_df, c_test_df = df_dc['train_categorical_df'], df_dc['test_categorical_df']
            c_train_df, balance_ind = self.rebalancer(c_train_df, label_id='Response', sample_index=balance_ind)
            full_c_df = self.add_test_flag_and_merge(c_train_df, c_test_df, flag_type='str')
            full_c_df = self.data_cleansing(full_c_df, exclude_ls=['Id', 'is_test'])
            full_c_df = full_c_df.sort_values(by='Id')
        else:
            full_c_df = None
        #end
        logger.info('Preliminary manipulation of datasets performed.')
        full_n_df = full_n_df.sort_values(by='Id')
        return full_n_df, full_c_df
    #end

#end

class CatEncoder():
    '''
    '''

    def __init__(self):
        pass
    #end

    def encode_labels(self, df, cols2encode_ls=None, encode_1by1=False):
        if not cols2encode_ls:
            cols2encode_ls = list(df.select_dtypes(include=['category','object']))
        #end
        if encode_1by1:
            le = LabelEncoder()
            for feature in cols2encode_ls:
                try:
                    transformed_df[feature] = le.fit_transform(df[feature])
                except:
                    print('Error encoding '+ feature)
                #end
            #end
            return le, transformed_df
        else:
            le_dc = defaultdict(LabelEncoder)
            transformed_df = df.apply(lambda x: le_dc[x.name].fit_transform(x))
            return le_dc, transformed_df
        #end
    #end

    def fit_onehot_to_cat(self, df, cols2encode_ls=None):
        if not cols2encode_ls:
            cols2encode_ls = list(df.select_dtypes(include=['category','object']))
        #end
        le_dc, le_df = self.encode_labels(df, cols2encode_ls=cols2encode_ls, encode_1by1=False) #call label encoder
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe = ohe.fit(le_df, cols2encode_ls)
        return le_dc, ohe
    #end

    def transform_onehot(self, df, le_dc, ohe):
        le_df = df.apply(lambda x: le_dc[x.name].transform(x))
        return ohe.transform(le_df)
    #end

#end

class PrelFeatsSelector():
    '''
    '''
    def __init__(self, Classifier, num_threshold=100, cat_threshold=10, sample_dc=None, best=False):
        self.Classifier = Classifier
        self.n_thresh = num_threshold
        self.c_thresh = cat_threshold
        self.best = best
        self.sample_dc = self.build_sample_dc(sample_dc)
    #end

    def build_sample_dc(self, sample_dc):
        if sample_dc==None:
            sample_dc = {}
            sample_dc['numeric'], sample_dc['categorical'] = 1.0, 1.0
            logger.info('Preliminary feature selection will be performed without downsampling of datasets.')
        else:
            logger.info('Sampling for feature selection: %s numeric and %s categorical' % (str(sample_dc['numeric']), str(sample_dc['categorical'])))
        #end
        return sample_dc
    #end

    def num_feats_selector(self, feats_df, label_id, threshold):
        '''
        '''
        score = make_scorer(MCC, greater_is_better=True)
        rfecv = RFECV(estimator=self.Classifier, step=10, cv=StratifiedKFold(3), scoring=score)
        feat_ls = [col for col in feats_df.columns if col!=label_id]
        feats_df = feats_df.sample(frac=self.sample_dc['numeric'])
        feat_ar = np.array(feats_df[feat_ls])
        label_ar = feats_df[label_id].ravel()
        rfecv.fit(feat_ar, label_ar)
        if self.best:
            ranked_feats_idx = [idx for idx, rank in enumerate(rfecv.ranking_) if rank==1]
        else:
            ranked_feats_idx = sorted(range(len(rfecv.ranking_)), key=lambda k: rfecv.ranking_[k])[0:threshold]
        #end
        ranked_feats_ls = [feat_ls[idx] for idx in ranked_feats_idx]
        return ranked_feats_ls 
    #end

    def cat_feats_selector(self, feats_df, label_id, threshold, return_scores=False):
        '''
        '''
        feat_ls = [col for col in feats_df.columns if col!=label_id]
        score = make_scorer(MCC, greater_is_better=True)
        feat_score_ls = []
        feats_df = feats_df.sample(frac=self.sample_dc['categorical'])
        for fn, feat in enumerate(feat_ls):
            if (10*fn)%int(10*np.round((len(feat_ls)/10.0)))==0:
                logger.info('Progress: %s %%' % (str((100*fn)/int(10*np.round((len(feat_ls)/10.0))-1))))
            #end
            fitted_le_dc, fitted_ohe = CatEncoder().fit_onehot_to_cat(feats_df[[feat]])
            encoded_sparse_arr = CatEncoder().transform_onehot(feats_df[[feat]], fitted_le_dc, fitted_ohe)
            cv_scores = cross_val_score(self.Classifier, encoded_sparse_arr, feats_df[label_id].ravel(), cv=StratifiedKFold(3), scoring=score)
            feat_score_ls.append((feat, cv_scores.mean()))
        #end
        rank_ls = [el[0] for el in sorted(feat_score_ls, key=lambda tup: tup[1], reverse=True)]
        score_ls = [el[1] for el in sorted(feat_score_ls, key=lambda tup: tup[1], reverse=True)]
        if return_scores:
            return rank_ls, score_ls
        else:
            return rank_ls[0:threshold]
        #end
    #end

    def select_feats(self, feats_df, label_id='response', feat_type='numeric'):
        logger.info('Performing features selection...')
        logger.info('Number of features to be selected: %s numeric, %s categorical' % (str(self.n_thresh), str(self.c_thresh)))
        logger.info('Downsampling for speed: %s numeric, %s cateforical' % (str(self.sample_dc['numeric']), str(self.sample_dc['categorical'])))
        if feat_type=='numeric':
            logger.info('Fitting Recursive Feature Elimination Model for numeric feature selection...')
            ranked_feats_ls = self.num_feats_selector(feats_df, label_id, self.n_thresh)
        elif feat_type=='categorical':
            logger.info('Performing feature ranking for categorical features...')
            ranked_feats_ls = self.cat_feats_selector(feats_df, label_id, self.c_thresh)
        #end
        logger.info('Feature ranking complete!')
        return ranked_feats_ls
    #end

#end


class Assembler():
    '''
    '''
    def __init__(self):
        pass
    #end

    def assemble_train_test(self, n_df, c_df):
        '''
        '''
        logger.info('Assembling training dataset...')
        test_id_ar = n_df['Id'][n_df['is_test']==1].astype(int).ravel()
        response_ar = n_df['Response'][n_df['is_test']==0].astype(int).ravel()

        train_n_ar = n_df[[col for col in n_df.columns if col not in ['is_test', 'Id', 'Response']]][n_df['is_test']==0]
        test_n_ar = n_df[[col for col in n_df.columns if col not in ['is_test', 'Id', 'Response']]][n_df['is_test']==1]

        #cols = [col for col in c_df.columns if col not in ['is_test', 'Id', 'Response']]

        #fitted_le_dc, fitted_ohe = CatEncoder().fit_onehot_to_cat(c_df[cols][c_df['is_test']=='0']) THIS HAS BEEN COMMENTED LONG AGO
        

        #fitted_le_dc, fitted_ohe = CatEncoder().fit_onehot_to_cat(c_df[cols])
        #encoded_train_ar = CatEncoder().transform_onehot(c_df[cols][c_df['is_test'].astype(int)==0], fitted_le_dc, fitted_ohe)  
        #encoded_test_ar = CatEncoder().transform_onehot(c_df[cols][c_df['is_test'].astype(int)==1], fitted_le_dc, fitted_ohe)

        assembled_train_ar = sparse.csr_matrix(train_n_ar.values) # sparse.hstack([train_n_ar, encoded_train_ar])
        assembled_test_ar = sparse.csr_matrix(test_n_ar.values) #sparse.hstack([test_n_ar, encoded_test_ar])
       
        logger.info('Training dataset assembled!')
        return  assembled_train_ar, response_ar, assembled_test_ar, test_id_ar
    #end

#end

class Classifier():
    '''
    '''
    def __init__(self):
        pass
    #end

    def classify(self, train_sparse, labels, test_sparse, parameters, cv=False):
        logger.info('Training data shape: ' + str(train_sparse.shape))
        logger.info('Test data shape: ' + str(test_sparse.shape))
        logger.info('Learning...')

        dtrain = xgb.DMatrix(train_sparse, label=labels)
        dtest = xgb.DMatrix(test_sparse)
        parameters['params']['base_score'] = np.sum(labels) / (1.0 * len(labels))
        if cv:
            logger.info('Performing CV...')
            prior = np.sum(labels) / (1.*len(labels))
            booster = xgb.cv(parameters['params'], dtrain, **parameters['cross_val'])
            logger.info('CV score: %s mean, %s st_dev' % (str(booster.iloc[-1, 0]), str(booster.iloc[-1, 1])))
        #end
        logger.info('Training...')
        booster = xgb.train(parameters['params'], dtrain)
        logger.info('Saving model...')
        pickle.dump(booster, open("trained_booster.dat", "wb"))
        logger.info('Predicting...')
        try:
            predictions = booster.predict(dtest)
        except:
            dtrain = xgb.DMatrix(train_sparse.toarray(), label=labels)
            dtest = xgb.DMatrix(test_sparse.toarray())
            booster = xgb.train(parameters['params'], dtrain)
            predictions = booster.predict(dtest)
        #end   
        return predictions
    #end

#end

class ThresholdOptimizer():
    '''
    '''
    def __init__(self):
        pass
    #end

    def get_threshold(self, prob_ls_ls=None, true_ls_ls=None):
        return 0.5
    #end

#end

class MCCThresholdOptimizer():
    '''
    '''
    def __init__(self):
        pass
    #end
    def mcc(self, tp, tn, fp, fn):
        sup = tp * tn - fp * fn
        inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if inf==0:
            return 0
        else:
            return sup / np.sqrt(inf)
        #end
    #end

    def eval_mcc(self, y_true_ls, y_prob_ls, show=False):
        idx = np.argsort(y_prob_ls)
        sorted_y_true_ls = y_true_ls[idx]
        n_el = y_true_ls.shape[0]
        num_pos = 1.0 * np.sum(y_true_ls) # number of positive
        num_neg = n_el - num_pos # number of negative
        tp, tn, fp, fn = num_pos, 0.0, num_neg, 0.0
        best_id = prev_proba = best_proba = -1
        best_mcc = 0.0
        mccs = np.zeros(n_el)
        for iel in range(n_el):
            # all items with idx < iel are predicted negative while others are predicted positive
            # only evaluate mcc when probability changes
            proba = y_prob_ls[idx[iel]]
            if proba != prev_proba:
                prev_proba = proba
                new_mcc = self.mcc(tp, tn, fp, fn)
                if new_mcc >= best_mcc:
                    best_mcc = new_mcc
                    best_id = iel
                    best_proba = proba
                #end
            #end
            mccs[iel] = new_mcc
            if sorted_y_true_ls[iel] == 1:
                tp = tp - 1.0
                fn = fn + 1.0
            else:
                fp = fp - 1.0
                tn = tn + 1.0
            #end
        #end
        if show:
            y_pred = (y_prob_ls >= best_proba).astype(int)
            score = MCC(y_true_ls, y_pred)
            logger.info('Best MCC: %s ' % str(best_mcc))
            return best_proba, best_mcc, y_pred
        else:
            return best_mcc
        #end
    #end

    def mcc_evaluator(self, y_prob_ls, dtrain):
        '''
        '''
        y_true_ls = dtrain.get_label()
        best_mcc = self.eval_mcc(y_true_ls, y_prob_ls)
        return 'MCC', best_mcc
    #end

#end

class ModelOptimizer():
    '''
    '''
    def __init__(self):
        pass
    #end

#end


class OutputHandler():
    '''
    '''

    def __init__(self):
        pass
    #end

    def output_writer(self, id_col, predictions, data_path='./', gz=True):
        output_df = pd.DataFrame({ 'id' : id_col, 'response': predictions})
        if gz:
            output_df.to_csv(data_path + '/submission.gz', index = False, compression='gzip')
            saveformat = '.gz'
        else:
            output_df.to_csv(data_path + '/submission.csv', index = False)
            saveformat = '.csv'
        #end
        logger.info('Data successfully saved as %s' % saveformat)
        return
    #end

#end


