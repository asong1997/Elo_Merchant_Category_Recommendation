import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, tqdm_notebook
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import BayesianRidge
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import sparse
import warnings
import time
import sys
import os
import gc
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss


train = pd.read_csv('./data/all_train_features.csv')
test = pd.read_csv('./data/all_test_features.csv')

inf_cols = ['new_cardf_card_id_cnt_divide_installments_nunique', 'hist_last2_card_id_cnt_divide_installments_nunique']
train[inf_cols] = train[inf_cols].replace(np.inf, train[inf_cols].replace(np.inf, -99).max().max())
# ntrain[inf_cols] = ntrain[inf_cols].replace(np.inf, ntrain[inf_cols].replace(np.inf, -99).max().max())
test[inf_cols] = test[inf_cols].replace(np.inf, test[inf_cols].replace(np.inf, -99).max().max())

# ## load sparse
# train_tags = sparse.load_npz('train_tags.npz')
# test_tags  = sparse.load_npz('test_tags.npz')

## 获取非异常值的index
normal_index = train[train['outliers'] == 0].index.tolist()
## without outliers
ntrain = train[train['outliers'] == 0]

target = train['target'].values
ntarget = ntrain['target'].values
target_binary = train['outliers'].values
###
y_train = target
y_ntrain = ntarget
y_train_binary = target_binary

print('train:', train.shape)
print('ntrain:', ntrain.shape)


def train_model(X, X_test, y, params, folds, model_type='lgb', eval_type='regression'):
    oof = np.zeros(X.shape[0])
    predictions = np.zeros(X_test.shape[0])
    scores = []
    for fold_n, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())

        if model_type == 'lgb':
            trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
            val_data = lgb.Dataset(X[val_idx], y[val_idx])
            clf = lgb.train(params, trn_data, num_boost_round=20000,
                            valid_sets=[trn_data, val_data],
                            verbose_eval=100, early_stopping_rounds=300)
            oof[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)
            predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

        if model_type == 'xgb':
            trn_data = xgb.DMatrix(X[trn_idx], y[trn_idx])
            val_data = xgb.DMatrix(X[val_idx], y[val_idx])
            watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
            clf = xgb.train(dtrain=trn_data, num_boost_round=20000,
                            evals=watchlist, early_stopping_rounds=200,
                            verbose_eval=100, params=params)
            oof[val_idx] = clf.predict(xgb.DMatrix(X[val_idx]), ntree_limit=clf.best_ntree_limit)
            predictions += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

        if (model_type == 'cat') and (eval_type == 'regression'):
            clf = CatBoostRegressor(iterations=20000, eval_metric='RMSE', **params)
            clf.fit(X[trn_idx], y[trn_idx],
                    eval_set=(X[val_idx], y[val_idx]),
                    cat_features=[], use_best_model=True, verbose=100)
            oof[val_idx] = clf.predict(X[val_idx])
            predictions += clf.predict(X_test) / folds.n_splits

        if (model_type == 'cat') and (eval_type == 'binary'):
            clf = CatBoostClassifier(iterations=20000, eval_metric='Logloss', **params)
            clf.fit(X[trn_idx], y[trn_idx],
                    eval_set=(X[val_idx], y[val_idx]),
                    cat_features=[], use_best_model=True, verbose=100)
            oof[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
            predictions += clf.predict_proba(X_test)[:, 1] / folds.n_splits
        print(predictions)
        if eval_type == 'regression':
            scores.append(mean_squared_error(oof[val_idx], y[val_idx]) ** 0.5)
        if eval_type == 'binary':
            scores.append(log_loss(y[val_idx], oof[val_idx]))

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    return oof, predictions, scores


#### lgb
lgb_params = {'num_leaves': 63,
              'min_data_in_leaf': 32,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': 0.01,
              "min_child_samples": 20,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              "bagging_seed": 11,
              "metric": 'rmse',
              "lambda_l1": 0.1,
              "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=4096)

fea_cols = test.columns
X_ntrain = ntrain[fea_cols].values
X_train = train[fea_cols].values
X_test = test[fea_cols].values
print('=' * 10, '回归模型', '=' * 10)
oof_lgb, predictions_lgb, scores_lgb = train_model(X_train, X_test, y_train, params=lgb_params, folds=folds,
                                                   model_type='lgb', eval_type='regression')

print('=' * 10, 'without outliers 回归模型', '=' * 10)
oof_nlgb, predictions_nlgb, scores_nlgb = train_model(X_ntrain, X_test, y_ntrain, params=lgb_params, folds=folds,
                                                      model_type='lgb', eval_type='regression')

print('=' * 10, '分类模型', '=' * 10)
lgb_params['objective'] = 'binary'
lgb_params['metric'] = 'binary_logloss'
oof_blgb, predictions_blgb, scores_blgb = train_model(X_train, X_test, y_train_binary, params=lgb_params, folds=folds,
                                                      model_type='lgb', eval_type='binary')

sub_df = pd.read_csv('data/sample_submission.csv')
sub_df["target"] = predictions_lgb
sub_df.to_csv('predictions_lgb.csv', index=False)

oof_lgb = pd.DataFrame(oof_lgb)
oof_nlgb = pd.DataFrame(oof_nlgb)
oof_blgb = pd.DataFrame(oof_blgb)

predictions_lgb = pd.DataFrame(predictions_lgb)
predictions_nlgb = pd.DataFrame(predictions_nlgb)
predictions_blgb = pd.DataFrame(predictions_blgb)

oof_lgb.to_csv('./result/oof_lgb.csv', header=False, index=False)
oof_blgb.to_csv('./result/oof_blgb.csv', header=False, index=False)
oof_nlgb.to_csv('./result/oof_nlgb.csv', header=False, index=False)

predictions_lgb.to_csv('./result/predictions_lgb.csv', header=False, index=False)
predictions_nlgb.to_csv('./result/predictions_nlgb.csv', header=False, index=False)
predictions_blgb.to_csv('./result/predictions_blgb.csv', header=False, index=False)

#### xgb
xgb_params = {'eta': 0.05, 'max_leaves': 47, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'min_child_weight': 40, 'max_bin': 128, 'reg_alpha': 2.0, 'reg_lambda': 2.0,
              'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
print('=' * 10, '回归模型', '=' * 10)
oof_xgb, predictions_xgb, scores_xgb = train_model(X_train, X_test, y_train, params=xgb_params, folds=folds,
                                                   model_type='xgb', eval_type='regression')
print('=' * 10, 'without outliers 回归模型', '=' * 10)
oof_nxgb, predictions_nxgb, scores_nxgb = train_model(X_ntrain, X_test, y_ntrain, params=xgb_params, folds=folds,
                                                      model_type='xgb', eval_type='regression')
print('=' * 10, '分类模型', '=' * 10)
xgb_params['objective'] = 'binary:logistic'
xgb_params['metric'] = 'binary_logloss'
oof_bxgb, predictions_bxgb, scores_bxgb = train_model(X_train, X_test, y_train_binary, params=xgb_params, folds=folds,
                                                      model_type='xgb', eval_type='binary')

sub_df = pd.read_csv('data/sample_submission.csv')
sub_df["target"] = predictions_xgb
sub_df.to_csv('predictions_xgb.csv', index=False)

oof_xgb = pd.DataFrame(oof_xgb)
oof_nxgb = pd.DataFrame(oof_nxgb)
oof_bxgb = pd.DataFrame(oof_bxgb)

predictions_xgb = pd.DataFrame(predictions_xgb)
predictions_nxgb = pd.DataFrame(predictions_nxgb)
predictions_bxgb = pd.DataFrame(predictions_bxgb)

oof_xgb.to_csv('./result/oof_xgb.csv', header=False, index=False)
oof_bxgb.to_csv('./result/oof_bxgb.csv', header=False, index=False)
oof_nxgb.to_csv('./result/oof_nxgb.csv', header=False, index=False)

predictions_xgb.to_csv('./result/predictions_xgb.csv', header=False, index=False)
predictions_nxgb.to_csv('./result/predictions_nxgb.csv', header=False, index=False)
predictions_bxgb.to_csv('./result/predictions_bxgb.csv', header=False, index=False)

#### cat
cat_params = {'learning_rate': 0.05, 'depth': 9, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
              'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}
folds = KFold(n_splits=5, shuffle=True, random_state=18)
print('=' * 10, '回归模型', '=' * 10)
oof_cat, predictions_cat, scores_cat = train_model(X_train, X_test, y_train, params=cat_params, folds=folds,
                                                   model_type='cat', eval_type='regression')
print('=' * 10, 'without outliers 回归模型', '=' * 10)
oof_ncat, predictions_ncat, scores_ncat = train_model(X_ntrain, X_test, y_ntrain, params=cat_params, folds=folds,
                                                      model_type='cat', eval_type='regression')
print('=' * 10, '分类模型', '=' * 10)
oof_bcat, predictions_bcat, scores_bcat = train_model(X_train, X_test, y_train_binary, params=cat_params, folds=folds,
                                                      model_type='cat', eval_type='binary')

sub_df = pd.read_csv('data/sample_submission.csv')
sub_df["target"] = predictions_cat
sub_df.to_csv('predictions_cat.csv', index=False)

oof_cat = pd.DataFrame(oof_cat)
oof_ncat = pd.DataFrame(oof_ncat)
oof_bcat = pd.DataFrame(oof_bcat)

predictions_cat = pd.DataFrame(predictions_cat)
predictions_ncat = pd.DataFrame(predictions_ncat)
predictions_bcat = pd.DataFrame(predictions_bcat)

oof_cat.to_csv('./result/oof_cat.csv', header=False, index=False)
oof_bcat.to_csv('./result/oof_bcat.csv', header=False, index=False)
oof_ncat.to_csv('./result/oof_ncat.csv', header=False, index=False)

predictions_cat.to_csv('./result/predictions_cat.csv', header=False, index=False)
predictions_ncat.to_csv('./result/predictions_ncat.csv', header=False, index=False)
predictions_bcat.to_csv('./result/predictions_bcat.csv', header=False, index=False)
