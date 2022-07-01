from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.feature_selection import f_regression
from numpy.random import RandomState
from bayes_opt import BayesianOptimization

# pip install bayesian-optimization

train = pd.read_csv('preprocess/train.csv')
test = pd.read_csv('preprocess/test.csv')

features = train.columns.tolist()
features.remove('card_id')
features.remove('target')

train_x = sparse.load_npz("preprocess/train_nlp.npz")
test_x = sparse.load_npz("preprocess/test_nlp.npz")

train_x = sparse.hstack((train_x, train[features])).tocsr()
test_x = sparse.hstack((test_x, test[features])).tocsr()


# 参数回调函数
def params_append(params):
    """

    :param params:
    :return:
    """
    params['objective'] = 'reg:squarederror'
    params['eval_metric'] = 'rmse'
    params["min_child_weight"] = int(params["min_child_weight"])
    params['max_depth'] = int(params['max_depth'])
    return params


# 模型优化函数
def param_beyesian(train):
    """

    :param train:
    :return:
    """
    # Part 1.数据准备
    train_y = pd.read_csv("data/train.csv")['target']
    # 数据封装
    sample_index = train_y.sample(frac=0.1, random_state=2020).index.tolist()
    train_data = xgb.DMatrix(train.tocsr()[sample_index, :
                             ], train_y.loc[sample_index].values, silent=True)

    # 借助cv过程构建目标函数
    def xgb_cv(colsample_bytree, subsample, min_child_weight, max_depth,
               reg_alpha, eta,
               reg_lambda):
        """

        :param colsample_bytree:
        :param subsample:
        :param min_child_weight:
        :param max_depth:
        :param reg_alpha:
        :param eta:
        :param reg_lambda:
        :return:
        """
        params = {'objective': 'reg:squarederror',
                  'early_stopping_round': 50,
                  'eval_metric': 'rmse'}
        params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['subsample'] = max(min(subsample, 1), 0)
        params["min_child_weight"] = int(min_child_weight)
        params['max_depth'] = int(max_depth)
        params['eta'] = float(eta)
        params['reg_alpha'] = max(reg_alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        print(params)
        cv_result = xgb.cv(params, train_data,
                           num_boost_round=1000,
                           nfold=2, seed=2,
                           stratified=False,
                           shuffle=True,
                           early_stopping_rounds=30,
                           verbose_eval=False)
        return -min(cv_result['test-rmse-mean'])

    # 调用贝叶斯优化器进行模型优化
    xgb_bo = BayesianOptimization(
        xgb_cv,
        {'colsample_bytree': (0.5, 1),
         'subsample': (0.5, 1),
         'min_child_weight': (1, 30),
         'max_depth': (5, 12),
         'reg_alpha': (0, 5),
         'eta': (0.02, 0.2),
         'reg_lambda': (0, 5)}
    )
    xgb_bo.maximize(init_points=21, n_iter=5)  # init_points表示初始点，n_iter代表迭代次数（即采样数）
    print(xgb_bo.max['target'], xgb_bo.max['params'])
    return xgb_bo.max['params']


# 交叉验证预测函数
def train_predict(train, test, params):
    """

    :param train:
    :param test:
    :param params:
    :return:
    """
    train_y = pd.read_csv("data/train.csv")['target']
    test_data = xgb.DMatrix(test)

    params = params_append(params)
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series()
    ESR = 30
    NBR = 10000
    VBE = 50
    for train_part_index, eval_index in kf.split(train, train_y):
        # 模型训练
        train_part = xgb.DMatrix(train.tocsr()[train_part_index, :],
                                 train_y.loc[train_part_index])
        eval = xgb.DMatrix(train.tocsr()[eval_index, :],
                           train_y.loc[eval_index])
        bst = xgb.train(params, train_part, NBR, [(train_part, 'train'),
                                                  (eval, 'eval')], verbose_eval=VBE,
                        maximize=False, early_stopping_rounds=ESR, )
        prediction_test += bst.predict(test_data)
        eval_pre = bst.predict(eval)
        prediction_train = prediction_train.append(pd.Series(eval_pre, index=eval_index))
        score = np.sqrt(mean_squared_error(train_y.loc[eval_index].values, eval_pre))
        cv_score.append(score)
    print(cv_score, sum(cv_score) / 5)
    pd.Series(prediction_train.sort_index().values).to_csv("preprocess/train_xgboost.csv", index=False)
    pd.Series(prediction_test / 5).to_csv("preprocess/test_xgboost.csv", index=False)
    test = pd.read_csv('data/test.csv')
    test['target'] = prediction_test / 5
    test[['card_id', 'target']].to_csv("result/submission_xgboost.csv", index=False)
    return


best_clf = param_beyesian(train_x)
train_predict(train_x, test_x, best_clf)
