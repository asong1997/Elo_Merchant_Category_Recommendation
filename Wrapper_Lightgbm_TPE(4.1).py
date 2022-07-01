import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error

train = pd.read_csv('preprocess/train.csv')
test = pd.read_csv('preprocess/test.csv')


# Wrapper特征筛选
def feature_select_wrapper(train, test):
    """
    lgm特征重要性筛选函数
    :param train:训练数据集
    :param test:测试数据集
    :return:特征筛选后的训练集和测试集
    """

    # Part 1.划分特征名称，删除ID列和标签列
    print('feature_select_wrapper...')
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    # Step 2.配置lgb参数
    # 模型参数
    params_initial = {
        'num_leaves': 31,
        'learning_rate': 0.1,
        'boosting': 'gbdt',
        'min_child_samples': 20,
        'bagging_seed': 2020,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'feature_fraction': 0.7,
        'max_depth': -1,
        'metric': 'rmse',
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'regression'
    }
    # 控制参数
    # 提前验证迭代效果或停止
    ESR = 30
    # 迭代次数
    NBR = 10000
    # 打印间隔
    VBE = 50

    # Part 3.交叉验证过程
    # 实例化评估器
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    # 创建空容器
    fse = pd.Series(0, index=features)

    for train_part_index, eval_index in kf.split(train[features], train[label]):
        # 封装训练数据集
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index])
        # 封装验证数据集
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index])
        # 在训练集上进行训练，并同时进行验证
        bst = lgb.train(params_initial, train_part, num_boost_round=NBR,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        # 输出特征重要性计算结果，并进行累加
        fse += pd.Series(bst.feature_importance(), features)

    # Part 4.选择最重要的300个特征
    feature_select = ['card_id'] + fse.sort_values(ascending=False).index.tolist()[:300]
    print('done')
    return train[feature_select + ['target']], test[feature_select]


# 参数回调函数
def params_append(params):
    """
    动态回调参数函数，params视作字典
    :param params:lgb参数字典
    :return params:修正后的lgb参数字典
    """
    params['feature_pre_filter'] = False
    params['objective'] = 'regression'
    params['metric'] = 'rmse'
    params['bagging_seed'] = 2020
    return params


# 模型训练与参数优化函数
def param_hyperopt(train):
    """
    模型参数搜索与优化函数
    :param train:训练数据集
    :return params_best:lgb最优参数
    """
    # Part 1.划分特征名称，删除ID列和标签列
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    # Part 2.封装训练数据
    train_data = lgb.Dataset(train[features], train[label])

    # Part 3.内部函数，输入模型超参数损失值输出函数
    def hyperopt_objective(params):
        """
        输入超参数，输出对应损失值
        :param params:
        :return:最小rmse
        """
        # 创建参数集
        params = params_append(params)
        print(params)

        # 借助lgb的cv过程，输出某一组超参数下损失值的最小值
        res = lgb.cv(params, train_data, 1000,
                     nfold=2,
                     stratified=False,
                     shuffle=True,
                     metrics='rmse',
                     early_stopping_rounds=20,
                     verbose_eval=False,
                     show_stdv=False,
                     seed=2020)
        return min(res['rmse-mean'])  # res是个字典

    # Part 4.lgb超参数空间
    params_space = {
        'learning_rate': hp.uniform('learning_rate', 1e-2, 5e-1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'num_leaves': hp.choice('num_leaves', list(range(10, 300, 10))),
        'reg_alpha': hp.randint('reg_alpha', 0, 10),
        'reg_lambda': hp.uniform('reg_lambda', 0, 10),
        'bagging_freq': hp.randint('bagging_freq', 1, 10),
        'min_child_samples': hp.choice('min_child_samples', list(range(1, 30, 5)))
    }

    # Part 5.TPE超参数搜索
    params_best = fmin(
        hyperopt_objective,
        space=params_space,
        algo=tpe.suggest,
        max_evals=30)

    # 返回最佳参数
    return params_best


def train_predict(train, test, params):
    """

    :param train:
    :param test:
    :param params:
    :return:
    """
    # Part 1.选择特征
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    # Part 2.再次申明固定参数与控制迭代参数
    params = params_append(params)
    ESR = 30
    NBR = 10000
    VBE = 50

    # Part 3.创建结果存储容器
    # 测试集预测结果存储器，后保存至本地文件
    prediction_test = 0
    # 验证集的模型表现，作为展示用
    cv_score = []
    # 验证集的预测结果存储器，后保存至本地文件
    prediction_train = pd.Series()

    # Part 3.交叉验证
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    for train_part_index, eval_index in kf.split(train[features], train[label]):
        # 训练数据封装
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index])
        # 测试数据封装
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index])
        # 依据验证集训练模型
        bst = lgb.train(params, train_part, num_boost_round=NBR,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        # 测试集预测结果并纳入prediction_test容器
        prediction_test += bst.predict(test[features])
        # 验证集预测结果并纳入prediction_train容器
        prediction_train = prediction_train.append(pd.Series(bst.predict(train[features].loc[eval_index]),
                                                             index=eval_index))
        # 验证集预测结果
        eval_pre = bst.predict(train[features].loc[eval_index])
        # 计算验证集上得分
        score = np.sqrt(mean_squared_error(train[label].loc[eval_index].values, eval_pre))
        # 纳入cv_score容器
        cv_score.append(score)

    # Part 4.打印/输出结果
    # 打印验证集得分与平均得分
    print(cv_score, sum(cv_score) / 5)
    # 将验证集上预测结果写入本地文件
    pd.Series(prediction_train.sort_index().values).to_csv("preprocess/train_lightgbm.csv", index=False)
    # 将测试集上预测结果写入本地文件
    pd.Series(prediction_test / 5).to_csv("preprocess/test_lightgbm.csv", index=False)
    # 测试集平均得分作为模型最终预测结果
    test['target'] = prediction_test / 5
    # 将测试集预测结果写成竞赛要求格式并保存至本地
    test[['card_id', 'target']].to_csv("result/submission_lightgbm.csv", index=False)
    return


print("特征选择前train.shape：", train.shape)
train_LGBM, test_LGBM = feature_select_wrapper(train, test)
print("经过wrapper特征选择后得到的特征train_LGBM.shape：", train_LGBM.shape)
best_clf = param_hyperopt(train_LGBM)
print("经过贝叶斯优化后得到的最优超参数为：", best_clf)
train_predict(train_LGBM, test_LGBM, best_clf)