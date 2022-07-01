import os
import numpy as np
import pandas as pd

train = pd.read_csv('preprocess/train.csv')
test = pd.read_csv('preprocess/test.csv')


# 提取特征名称
features = train.columns.tolist()
features.remove("card_id")
features.remove("target")
featureSelect = features[:]

# 计算相关系数
corr = []
for fea in featureSelect:
    corr.append(abs(train[[fea, 'target']].fillna(0).corr().values[0][1]))

# 取top300的特征进行建模，具体数量可选
se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
feature_select = ['card_id'] + se[:300].index.tolist()

# 输出结果
train_RF = train[feature_select + ['target']]
test_RF = test[feature_select]

def feature_select_pearson(train, test):
    """
    利用pearson系数进行相关性特征选择
    :param train:训练集
    :param test:测试集
    :return:经过特征选择后的训练集与测试集
    """
    print('feature_select...')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")
    featureSelect = features[:]

    # 去掉缺失值比例超过0.99的
    for fea in features:
        if train[fea].isnull().sum() / train.shape[0] >= 0.99:
            featureSelect.remove(fea)

    # 进行pearson相关性计算
    corr = []
    for fea in featureSelect:
        corr.append(abs(train[[fea, 'target']].fillna(0).corr().values[0][1]))

    # 取top300的特征进行建模，具体数量可选
    se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
    feature_select = ['card_id'] + se[:300].index.tolist()
    print('done')
    return train[feature_select + ['target']], test[feature_select]


from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def param_grid_search(train):
    """
    网格搜索参数调优
    :param train:训练集
    :return:网格搜索训练结果
    """
    # Step 1.创建网格搜索空间
    print('param_grid_search')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")
    parameter_space = {
        "n_estimators": [81],
        "min_samples_leaf": [31],
        "min_samples_split": [2],
        "max_depth": [10],
        "max_features": [80]
    }

    # Step 2.执行网格搜索过程
    print("Tuning hyper-parameters for mse")
    # 实例化随机森林模型
    clf = RandomForestRegressor(
        criterion="mse",
        n_jobs=15,
        random_state=22)
    # 带入网格搜索
    grid = GridSearchCV(clf, parameter_space, cv=2, scoring="neg_mean_squared_error")
    grid.fit(train[features].values, train['target'].values)

    # Step 3.输出网格搜索结果
    print("best_params_:")
    print(grid.best_params_)
    means = grid.cv_results_["mean_test_score"]
    stds = grid.cv_results_["std_test_score"]
    # 此处额外考虑观察交叉验证过程中不同超参数的
    for mean, std, params in zip(means, stds, grid.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    return grid

grid = param_grid_search(train_RF)

print(grid)
print(grid.best_estimator_)
print(np.sqrt(-grid.best_score_))

############
feature_select.remove('card_id')
test['target'] = grid.best_estimator_.predict(test[feature_select])
test[['card_id', 'target']].to_csv("result/submission_randomforest.csv", index=False)


"""
随机森林交叉验证评估与中间结果保存
"""
from sklearn.model_selection import KFold
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error

def train_predict(train, test, best_clf):
    """
    进行训练和预测输出结果
    :param train:训练集
    :param test:测试集
    :param best_clf:最优的分类器模型
    :return:
    """

    # Step 1.选择特征
    print('train_predict...')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")

    # Step 2.创建存储器
    # 测试集评分存储器
    prediction_test = 0
    # 交叉验证评分存储器
    cv_score = []
    # 验证集的预测结果
    prediction_train = pd.Series()

    # Step 3.交叉验证
    # 实例化交叉验证评估器
    kf = KFold(n_splits=5, random_state=22, shuffle=True)
    # 执行交叉验证过程
    for train_part_index, eval_index in kf.split(train[features], train['target']):
        # 在训练集上训练模型
        best_clf.fit(train[features].loc[train_part_index].values, train['target'].loc[train_part_index].values)
        # 模型训练完成后，输出测试集上预测结果并累加至prediction_test中
        prediction_test += best_clf.predict(test[features].values)
        # 输出验证集上预测结果，eval_pre为临时变量
        eval_pre = best_clf.predict(train[features].loc[eval_index].values)
        # 输出验证集上预测结果评分，评估指标为MSE
        score = np.sqrt(mean_squared_error(train['target'].loc[eval_index].values, eval_pre))
        # 将本轮验证集上的MSE计算结果添加至cv_score列表中
        cv_score.append(score)
        print(score)
        # 将验证集上的预测结果放到prediction_train中
        prediction_train = prediction_train.append(pd.Series(best_clf.predict(train[features].loc[eval_index]),
                                                             index=eval_index))

    # 打印每轮验证集得分、5轮验证集的平均得分
    print(cv_score, sum(cv_score) / 5)
    # 验证集上预测结果写入本地文件
    pd.Series(prediction_train.sort_index().values).to_csv("preprocess/train_randomforest.csv", index=False)
    # 测试集上平均得分写入本地文件
    pd.Series(prediction_test / 5).to_csv("preprocess/test_randomforest.csv", index=False)
    # 在测试集上加入target，也就是预测标签
    test['target'] = prediction_test / 5
    # 将测试集id和标签组成新的DataFrame并写入本地文件，该文件就是后续提交结果
    test[['card_id', 'target']].to_csv("result/submission_randomforest.csv", index=False)
    return

train_predict(train_RF, test_RF, grid.best_estimator_)