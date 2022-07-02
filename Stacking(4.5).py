import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

oof_rf = pd.read_csv('./preprocess/train_randomforest.csv')
predictions_rf = pd.read_csv('./preprocess/test_randomforest.csv')

oof_lgb = pd.read_csv('./preprocess/train_lightgbm.csv')
predictions_lgb = pd.read_csv('./preprocess/test_lightgbm.csv')

oof_xgb = pd.read_csv('./preprocess/train_xgboost.csv')
predictions_xgb = pd.read_csv('./preprocess/test_xgboost.csv')


def stack_model(oof_1, oof_2, oof_3, predictions_1, predictions_2, predictions_3, y):
    # Part 1.数据准备
    # 按行拼接列，拼接验证集所有预测结果
    # train_stack就是final model的训练数据
    train_stack = np.hstack([oof_1, oof_2, oof_3])
    # 按行拼接列，拼接测试集上所有预测结果
    # test_stack就是final model的测试数据
    test_stack = np.hstack([predictions_1, predictions_2, predictions_3])
    # 创建一个和验证集行数相同的全零数组
    # oof = np.zeros(train_stack.shape[0])
    # 创建一个和测试集行数相同的全零数组
    predictions = np.zeros(test_stack.shape[0])

    # Part 2.多轮交叉验证
    from sklearn.model_selection import RepeatedKFold
    folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2020)

    # fold_为折数，trn_idx为每一折训练集index，val_idx为每一折验证集index
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, y)):
        # 打印折数信息
        print("fold n°{}".format(fold_ + 1))
        # 训练集中划分为训练数据的特征和标签
        trn_data, trn_y = train_stack[trn_idx], y[trn_idx]
        # 训练集中划分为验证数据的特征和标签
        val_data, val_y = train_stack[val_idx], y[val_idx]
        # 开始训练时提示
        print("-" * 10 + "Stacking " + str(fold_ + 1) + "-" * 10)
        # 采用贝叶斯回归作为结果融合的模型（final model）
        clf = BayesianRidge()
        # 在训练数据上进行训练
        clf.fit(trn_data, trn_y)
        # 在验证数据上进行预测，并将结果记录在oof对应位置
        # oof[val_idx] = clf.predict(val_data)
        # 对测试集数据进行预测，每一轮预测结果占比额外的1/10
        predictions += clf.predict(test_stack) / (5 * 2)

    # 返回测试集的预测结果
    return predictions


train = pd.read_csv('preprocess/train.csv')
target = train['target'].values
predictions_stack = stack_model(oof_rf, oof_lgb, oof_xgb, predictions_rf, predictions_lgb, predictions_xgb, target)

print(predictions_stack)
sub_df = pd.read_csv('data/sample_submission.csv')
sub_df["target"] = predictions_stack
sub_df.to_csv('predictions_stack1.csv', index=False)
