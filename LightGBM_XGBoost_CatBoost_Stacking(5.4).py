import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss


def stack_model(oof_1, oof_2, oof_3, predictions_1, predictions_2, predictions_3, y, eval_type='regression'):
    # Part 1.数据准备
    # 按行拼接列，拼接验证集所有预测结果
    # train_stack就是final model的训练数据
    train_stack = np.hstack([oof_1, oof_2, oof_3])
    # 按行拼接列，拼接测试集上所有预测结果
    # test_stack就是final model的测试数据
    test_stack = np.hstack([predictions_1, predictions_2, predictions_3])
    # 创建一个和验证集行数相同的全零数组
    oof = np.zeros(train_stack.shape[0])
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

    if eval_type == 'regression':
        print('mean: ', np.sqrt(mean_squared_error(y, oof)))
    if eval_type == 'binary':
        print('mean: ', log_loss(y, oof))

    # 返回测试集的预测结果
    return oof, predictions


predictions_lgb = pd.read_csv("./result/predictions_lgb.csv", header=None)
predictions_xgb = pd.read_csv("./result/predictions_xgb.csv", header=None)
predictions_cat = pd.read_csv("./result/predictions_cat.csv", header=None)

oof_lgb = pd.read_csv('./result/oof_lgb.csv', header=None)
oof_blgb = pd.read_csv('./result/oof_blgb.csv', header=None)
oof_nlgb = pd.read_csv('./result/oof_nlgb.csv', header=None)
oof_xgb = pd.read_csv('./result/oof_xgb.csv', header=None)
oof_cat = pd.read_csv('./result/oof_cat.csv', header=None)
oof_nxgb = pd.read_csv('./result/oof_nxgb.csv', header=None)
oof_ncat = pd.read_csv('./result/oof_ncat.csv', header=None)
oof_bxgb = pd.read_csv('./result/oof_bxgb.csv', header=None)
oof_bcat = pd.read_csv('./result/oof_bcat.csv', header=None)

predictions_nxgb = pd.read_csv('./result/predictions_nxgb.csv', header=None)
predictions_nlgb = pd.read_csv('./result/predictions_nlgb.csv', header=None)
predictions_ncat = pd.read_csv('./result/predictions_ncat.csv', header=None)
predictions_blgb = pd.read_csv('./result/predictions_blgb.csv', header=None)
predictions_bxgb = pd.read_csv('./result/predictions_bxgb.csv', header=None)
predictions_bcat = pd.read_csv('./result/predictions_bcat.csv', header=None)

train = pd.read_csv('./data/all_train_features.csv')
test = pd.read_csv('./data/all_test_features.csv')

inf_cols = ['new_cardf_card_id_cnt_divide_installments_nunique', 'hist_last2_card_id_cnt_divide_installments_nunique']
train[inf_cols] = train[inf_cols].replace(np.inf, train[inf_cols].replace(np.inf, -99).max().max())
# ntrain[inf_cols] = ntrain[inf_cols].replace(np.inf, ntrain[inf_cols].replace(np.inf, -99).max().max())
test[inf_cols] = test[inf_cols].replace(np.inf, test[inf_cols].replace(np.inf, -99).max().max())

# ## load sparse
# train_tags = sparse.load_npz('train_tags.npz')
# test_tags  = sparse.load_npz('test_tags.npz')
# 获取非异常值的index
normal_index = train[train['outliers'] == 0].index.tolist()
# without outliers
ntrain = train[train['outliers'] == 0]

target = train['target'].values
ntarget = ntrain['target'].values
target_binary = train['outliers'].values
###
y_train = target
y_ntrain = ntarget
y_train_binary = target_binary

print('=' * 30)
oof_stack, predictions_stack = stack_model(oof_lgb, oof_xgb, oof_cat, predictions_lgb, predictions_xgb, predictions_cat,
                                           target)
print('=' * 30)
oof_nstack, predictions_nstack = stack_model(oof_nlgb, oof_nxgb, oof_ncat, predictions_nlgb, predictions_nxgb,
                                             predictions_ncat, ntarget)
print('=' * 30)
oof_bstack, predictions_bstack = stack_model(oof_blgb, oof_bxgb, oof_bcat, predictions_blgb, predictions_bxgb,
                                             predictions_bcat, target_binary, eval_type='binary')

sub_df = pd.read_csv('data/sample_submission.csv')
sub_df["target"] = predictions_stack
sub_df.to_csv('predictions_stack.csv', index=False)


"""
LightGBM+XGBoost+CatBoost加上stacking模型融合 + 加上二阶段建模
"""
sub_df = pd.read_csv('data/sample_submission.csv')
sub_df["target"] = predictions_bstack * -33.219281 + (1 - predictions_bstack) * predictions_nstack
sub_df.to_csv('predictions_trick.csv', index=False)

sub_df = pd.read_csv('data/sample_submission.csv')
sub_df["target"] = (predictions_bstack * -33.219281 + (1 - predictions_bstack) * predictions_nstack) * 0.5 + predictions_stack * 0.5
sub_df.to_csv('predictions_trick&stacking.csv', index=False)
