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

# 注意，该数据集是最初始的数据集
train = pd.read_csv('data/train.csv')
test =  pd.read_csv('data/test.csv')
merchant = pd.read_csv('data/merchants.csv')
new_transaction = pd.read_csv('data/new_merchant_transactions.csv')
history_transaction = pd.read_csv('data/historical_transactions.csv')
transaction = pd.concat([new_transaction, history_transaction], axis=0, ignore_index=True)
del new_transaction
del history_transaction
gc.collect()

"""
NLP特征衍生
首先我们注意到，在数据集中存在大量的ID相关的列（除了card_id外），包括'merchant_id'、'merchant_category_id'、'state_id'、'subsector_id'、'city_id'等，
考虑到这些ID在出现频率方面都和用户实际的交易行为息息相关，例如对于单独用户A来说，在其交易记录中频繁出现某商户id（假设为B），则说明该用户A对商户B情有独钟，
而如果在不同的用户交易数据中，都频繁的出现了商户B，则说明这家商户受到广泛欢迎，而进一步的说明A的喜好可能和大多数用户一致，而反之则说明A用户的喜好较为独特。
为了能够挖掘出类似信息，我们可以考虑采用NLP中CountVector和TF-IDF两种方法来进行进一步特征衍生，其中CountVector可以挖掘类似某用户钟爱某商铺的信息，
而TF-IDF则可进一步挖掘出类似某用户的喜好是否普遍或一致等信息。
"""

nlp_features = ['merchant_id', 'merchant_category_id', 'state_id', 'subsector_id', 'city_id']

for co in nlp_features:
    print(co)
    transaction[co] = transaction[co].astype(str)
    temp = transaction[transaction['month_lag']>=0].groupby("card_id")[co].apply(list).apply(lambda x:' '.join(x)).reset_index()
    temp.columns = ['card_id', co+'_new']
    train = pd.merge(train, temp, how='left', on='card_id')
    test = pd.merge(test, temp, how='left', on='card_id')

    temp = transaction[transaction['month_lag']<0].groupby("card_id")[co].apply(list).apply(lambda x:' '.join(x)).reset_index()
    temp.columns = ['card_id', co+'_hist']
    train = pd.merge(train, temp, how='left', on='card_id')
    test = pd.merge(test, temp, how='left', on='card_id')

    temp = transaction.groupby("card_id")[co].apply(list).apply(lambda x:' '.join(x)).reset_index()
    temp.columns = ['card_id', co+'_all']
    train = pd.merge(train, temp, how='left', on='card_id').fillna("-1")
    test = pd.merge(test, temp, how='left', on='card_id').fillna("-1")

# 创建空DataFrame用于保存NLP特征
train_x = pd.DataFrame()
test_x = pd.DataFrame()

# 实例化CountVectorizer评估器与TfidfVectorizer评估器
cntv = CountVectorizer()
tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)

# 创建空列表用户保存修正后的列名称
vector_feature = []
for co in ['merchant_id', 'merchant_category_id', 'state_id', 'subsector_id', 'city_id']:
    vector_feature.extend([co + '_new', co + '_hist', co + '_all'])

# 提取每一列进行新特征衍生
for feature in vector_feature:
    print(feature)
    cntv.fit(train[feature].append(test[feature]))
    train_x = sparse.hstack((train_x, cntv.transform(train[feature]))).tocsr()
    test_x = sparse.hstack((test_x, cntv.transform(test[feature]))).tocsr()

    tfv.fit(train[feature].append(test[feature]))
    train_x = sparse.hstack((train_x, tfv.transform(train[feature]))).tocsr()
    test_x = sparse.hstack((test_x, tfv.transform(test[feature]))).tocsr()

# 保存NLP特征衍生结果
sparse.save_npz("preprocess/train_nlp.npz", train_x)
sparse.save_npz("preprocess/test_nlp.npz", test_x)