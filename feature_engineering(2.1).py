"""
1.2 基于transaction数据集创建通用组合特征

"""
import gc
import time
import numpy as np
import pandas as pd
from datetime import datetime

train = pd.read_csv('preprocess/train_pre.csv')
test = pd.read_csv('preprocess/test_pre.csv')
transaction = pd.read_csv('preprocess/transaction_d_pre.csv')

# 标注离散字段or连续型字段
numeric_cols = ['purchase_amount', 'installments']

category_cols = ['authorized_flag', 'city_id', 'category_1',
                 'category_3', 'merchant_category_id', 'month_lag', 'most_recent_sales_range',
                 'most_recent_purchases_range', 'category_4',
                 'purchase_month', 'purchase_hour_section', 'purchase_day']

id_cols = ['card_id', 'merchant_id']


# 特征创建
# 创建字典用于保存数据
features = {}
card_all = train['card_id'].append(test['card_id']).values.tolist()
for card in card_all:
    features[card] = {}

# 标记不同类型字段的索引
columns = transaction.columns.tolist()
idx = columns.index('card_id')
category_cols_index = [columns.index(col) for col in category_cols]
numeric_cols_index = [columns.index(col) for col in numeric_cols]

# 记录运行时间
s = time.time()
num = 0

# 执行循环，并在此过程中记录时间
for i in range(transaction.shape[0]):
    va = transaction.loc[i].values
    card = va[idx]
    for cate_ind in category_cols_index:
        for num_ind in numeric_cols_index:
            col_name = '&'.join([columns[cate_ind], str(va[cate_ind]), columns[num_ind]])
            features[card][col_name] = features[card].get(col_name, 0) + va[num_ind]
    num += 1
    if num % 1000000 == 0:
        print(time.time() - s, "s")
del transaction
gc.collect()

# 字典转dataframe
df = pd.DataFrame(features).T.reset_index()
del features
cols = df.columns.tolist()
df.columns = ['card_id'] + cols[1:]

# 生成训练集与测试集
train = pd.merge(train, df, how='left', on='card_id')
test =  pd.merge(test, df, how='left', on='card_id')
del df
train.to_csv("preprocess/train_dict.csv", index=False)
test.to_csv("preprocess/test_dict.csv", index=False)

gc.collect()