"""
业务统计特征创建
"""
import gc
import pandas as pd


train = pd.read_csv('preprocess/train_pre.csv')
test = pd.read_csv('preprocess/test_pre.csv')

transaction = pd.read_csv('preprocess/transaction_g_pre.csv')

# 标注离散字段or连续型字段
numeric_cols = ['authorized_flag', 'category_1', 'installments',
                'category_3', 'month_lag', 'purchase_month', 'purchase_day', 'purchase_day_diff', 'purchase_month_diff',
                'purchase_amount', 'category_2',
                'purchase_month', 'purchase_hour_section', 'purchase_day',
                'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']
categorical_cols = ['city_id', 'merchant_category_id', 'merchant_id', 'state_id', 'subsector_id']

# 特征创建
# 创建空字典
aggs = {}

# 连续/离散字段统计量提取范围
for col in numeric_cols:
    aggs[col] = ['nunique', 'mean', 'min', 'max', 'var', 'skew', 'sum']
for col in categorical_cols:
    aggs[col] = ['nunique']
aggs['card_id'] = ['size', 'count']
cols = ['card_id']

# 借助groupby实现统计量计算
for key in aggs.keys():
    cols.extend([key + '_' + stat for stat in aggs[key]])

df = transaction[transaction['month_lag'] < 0].groupby('card_id').agg(aggs).reset_index()
df.columns = cols[:1] + [co + '_hist' for co in cols[1:]]

df2 = transaction[transaction['month_lag'] >= 0].groupby('card_id').agg(aggs).reset_index()
df2.columns = cols[:1] + [co + '_new' for co in cols[1:]]
df = pd.merge(df, df2, how='left', on='card_id')

df2 = transaction.groupby('card_id').agg(aggs).reset_index()
df2.columns = cols
df = pd.merge(df, df2, how='left', on='card_id')
del transaction
gc.collect()

# 生成训练集与测试集
train = pd.merge(train, df, how='left', on='card_id')
test = pd.merge(test, df, how='left', on='card_id')
del df
train.to_csv("preprocess/train_groupby.csv", index=False)
test.to_csv("preprocess/test_groupby.csv", index=False)


