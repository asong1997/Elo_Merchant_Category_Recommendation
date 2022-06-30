"""
随机森林模型预测
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("preprocess/train.csv")
test = pd.read_csv("preprocess/test.csv")



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
train = train[feature_select + ['target']]
test = test[feature_select]


features = train.columns.tolist()
features.remove("card_id")
features.remove("target")

# 因为太耗时间了所以直接给定最优参数
parameter_space = {
    "n_estimators": [80],
    "min_samples_leaf": [30],
    "min_samples_split": [2],
    "max_depth": [10],
    "max_features": ["auto", 80]
}

clf = RandomForestRegressor(
    criterion="mse",
    n_jobs=15,
    random_state=22)


grid = GridSearchCV(clf, parameter_space, cv=2, scoring="neg_mean_squared_error")
grid.fit(train[features].values, train['target'].values)

print(grid.best_params_)
print(grid.best_estimator_)
print(np.sqrt(-grid.best_score_))
# 将结果按照所需要提交的格式写入csv文档
test['target'] = grid.best_estimator_.predict(test[features])
test[['card_id', 'target']].to_csv("result/submission_randomforest.csv", index=False)