import pandas as pd

data = pd.read_csv("result/submission_randomforest.csv")
data['randomforest'] = data['target'].values

temp = pd.read_csv("result/submission_lightgbm.csv")
data['lightgbm'] = temp['target'].values

temp = pd.read_csv("result/submission_xgboost.csv")
data['xgboost'] = temp['target'].values

print(data.corr())
# 平均融合
data['target'] = (data['randomforest'] + data['lightgbm'] + data['xgboost']) / 3
data[['card_id', 'target']].to_csv("result/voting_avr.csv", index=False)

# 加权融合
data['target'] = data['randomforest'] * 0.2 + data['lightgbm'] * 0.3 + data['xgboost'] * 0.5
data[['card_id', 'target']].to_csv("result/voting_wei1.csv", index=False)
