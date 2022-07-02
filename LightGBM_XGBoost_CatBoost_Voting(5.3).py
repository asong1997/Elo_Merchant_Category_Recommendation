import pandas as pd

predictions_lgb = pd.read_csv("result/")
predictions_xgb = pd.read_csv("result/")
predictions_cat = pd.read_csv("result/")

sub_df = pd.read_csv('data/sample_submission.csv')
sub_df["target"] = (predictions_lgb + predictions_xgb.values.flatten() + predictions_cat.values.flatten()) / 3
sub_df.to_csv('predictions_wei_average.csv', index=False)