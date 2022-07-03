import pandas as pd

predictions_lgb = pd.read_csv("predictions_lgb.csv")
predictions_xgb = pd.read_csv("predictions_xgb.csv")
predictions_cat = pd.read_csv("predictions_cat.csv")

sub_df = pd.read_csv('data/sample_submission.csv')
sub_df['target'] = (predictions_lgb['target'] + predictions_xgb['target'] + predictions_cat['target']) / 3
sub_df.to_csv('predictions_wei_average.csv', index=False)