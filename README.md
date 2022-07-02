### 创建虚拟环境
```bash
conda create -n lightgbm python=3.8
conda activate lightgbm
```

### 安装项目所需第三包
```bash
pip install -r requirements.txt
```

### 分别运行每一个py文件
data_exploration(1.1).py:训练测试数据预处理<br />

data_exploration(1.2).py:商户数据和交易数据的数据探索与数据清洗：方案一 在合并的过程中：对缺失值进行-1填补，然后将所有离散型字段化为字符串类型（为了后续字典合并做准备）<br />

data_exploration(1.3).py:商户数据和交易数据的数据探索与数据清洗：方案二 在合并的过程中：新增两列，分别是purchase_day_diff和purchase_month_diff，
其数据为交易数据以card_id进行groupby并最终提取出purchase_day/month并进行差分的结果。<br />

feature_engineering(2.1).py:基于transaction数据集创建通用组合特征<br />

feature_engineering(2.2).py:基于transaction数据集创建业务统计特征创建<br />

data_merge(2.3).py:对不同方式衍生出来的特征做拼接<br />

random_forest(2.4).py:用随机森林跑baseline<br />

Filter_RF_GridSearchCV(3).py:相关系数选择top300特征，随机森林建模+网格搜索超参数优化（巨耗时）

Wrapper_Lightgbm_TPE(4.1).py：用lightgbm模型的feature_importance筛选top300特征，lightgbm建模+贝叶斯超参数优化

nlp_xgboost_bayes(4.2).py：在数据集中存在大量的ID相关的列（除了card_id外），可以考虑采用NLP中CountVector和TF-IDF两种方法来进行进一步特征衍生，其中CountVector可以挖掘类似某用户钟爱某商铺的信息，

nlp_xgboost_bayes(4.3).py：xgboost建模+贝叶斯产参数优化（耗时较久）

Voting(4.4).py：考虑对random_forest、lightgbm、xgboost做Voting模型融合，如平均融合，加权融合等

Stacking(4.5).py：考虑对random_forest、lightgbm、xgboost做Stacking模型融合

feature_optimization(5.1).py：进一步特征优化






