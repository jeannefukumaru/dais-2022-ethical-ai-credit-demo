# Databricks notebook source
# MAGIC %pip install veritastool

# COMMAND ----------

# MAGIC %pip install pandas==1.3

# COMMAND ----------

import pickle
import numpy as np
# from veritastool.model import ModelContainer
# from veritastool.fairness import CreditScoring

# COMMAND ----------

filename = "./credit_score_dict.pickle"
input_file = open(filename, "rb")
cs = pickle.load(input_file)

# COMMAND ----------

cs['model']

# COMMAND ----------

cs["X_train"]['MARRIAGE'] = cs["X_train"]['MARRIAGE'].replace([0, 3],1)
cs["X_test"]['MARRIAGE'] = cs["X_test"]['MARRIAGE'].replace([0, 3],1)

#Model Container Parameters
y_true = np.array(cs["y_test"])
y_pred = np.array(cs["y_pred"])
y_train = np.array(cs["y_train"])
p_var = ['SEX', 'MARRIAGE']
p_grp = {'SEX': [1], 'MARRIAGE':[1]}
x_train = cs["X_train"]
x_test = cs["X_test"]
model_object = cs["model"]
model_name = "credit scoring"
model_type = "credit"
y_prob = cs["y_prob"]

# COMMAND ----------

container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, 
x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,
model_name =  model_name, y_pred= y_pred, y_prob= y_prob)

cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, 
fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", 
perf_metric_name = "balanced_acc", fair_metric_name = "equal_opportunity")

# COMMAND ----------

cre_sco_obj.tradeoff()

# COMMAND ----------

cre_sco_obj.evaluate(visualize = True)

# COMMAND ----------

cre_sco_obj.feature_importance()

# COMMAND ----------

import pandas as pd
train_01 = pd.concat([x_train, pd.Series(cs["y_train"], name="default")], axis=1)
test_01 = pd.concat([x_test, pd.Series(cs["y_test"], name="default")], axis=1)

# COMMAND ----------

test_01.head()

# COMMAND ----------

outname = 'credit_train_01.csv'
outdir = '/dbfs/FileStore/jeanne/veritas_artifacts/'
train_01.to_csv(outdir+outname, index=False, encoding="utf-8")
outname = 'credit_test_01.csv'
outdir = '/dbfs/FileStore/jeanne/veritas_artifacts/'
test_01.to_csv(outdir+outname, index=False, encoding="utf-8")

# COMMAND ----------

# MAGIC %fs ls /FileStore/jeanne/veritas_artifacts

# COMMAND ----------

train_01.head()

# COMMAND ----------

drop_cols = [c for c in train_01.columns if c.startswith("BILL_AMT") or c.startswith("PAY_AMT")]

# COMMAND ----------

train_02 = train_01.copy(deep=True)
test_02 = test_01.copy(deep=True)

# COMMAND ----------

import random
for d in train_02.columns:
  if d not in ["default"]:
    train_02[d] = 0

for d in test_02:
  if d not in ["default"]:
    train_02[d] = 0

# COMMAND ----------

x_train_02 = train_02.drop(["default"], axis=1) 
y_train_02 = train_02["default"]
x_test_02 = test_02.drop(["default"], axis=1)
y_test_02 = test_02["default"]

# COMMAND ----------

cs["model"].fit(x_train_02, y_train_02)
cs["model"].score(x_test_02, y_test_02)

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri=model_uri, name="credit_scoring")

# COMMAND ----------

dbutils.fs.mkdirs("/FileStore/jeanne/veritas_artifacts")

# COMMAND ----------

outname = 'credit_train_02.csv'
outdir = '/dbfs/FileStore/jeanne/veritas_artifacts/'
train_02.to_csv(outdir+outname, index=False, encoding="utf-8")

# COMMAND ----------

outname = 'credit_test_02.csv'
outdir = '/dbfs/FileStore/jeanne/veritas_artifacts/'
test_02.to_csv(outdir+outname, index=False, encoding="utf-8")

# COMMAND ----------

dbutils.fs.ls("/FileStore/jeanne/veritas_artifacts")

# COMMAND ----------


