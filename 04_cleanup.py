# Databricks notebook source
import sys
import os
sys.path.append(os.path.abspath("/Workspace/Repos/jeanne.choo@databricks.com/dais-2022-ethical-ai-credit-demo/src"))
from utils import *

# COMMAND ----------

# remove data 
data_folders = ["dbfs:/FileStore/fairness_artifacts", "dbfs:/FileStore/fairness_data", "dbfs:/FileStore/fairness_monitoring"]
for d in data_folders:
  dbutils.fs.rm(d, recurse=True)

# COMMAND ----------

# remove models
from mlflow.tracking import MlflowClient
client = MlflowClient()
cleanup_registered_model("credit_scoring")

# COMMAND ----------

# remove experiment
model_name="credit_scoring"
filter_string = "name='{}'".format(model_name)
results = client.search_registered_models(filter_string=filter_string)
results

# COMMAND ----------


