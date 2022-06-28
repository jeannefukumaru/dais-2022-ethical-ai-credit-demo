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

# clean up any leftover webhooks before creating new ones
from databricks_registry_webhooks import RegistryWebhooksClient
whs = RegistryWebhooksClient().list_webhooks(model_name="credit_scoring")
wh_ids = [w.id for w in whs]
for id in wh_ids:
  RegistryWebhooksClient().delete_webhook(id=id)

# COMMAND ----------


