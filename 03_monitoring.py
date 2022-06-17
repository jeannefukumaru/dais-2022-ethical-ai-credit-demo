# Databricks notebook source
from ais_model_monitoring import modmon

# COMMAND ----------

import pandas as pd

# COMMAND ----------

modmon = modmon.ModMon("veritas", "credit_scoring")

# COMMAND ----------

baseline_train = pd.read_csv("/FileStore/jeanne/veritas_artifacts/credit_train_01.csv")
baseline_test = pd.read_csv("/FileStore/jeanne/veritas_artifacts/credit_train_02.csv")
baseline = pd.concat([baseline_train, baseline_test])
modmon.register_baseline_dataset(baseline)

# COMMAND ----------

new_data_train = pd.read_csv("/FileStore/jeanne/veritas_artifacts/credit_train_02.csv")

# COMMAND ----------

# write new data to baseline delta table

# COMMAND ----------

# create drift detection visualisations
