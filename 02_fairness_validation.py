# Databricks notebook source
# MAGIC %md 
# MAGIC ![recap_demo_workflow](/files/jeanne/recap_demo_workflow_2.png)

# COMMAND ----------

import sys
import os
sys.path.append(os.path.abspath("/Workspace/Repos/jeanne.choo@databricks.com/dais-2022-ethical-ai-credit-demo/src"))
import json
from utils import *

# COMMAND ----------

registry_event = json.loads(dbutils.widgets.get('event_message'))

# COMMAND ----------

# extract job webhook payload information
dict = { 
  'model_name': registry_event["model_name"],
  'version': registry_event["version"],
#   'registered_timestamp': dbutils.widgets.get("event_timestamp"),
  'text': registry_event["text"],
  'webhook_id': registry_event["webhook_id"],
}

# COMMAND ----------

from mlflow.tracking import MlflowClient
import json
import pandas as pd
import os
import seaborn as sns
import mlflow
import pickle
import veritastool
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
tmp_path = "/FileStore/fairness_artifacts/"
model_name = "credit_scoring"

# COMMAND ----------

latest_model_run_id = get_current_and_previous_model_run_id(dict['model_name'])

# COMMAND ----------

cre_sco_obj, latest_model_metrics = download_artifacts(tmp_path, latest_model_run_id)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Parity measure

# COMMAND ----------

metrics = ['demographic_parity', 'equal_opportunity', 'equal_odds']
fairness_df = build_fairness_df([latest_model_metrics], metrics)
sns.catplot(x="metric", y="parity", hue="metric", col="protected_var", kind="bar", data=fairness_df)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Parity checks

# COMMAND ----------

df = fairness_df[fairness_df['model_run_id']==latest_model_run_id]
pvars = df['protected_var'].unique()

# COMMAND ----------

for p in pvars:
  if all(df[df['protected_var']==p]['parity'] > 0.1) == False:
    client.set_model_version_tag(name=dict["model_name"], version=dict["version"], key=f"parity_check_{p}", value=1)
    print("Parity checks passed")
  else:
    client.set_model_version_tag(name=dict["model_name"], version=dict["version"], key=f"parity_check_{p}", value=0)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Performance metrics

# COMMAND ----------

# drop second row because it contains confidence interval values, which we don't need
latest_perf_metrics = pd.DataFrame.from_dict(latest_model_metrics[latest_model_run_id]['perf_metrics']).drop(1)
latest_perf_metrics['run_id'] = latest_model_run_id
latest_perf_metrics

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Performance-tradeoff metrics

# COMMAND ----------

cre_sco_obj.tradeoff()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Fairness conclusion

# COMMAND ----------

pvars = ['SEX', 'MARRIAGE']
fair_conc = []
for p in pvars:
  concl = latest_model_metrics[latest_model_run_id]['fair_conclusion'][p]
  concl['pvar'] = p
  fair_conc.append(concl)
fair_conc_df = pd.DataFrame(fair_conc)

# COMMAND ----------

fair_conc_df

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Save fairness metrics and model metadata into Delta table to keep track of deployment history

# COMMAND ----------

from pyspark.sql import Row
import pyspark.sql.functions as F
df = spark.createDataFrame(Row(dict)).withColumn("run_ts", F.current_timestamp())
df.write.format("delta").mode("append").option("mergeSchema", "true").save("/dbfs:/FileStore/fairness/deployment-history-fairness-metrics.delta")

# COMMAND ----------

display(spark.read.format("delta").load("/dbfs:/FileStore/fairness/deployment-history-fairness-metrics.delta"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ![quant_fairness](/files/jeanne/quantified_fairness.png)

# COMMAND ----------


