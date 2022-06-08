# Databricks notebook source
# Databricks notebook source
dbutils.widgets.text(name = "model_name", defaultValue = "unknown model", label = "Model Name")
dbutils.widgets.text(name = "version", defaultValue="-1",label = "Version")
dbutils.widgets.text(name = "stage", defaultValue="Unknown",label = "To Stage")
dbutils.widgets.text(name = "timestamp", defaultValue="0",label = "Version")
dbutils.widgets.text(name = "text", defaultValue="",label = "Version")
dbutils.widgets.text(name = "webhook_id", defaultValue="",label = "Version")
dbutils.widgets.text(name = "description", defaultValue="", label= "run_id")


# COMMAND ----------

dict = { 
  'model_name': dbutils.widgets.get("model_name"),
  'version': dbutils.widgets.get("version"),
  'stage': dbutils.widgets.get("stage"),
  'timestamp': dbutils.widgets.get("timestamp"),
  'text': dbutils.widgets.get("text"),
  'webhook_id': dbutils.widgets.get("webhook_id"),
  'description': dbutils.widgets.get("description")
}

# COMMAND ----------

from pyspark.sql import Row
import pyspark.sql.functions as F
df = spark.createDataFrame(Row(dict)).withColumn("run_ts", F.current_timestamp())

# COMMAND ----------

df.write.format("delta").mode("append").option("mergeSchema", "true").save("/tmp/jchoo/deployment-history/")

# COMMAND ----------

display(spark.read.format("delta").load("/tmp/jchoo/deployment-history/"))

# COMMAND ----------

from mlflow.tracking import MlflowClient
import json
import pandas as pd
import os
import seaborn as sns
import mlflow

client = mlflow.tracking.MlflowClient()
tmp_path = "/FileStore/jeanne/veritas_artifacts/"
model_name = "credit_scoring"

def download_artifacts(tmp_path, run_id):
    local_dir = tmp_path + run_id
    if not os.path.exists(local_dir):
      dbutils.fs.mkdirs(local_dir)
    
    fairness_conc_fp = client.download_artifacts(run_id, "fair_conclusion.json", "/dbfs" + local_dir)
    fairness_metrics_fp = client.download_artifacts(run_id, "fair_metrics.json", "/dbfs" + local_dir)
    perf_metrics_fp = client.download_artifacts(run_id, "perf_metrics.json", "/dbfs" + local_dir)
    tradeoff_fp = client.download_artifacts(run_id, "tradeoff_metrics.json", "/dbfs" + local_dir)

    print("file downloaded in: {}".format(fairness_conc_fp))
    print("file downloaded in: {}".format(fairness_metrics_fp))
    print("file downloaded in: {}".format(perf_metrics_fp))
    print("file downloaded in: {}".format(tradeoff_fp))
  
    with open(fairness_conc_fp, "r") as file:
      fair_conclusion = json.load(file)
  
    with open(fairness_metrics_fp, "r") as file:
      fair_metrics = json.load(file)
  
    with open(perf_metrics_fp, "r") as file:
      perf_metrics = json.load(file)
  
    with open(tradeoff_fp, "r") as file:
      tradeoff_metrics = json.load(file)
    return {run_id: {"fair_conclusion": fair_conclusion, "fair_metrics": fair_metrics, "perf_metrics": perf_metrics, "tradeoff_metrics": tradeoff_metrics}}
  
def get_current_and_previous_model_run_id(model_name):
    model = client.search_registered_models(filter_string=f"name = '{model_name}'")
    if len(model[0].latest_versions) > 1:
      latest_model_run_id = model[0].latest_versions[0].run_id
      previous_model_run_id = model[0].latest_versions[1].run_id
      return latest_model_run_id, previous_model_run_id
    else:
      return model[0].latest_versions[0].run_id
  
def build_fairness_df(fair_metrics_df_ls, metrics):
    fairness_dfs = []  
    for f in fair_metrics_df_ls:
      model_run_id = list(f.keys())[0]
      protected_var = f[model_run_id]['fair_metrics'].keys()
      for p in protected_var: 
        parity_measure = [f[model_run_id]['fair_metrics'][p][m][0] for m in metrics]
        pvar_df = pd.DataFrame(list(zip(metrics, parity_measure)), columns=['metric', 'parity'])
        pvar_df['protected_var'] = p
        pvar_df['model_run_id'] = model_run_id
        fairness_dfs.append(pvar_df)
    return pd.concat(fairness_dfs)

# COMMAND ----------

latest_model_run_id, previous_model_run_id = get_current_and_previous_model_run_id(model_name)

# COMMAND ----------

latest_model_metrics = download_artifacts(tmp_path, latest_model_run_id)
previous_model_metrics = download_artifacts(tmp_path, previous_model_run_id)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Parity measure

# COMMAND ----------

metrics = ['demographic_parity', 'equal_opportunity', 'equal_odds']
build_fairness_df([latest_model_metrics, previous_model_metrics], metrics)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Performance metrics

# COMMAND ----------

# drop second row because it contains confidence interval values, which we don't need
latest_perf_metrics = pd.DataFrame.from_dict(latest_model_metrics[latest_model_run_id]['perf_metrics']).drop(1)
latest_perf_metrics['run_id'] = latest_model_run_id
previous_perf_metrics = pd.DataFrame(previous_model_metrics[previous_model_run_id]['perf_metrics']).drop(1)
previous_perf_metrics['run_id'] = previous_model_run_id
perf_metrics_df = pd.concat([latest_perf_metrics, previous_perf_metrics])
perf_metrics_df

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Fairness conclusion

# COMMAND ----------

latest_model_metrics[latest_model_run_id]['fair_conclusion']

# COMMAND ----------

previous_model_metrics[previous_model_run_id]['fair_conclusion']

# COMMAND ----------


