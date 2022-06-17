# Databricks notebook source
import json
registry_event = json.loads(dbutils.widgets.get('event_message'))

# COMMAND ----------

# extract job webhook payload information
dict = { 
  'model_name': dbutils.widgets.get("model_name"),
  'version': dbutils.widgets.get("version"),
#   'registered_timestamp': dbutils.widgets.get("event_timestamp"),
  'text': dbutils.widgets.get("text"),
  'webhook_id': dbutils.widgets.get("webhook_id"),
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

def download_artifacts(tmp_path, run_id):
    local_dir = tmp_path + run_id
    if not os.path.exists(local_dir):
      dbutils.fs.mkdirs(local_dir)
    
    fairness_conc_fp = client.download_artifacts(run_id, "fair_conclusion.json", "/dbfs" + local_dir)
    fairness_metrics_fp = client.download_artifacts(run_id, "fair_metrics.json", "/dbfs" + local_dir)
    perf_metrics_fp = client.download_artifacts(run_id, "perf_metrics.json", "/dbfs" + local_dir)
    tradeoff_fp = client.download_artifacts(run_id, "tradeoff.json", "/dbfs" + local_dir)
    cre_sco_obj_fp = client.download_artifacts(run_id, "cre_sco_obj.pkl", "/dbfs" + local_dir)
  
    with open(fairness_conc_fp, "r") as file:
      fair_conclusion = json.load(file)
  
    with open(fairness_metrics_fp, "r") as file:
      fair_metrics = json.load(file)
  
    with open(perf_metrics_fp, "r") as file:
      perf_metrics = json.load(file)
  
    with open(tradeoff_fp, "r") as file:
      tradeoff_metrics = json.load(file)
      
    with open(cre_sco_obj_fp, "rb") as f:
      cs = pickle.load(f)
    return cs, {run_id: {"fair_conclusion": fair_conclusion, "fair_metrics": fair_metrics, "perf_metrics": perf_metrics, "tradeoff_metrics": tradeoff_metrics}}
  
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
    return pd.concat(fairness_dfs).reset_index(drop=True)

# COMMAND ----------

latest_model_run_id = get_current_and_previous_model_run_id(dict['model_name'])

# COMMAND ----------

cre_sco_obj, latest_model_metrics = download_artifacts(tmp_path, latest_model_run_id)
# previous_model_metrics = download_artifacts(tmp_path, previous_model_run_id)

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
    client.set_model_version_tag(name=model_name, version=version, key=f"parity_check_{pvar}", value=1)
    print("Parity checks passed")
  else:
    client.set_model_version_tag(name=model_name, version=version, key=f"parity_check_{pvar}", value=0)

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
df.write.format("delta").mode("append").option("mergeSchema", "true").save("/tmp/jchoo/deployment-history-fairness-metrics/")

# COMMAND ----------

display(spark.read.format("delta").load("/tmp/jchoo/deployment-history/"))
