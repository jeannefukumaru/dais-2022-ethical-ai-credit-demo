# Databricks notebook source
from veritastool.model import ModelContainer
from veritastool.fairness import CreditScoring
import pickle
import os
import numpy as np
import mlflow
import pandas as pd
from databricks import feature_store
from databricks.feature_store import FeatureLookup
import pyspark.pandas as ps
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split
import json
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from databricks_registry_webhooks import RegistryWebhooksClient, HttpUrlSpec, JobSpec

sys.path.append(os.path.abspath("/Workspace/Repos/jeanne.choo@databricks.com/dais-2022-ethical-ai-credit-demo/src"))
from utils import *

# COMMAND ----------

train = pd.read_csv("./data/credit_train_01(1).csv")
test = pd.read_csv("./data/credit_test_01(1).csv")
bronze_data = pd.concat([train, test])

# COMMAND ----------

# MAGIC %md 
# MAGIC # Write raw data to a bronze delta table

# COMMAND ----------

bronze = spark.createDataFrame(bronze_data)

# COMMAND ----------

BRONZE_DELTA_TABLE_PATH = "dbfs:/FileStore/fairness_data/credit_bronze.delta"
dbutils.fs.rm(BRONZE_DELTA_TABLE_PATH, recurse=True)

# COMMAND ----------

bronze.write.mode('overwrite').format('delta').save(BRONZE_DELTA_TABLE_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC # Clean bronze data and write to silver table
# MAGIC 
# MAGIC Reduce classes in `Marriage` column from three to 2

# COMMAND ----------

bronze = spark.read.format("delta").load(BRONZE_DELTA_TABLE_PATH)

# COMMAND ----------

silver = bronze.withColumn('MARRIAGE', F.when((F.col('MARRIAGE') == 0) | (F.col('MARRIAGE') == 3), 1).otherwise(F.col('MARRIAGE')))

# COMMAND ----------

SILVER_DELTA_TABLE_PATH = "dbfs:/FileStore/fairness_data/credit_silver.delta"
dbutils.fs.rm(SILVER_DELTA_TABLE_PATH, recurse=True)

# COMMAND ----------

silver.write.mode('overwrite').format('delta').save(SILVER_DELTA_TABLE_PATH)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Create features with Feature Store 
# MAGIC 
# MAGIC This [dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.
# MAGIC 
# MAGIC ### Payment indicator features
# MAGIC For the PAY_* columns,   
# MAGIC (-2=no credit to pay, -1=pay duly, 0=meeting the minimum payment, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
# MAGIC 
# MAGIC We create a featue called `num_paym_unmet` that sums up the total number of times, over 6 months, that a customer had a delayed payment
# MAGIC 
# MAGIC We also create a second feature called `num_paym_met` that sums up the total number of times, over 6 months, that a customer met their payment obligations

# COMMAND ----------

# MAGIC %md
# MAGIC ![feature-store](/files/jeanne/feature_store_table)

# COMMAND ----------

SILVER_DELTA_TABLE_PATH = "dbfs:/FileStore/fairness_data/credit_silver.delta"
silver = spark.read.format("delta").load(SILVER_DELTA_TABLE_PATH)

# COMMAND ----------

from pyspark.sql import functions as F
from functools import reduce
from operator import add 
pay_cols = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

def count_unmet(col):
  return F.when(F.col(col) > 0, 1).otherwise(0)

def count_met(col):
  return F.when(F.col(col) <= 0, 1).otherwise(0)

silver = silver.withColumn("num_paym_unmet", reduce(add, [count_unmet(x) for x in silver.columns if x in pay_cols]))
silver = silver.withColumn("num_paym_met", reduce(add, [count_met(x) for x in silver.columns if x in pay_cols]))

# COMMAND ----------

def num_paym_unmet_fn(df):
  """
  sums up the total number of times, over 6 months, that a customer had a delayed payment
  """
  def count_unmet(col):
    return F.when(F.col(col) > 0, 1).otherwise(0)
  
  feature_df = df[["USER_ID", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]]
  feature_df = feature_df.withColumn("NUM_PAYM_UNMET", reduce(add, [count_unmet(x) for x in silver.columns if x in pay_cols])) \
                         .select("USER_ID", "NUM_PAYM_UNMET")
  return feature_df

def num_paym_met_fn(df):
  """
  sums up the total number of times, over 6 months, that a customer met their payment obligations
  """
  def count_met(col):
    return F.when(F.col(col) <= 0, 1).otherwise(0)
  
  feature_df = df[["USER_ID", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]]
  feature_df = feature_df.withColumn("NUM_PAYM_MET", reduce(add, [count_met(x) for x in silver.columns if x in pay_cols])) \
                         .select("USER_ID", "NUM_PAYM_MET")
  return feature_df

num_paym_unmet_feature = num_paym_unmet_fn(silver)
num_paym_met_feature = num_paym_met_fn(silver)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE silver_credit_feature_store CASCADE;
# MAGIC CREATE DATABASE IF NOT EXISTS silver_credit_feature_store

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

fs.drop_table("silver_credit_feature_store.num_paym_unmet_feature")
fs.drop_table("silver_credit_feature_store.num_paym_met_feature")

# COMMAND ----------

fs.create_table(
    name='silver_credit_feature_store.num_paym_unmet_feature',
    primary_keys=['USER_ID'],
    df=num_paym_unmet_feature,
    description='Number of times customer delayed payment'
)

fs.create_table(
    name='silver_credit_feature_store.num_paym_met_feature',
    primary_keys=['USER_ID'],
    df=num_paym_met_feature,
    description='Number of times customer made payment',
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Train model and log fairness metrics with MLFlow

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Perform Feature Store lookups get features that will be joined with raw training data

# COMMAND ----------

num_paym_unmet_table = "silver_credit_feature_store.num_paym_unmet_feature"
num_paym_met_table = "silver_credit_feature_store.num_paym_met_feature"
 
num_paym_unmet_feature_lookups = [
    FeatureLookup( 
      table_name = num_paym_unmet_table,
      feature_names = ['NUM_PAYM_UNMET'],
      lookup_key = ['USER_ID'],
    )]
 
num_paym_met_feature_lookups = [
    FeatureLookup( 
      table_name =num_paym_met_table,
      feature_names = ['NUM_PAYM_MET'],
      lookup_key = ['USER_ID'],
    )]

# COMMAND ----------

# MAGIC %md 
# MAGIC #### create training dataset

# COMMAND ----------

exclude_columns = ['USER_ID']

# temp way of dropping original cols from training data
include_columns = [c for c in silver.columns if 'BILL_AMT' in c or 'PAY_AMT' in c]+['USER_ID','LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']+['default']

training_set = fs.create_training_set(
  silver.select(include_columns),
  feature_lookups = num_paym_unmet_feature_lookups + num_paym_met_feature_lookups,
  label = 'default',
  exclude_columns = exclude_columns
)

training_df = training_set.load_df()

# COMMAND ----------

# MAGIC %md
# MAGIC # Begin model training
# MAGIC 
# MAGIC ![veritas](/files/jeanne/veritas.png)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Build train and test data

# COMMAND ----------

data = training_df.toPandas()
 
train, test = train_test_split(data, random_state=123)
X_train = train.drop(['default'], axis=1).reset_index(drop=True)
X_test = test.drop(['default'], axis=1).reset_index(drop=True)
y_train = train['default'].reset_index(drop=True)
y_test = test['default'].reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define scikit-learn training pipeline

# COMMAND ----------

pipeline = Pipeline([("scaling", StandardScaler()), ("model", LogisticRegression(C=0.1, max_iter=4000, random_state=0))])
pipeline

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Define protected variables and groups
# MAGIC In this example we set SEX and MARRIAGE status to be protected variables  
# MAGIC 
# MAGIC Within those protected variables, we also choose which are the privileged  groups. In the case of the `SEX` variable, this is the “male” segment, and in the case of the `MARRIAGE` variable, this is the “married” segment. 

# COMMAND ----------

p_var = ['SEX', 'MARRIAGE']
p_grp = {'SEX': [1], 'MARRIAGE':[1]}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Begin training

# COMMAND ----------

mlflow.sklearn.autolog()
with mlflow.start_run(run_name="credit_scoring") as run:
  pipeline.fit(X_train, y_train)
  pipeline.score(X_test, y_test)
  
  fs.log_model(
    pipeline,
    "credit_scoring",
    flavor=mlflow.sklearn,
    training_set=training_set,
    registered_model_name="credit_scoring"
  )
  
  y_true = np.array(y_test)
  y_pred = np.array(pipeline.predict(X_test))
  y_train = np.array(y_train)
  y_prob = np.array(np.max(pipeline.predict_proba(X_test), axis=1))

  model_name = "credit scoring"
  model_type = "credit"

  container = ModelContainer(y_true=y_true, y_train=y_train, p_var=p_var, p_grp=p_grp, 
  x_train=X_train, x_test =X_test, model_object=pipeline, model_type=model_type,
  model_name=model_name, y_pred=y_pred, y_prob=y_prob)
  
  cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", 
                             perf_metric_name = "balanced_acc", fair_metric_name = "equal_opportunity") 
  
  cre_sco_obj.evaluate()
  cre_sco_obj.tradeoff(output=False)
  
  cre_sco_obj_savepath = "/FileStore/fairness_artifacts"
  if not os.path.exists(cre_sco_obj_savepath):
      dbutils.fs.mkdirs(cre_sco_obj_savepath)
  
  with open("/dbfs/FileStore/fairness_artifacts/cre_sco_obj.pkl", "wb") as outp:
    pickle.dump(cre_sco_obj, outp)
  
  mlflow.log_artifact("/dbfs/FileStore/fairness_artifacts/cre_sco_obj.pkl")
  mlflow.log_dict(cre_sco_obj.fair_conclusion, "fair_conclusion.json")
  mlflow.log_dict(cre_sco_obj.get_fair_metrics_results(), "fair_metrics.json")
  mlflow.log_dict(cre_sco_obj.get_perf_metrics_results(), "perf_metrics.json")
  mlflow.log_dict(get_tradeoff_metrics(cre_sco_obj), "tradeoff.json")
  mlflow.set_tags({"source_data":SILVER_DELTA_TABLE_PATH})
  
run_id = run.info.run_id
artifact_uri = run.info.artifact_uri

# COMMAND ----------

client = MlflowClient()
latest_model_version = client.search_model_versions("name='credit_scoring'")[0].version

client.transition_model_version_stage(
    name="credit_scoring",
    version=latest_model_version,
    stage="Staging"
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ![reproducible](/files/jeanne/reproducible.png)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Recap

# COMMAND ----------

# MAGIC %md 
# MAGIC ![recap](/files/jeanne/recap_demo_workflow_1.png)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Setup webhook

# COMMAND ----------

MODEL_NAME="credit_scoring"
PAT=dbutils.secrets.get(scope = "jeanne_veritas_demo", key = "PAT")
DBHOST=dbutils.secrets.get(scope = "jeanne_veritas_demo", key = "DBHOST")
AZFUNC="jeanne-mlflow-azfunc-webhook"
AZHOOKNAME="MLflowWebHookTransition"
JOB_ID="1062907355331838"

# COMMAND ----------

# clean up any leftover webhooks before creating new onws
whs = RegistryWebhooksClient().list_webhooks(model_name=MODEL_NAME)
wh_ids = [w.id for w in whs]
for id in wh_ids:
  RegistryWebhooksClient().delete_webhook(id=id)

# COMMAND ----------

job_spec = JobSpec(
  job_id=JOB_ID,
  workspace_url=DBHOST,
  access_token=PAT
)
job_webhook = RegistryWebhooksClient().create_webhook(
  model_name="credit_scoring",
  events=["TRANSITION_REQUEST_TO_PRODUCTION_CREATED"],
  job_spec=job_spec,
  description="Job webhook trigger",
  status="ACTIVE"
)

# COMMAND ----------

slack_http_url_spec = HttpUrlSpec(
 url="https://hooks.slack.com/services/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",
  secret="secret_string",
  authorization=f"Bearer {PAT}"
)
slack_http_webhook = RegistryWebhooksClient().create_webhook(
  model_name=MODEL_NAME,
  events=["TRANSITION_REQUEST_TO_PRODUCTION_CREATED"],
  http_url_spec=slack_http_url_spec,
  description=f"CICD for credit scoring model",
  status="ACTIVE"
)


# COMMAND ----------

RegistryWebhooksClient().list_webhooks(model_name="credit_scoring")

# COMMAND ----------

RegistryWebhooksClient().test_webhook(id='f815b36ab5db4b6f8d1f5c4eae7b117f')

# COMMAND ----------

# MAGIC %md 
# MAGIC # Request model transition to Production
# MAGIC (show through the UI)
