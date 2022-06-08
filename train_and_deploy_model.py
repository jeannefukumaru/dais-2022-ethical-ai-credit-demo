# Databricks notebook source
# MAGIC %pip install databricks-registry-webhooks

# COMMAND ----------

def get_tradeoff_metrics(cre_sco_obj):
  metrics = ['max_perf_point', 'max_perf_single_th', 'max_perf_neutral_fair']
  tradeoff_dfs = []
  for k in cre_sco_obj.tradeoff_obj.result.keys():
    for m in metrics:
      tradeoff_df = pd.DataFrame({'thresh_1': cre_sco_obj.tradeoff_obj.result[k][m][0], 'thresh_2': cre_sco_obj.tradeoff_obj.result[k][m][1], \
                                 'balanced_accuracy': cre_sco_obj.tradeoff_obj.result[k][m][2]}, index=[0])
      tradeoff_df['metrics']= m
      tradeoff_df['protected_var']= k
      tradeoff_dfs.append(tradeoff_df)
  tradeoff_dfs = pd.concat(tradeoff_dfs)
  return tradeoff_dfs.reset_index().to_json()

# COMMAND ----------

from veritastool.util.utility import test_function_cs
test_function_cs()

# COMMAND ----------

# MAGIC %md
# MAGIC # Load packages

# COMMAND ----------

from veritastool.model import ModelContainer
from veritastool.fairness import CreditScoring
import pickle
import os
import numpy as np
import json
import mlflow
import pandas as pd
import uuid
from databricks import feature_store
from databricks.feature_store import FeatureLookup

import pyspark.pandas as ps
from pyspark.sql import functions as F

from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md
# MAGIC # Read data

# COMMAND ----------

file = "./credit_score_dict.pickle"
input_file = open(file, "rb")
cs = pickle.load(input_file)

# COMMAND ----------

cs['X_train']

# COMMAND ----------

# MAGIC %md 
# MAGIC # Write data to a bronze delta table

# COMMAND ----------

y_train = pd.DataFrame(cs['y_train'], columns=['default'])
y_test = pd.DataFrame(cs['y_test'], columns=['default'])

# COMMAND ----------

train = pd.concat([cs['X_train'], y_train],axis=1)
test = pd.concat([cs['X_test'], y_test], axis=1)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS default.credit_train_bronze;
# MAGIC DROP TABLE IF EXISTS default.credit_test_bronze

# COMMAND ----------

spark_train = spark.createDataFrame(train)
spark_test = spark.createDataFrame(test)
                   
spark_train.write.mode('overwrite').format('delta').saveAsTable('default.credit_train_bronze')
spark_test.write.mode('overwrite').format('delta').saveAsTable('default.credit_test_bronze')

# COMMAND ----------

# MAGIC %md
# MAGIC # Clean data

# COMMAND ----------

# some data cleaning (reduce marriage column into two classes)
bronze_train = spark.table('default.credit_train_bronze')
bronze_test = spark.table('default.credit_test_bronze')

# COMMAND ----------

bronze_train = bronze_train.withColumn('MARRIAGE', F.when((F.col('MARRIAGE') == 0) | (F.col('MARRIAGE') == 3), 1).otherwise(F.col('MARRIAGE'))).to_pandas_on_spark()
bronze_test = bronze_test.withColumn('MARRIAGE', F.when((F.col('MARRIAGE') == 0) | (F.col('MARRIAGE') == 3), 1).otherwise(F.col('MARRIAGE'))).to_pandas_on_spark()

bronze = ps.concat([bronze_train, bronze_test])
bronze['USER_ID'] = [str(uuid.uuid4()) for _ in range(len(bronze.index))]

# COMMAND ----------

# MAGIC %md 
# MAGIC # Save data frame as feature table

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE silver_credit_feature_store CASCADE;
# MAGIC CREATE DATABASE IF NOT EXISTS silver_credit_feature_store

# COMMAND ----------

from pyspark.sql.types import *

def demographic_features_fn(df):
  """
  get demographic features from credit scoring dataset
  """
  demo_df = df[['USER_ID', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']]
  demo_df = demo_df.withColumn('SEX', F.col('SEX').cast(IntegerType())).withColumn('MARRIAGE', F.col('MARRIAGE').cast(IntegerType())).withColumn('EDUCATION', F.col('EDUCATION').cast(IntegerType())).withColumn('AGE', F.col('AGE').cast(IntegerType()))
  return demo_df 

def payment_indicator_fn(df):
  """
  indicates whether monthly bill payment was made
  """
  payment_df = df[['USER_ID', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
  payment_df = payment_df.withColumn('PAY_1', F.col('PAY_1').cast(IntegerType())).withColumn('PAY_2', F.col('PAY_2').cast(IntegerType())).withColumn('PAY_3', F.col('PAY_3').cast(IntegerType())).withColumn('PAY_4', F.col('PAY_4').cast(IntegerType())).withColumn('PAY_5', F.col('PAY_5').cast(IntegerType())).withColumn('PAY_6', F.col('PAY_6').cast(IntegerType()))
  return payment_df

demographic_features = demographic_features_fn(bronze.to_spark())
payment_indicator_features = payment_indicator_fn(bronze.to_spark())

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

fs.drop_table("silver_credit_feature_store.demographic_features")
fs.drop_table("silver_credit_feature_store.payment_indicator_features")

# COMMAND ----------

fs.create_table(
    name='silver_credit_feature_store.demographic_features',
    primary_keys=['USER_ID'],
    df=demographic_features,
    description='Credit Scoring Demographic Features'
)

fs.create_table(
    name='silver_credit_feature_store.payment_indicator_features',
    primary_keys=['USER_ID'],
    df=payment_indicator_features,
    description='Credit Scoring Payment Indicator Features',
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Define protected variables and groups
# MAGIC In this example we set SEX and MARRIAGE status to be protected variables and groups

# COMMAND ----------

p_var = ['SEX', 'MARRIAGE']
p_grp = {'SEX': [1], 'MARRIAGE':[1]}

# COMMAND ----------

# MAGIC %md 
# MAGIC # Define model pipeline

# COMMAND ----------

# rewrite as sklearn pipeline instead of calling from model container
model_object = cs["model"]
model_object

# COMMAND ----------

# MAGIC %md 
# MAGIC # Train model and log fairness metrics with MLFlow

# COMMAND ----------

demographic_features_table = "silver_credit_feature_store.demographic_features"
payment_indicator_features_table = "silver_credit_feature_store.payment_indicator_features"
 
demographic_feature_lookups = [
    FeatureLookup( 
      table_name = demographic_features_table,
      feature_names = ['AGE', 'MARRIAGE', 'SEX', 'EDUCATION'],
      lookup_key = ['USER_ID'],
    )]
 
payment_feature_lookups = [
    FeatureLookup( 
      table_name = payment_indicator_features_table,
      feature_names = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'],
      lookup_key = ['USER_ID'],
    )]

# COMMAND ----------

exclude_columns = []

# temp way of dropping original cols from training data
include_columns = [c for c in bronze.columns if 'BILL_AMT' in c or 'PAY_AMT' in c]+['USER_ID']+['default']

training_set = fs.create_training_set(
  bronze.to_spark().select(include_columns),
  feature_lookups = demographic_feature_lookups + payment_feature_lookups,
  label = 'default',
  exclude_columns = exclude_columns
)

training_df = training_set.load_df()

# COMMAND ----------

data = training_df.toPandas()
 
train, test = train_test_split(data, random_state=123)
X_train = train.drop(['default', 'USER_ID'], axis=1).reset_index(drop=True)
X_test = test.drop(['default', 'USER_ID'], axis=1).reset_index(drop=True)
y_train = train['default'].reset_index(drop=True)
y_test = test['default'].reset_index(drop=True)

# COMMAND ----------

mlflow.sklearn.autolog(silent=True)
with mlflow.start_run(run_name="credit_scoring") as run:
  cs["model"].fit(X_train, y_train)
  cs["model"].score(X_test, y_test)
  
  y_true = np.array(y_test)
  y_pred = np.array(cs['model'].predict(X_test))
  y_train = np.array(y_train)
  y_prob = np.array(np.max(cs['model'].predict_proba(X_test), axis=1))

  model_name = "credit scoring"
  model_type = "credit"

  container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, 
  x_train = X_train,  x_test = X_test, model_object = model_object, model_type  = model_type,
  model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
  
  cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", 
                             perf_metric_name = "balanced_acc", fair_metric_name = "equal_opportunity") 
  
  cre_sco_obj.evaluate()
  cre_sco_obj.tradeoff(output=False)
  mlflow.log_dict(cre_sco_obj.fair_conclusion, "fair_conclusion.json")
  mlflow.log_dict(cre_sco_obj.get_fair_metrics_results(), "fair_metrics.json")
  mlflow.log_dict(cre_sco_obj.get_perf_metrics_results(), "perf_metrics.json")
  mlflow.log_dict(get_tradeoff_metrics(cre_sco_obj), "tradeoff.json")
  
run_id = run.info.run_id
artifact_uri = run.info.artifact_uri

# COMMAND ----------

# MAGIC %md 
# MAGIC # Register model

# COMMAND ----------

run_id

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri=model_uri, name="credit_scoring")

# COMMAND ----------

# MAGIC %md 
# MAGIC # Setup webhook

# COMMAND ----------

MODEL_NAME='credit_scoring'
PAT=dbutils.secrets.get(scope = "jeanne_veritas_demo", key = "PAT")
DBHOST=dbutils.secrets.get(scope = "jeanne_veritas_demo", key = "DBHOST")
AZFUNC='jeanne-mlflow-azfunc-webhook'
AZHOOKNAME='MLflowWebHookTransition'

# COMMAND ----------

from databricks_registry_webhooks import RegistryWebhooksClient, HttpUrlSpec

# COMMAND ----------

http_url_spec = HttpUrlSpec(
  url = "https://jeanne-github-actions.azurewebsites.net/api/HttpExample",
  secret="secret_string",
  authorization=f"Bearer {PAT}"
)
http_webhook = RegistryWebhooksClient().create_webhook(
  model_name=MODEL_NAME,
  events=["TRANSITION_REQUEST_TO_STAGING_CREATED"],
  http_url_spec=http_url_spec,
  description=f"CICD for credit scoring model",
  status="ACTIVE"
)

# COMMAND ----------

http_url_spec = HttpUrlSpec(
#   url="https://jeanne-mlflow-azfunc-webhook.azurewebsites.net",
  url = f"https://{AZFUNC}.azurewebsites.net/api/{AZHOOKNAME}",
  secret="secret_string",
  authorization=f"Bearer {PAT}"
)
http_webhook = RegistryWebhooksClient().create_webhook(
  model_name=MODEL_NAME,
  events=["TRANSITION_REQUEST_TO_STAGING_CREATED"],
  http_url_spec=http_url_spec,
  description=f"CICD for credit scoring model",
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
  events=["TRANSITION_REQUEST_TO_STAGING_CREATED"],
  http_url_spec=slack_http_url_spec,
  description=f"CICD for credit scoring model",
  status="ACTIVE"
)

# COMMAND ----------

RegistryWebhooksClient().list_webhooks(model_name="credit_scoring")

# COMMAND ----------

whs = RegistryWebhooksClient().list_webhooks(model_name="credit_scoring")
wh_ids = [w.id for w in whs]
for id in wh_ids:
  RegistryWebhooksClient().delete_webhook(id=id)

# COMMAND ----------

RegistryWebhooksClient().test_webhook(id='9f839bee548c42ba80903d72f96085d2')

# COMMAND ----------

# MAGIC %md 
# MAGIC # Request model transition to Staging
# MAGIC (show through the UI)

# COMMAND ----------


