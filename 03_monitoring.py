# Databricks notebook source
import sys
import os
sys.path.append(os.path.abspath("/Workspace/Repos/jeanne.choo@databricks.com/dais-2022-ethical-ai-credit-demo/src"))

# COMMAND ----------

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import RegressionPreset
from evidently.metric_preset import ClassificationPreset
from evidently.metric_preset import TargetDriftPreset

# COMMAND ----------

import pandas as pd
import numpy as np

from sklearn import datasets, ensemble, model_selection

# COMMAND ----------

#Dataset for Data Quality and Integrity
adult_data = datasets.fetch_openml(name='adult', version=2, as_frame='auto')
adult = adult_data.frame

adult_ref = adult[~adult.education.isin(['Some-college', 'HS-grad', 'Bachelors'])]
adult_cur = adult[adult.education.isin(['Some-college', 'HS-grad', 'Bachelors'])]

adult_cur.iloc[:2000, 3:5] = np.nan
data_drift_report = Report(metrics=[
    DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.2, cat_stattest_threshold=0.2),
])

data_drift_report.run(reference_data=adult_ref, current_data=adult_cur)
data_drift_report.show(mode="inline")

# COMMAND ----------

from modmon import ModMon, ModMonDatasetDrift, ModMonDatasetQuality, ModMonLabelDrift
from utils import *
import pandas as pd
import datetime
from pyspark.sql import functions as F
import uuid
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

# COMMAND ----------

# MAGIC %md 
# MAGIC ![recep_demo_workflow](/files/jeanne_choo@databricks.com/dais-2022/DAIS_2022_ethical_credit_scoring_demo_workflow.png)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Monitoring overview
# MAGIC A general monitoring framework basically consists of:  
# MAGIC - a reference dataset (this is the baseline we want to compare new data against)
# MAGIC - new, incoming data (here we will cal this the "production" dataset) that will be compared against the reference dataset
# MAGIC - drift statistics for labels, features and data (a p < 0.05 indicates that the reference distribution is significantly different from the incoming distribution)
# MAGIC - visualisations to detect drift or data issues that statistics may not capture
# MAGIC
# MAGIC ![monitoring](/files/jeanne_choo@databricks.com/dais-2022/model_monitoring_flowchart.png)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1. Setup monitoring by registering the project and model we want to monitor with the helper `ModMon` class

# COMMAND ----------

project_name = "fairness"
model_name = "credit_scoring"

# COMMAND ----------

monitor = ModMon(project_name, model_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2. Register a baseline dataset
# MAGIC #### The baseline dataset, which will be used for comparison later, is the latest registered model's training dataset

# COMMAND ----------

table_name = monitor.register_baseline_dataset(label_colname="default")

# COMMAND ----------

display(spark.read.load("dbfs:/FileStore/fairness_monitoring/credit_scoring.delta"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Log new, incoming data
# MAGIC ### Six months later, we have more credit data, which we append to the baseline data table

# COMMAND ----------

# write new data to baseline delta table
# ordinarily we would define an ETL pipeline to write to the baseline table
import pandas as pd
from databricks.feature_store import FeatureStoreClient
from pyspark.sql.types import *
train_02 = pd.read_csv("./data/credit_train_02.csv")
test_02 = pd.read_csv("./data/credit_test_02.csv")
new_data = pd.concat([train_02, test_02])

date_colname = "monitoring_date"

new_data = spark.createDataFrame(new_data)

# monitor.add_new_data(new_data, label_colname="default")
from databricks.feature_store import FeatureStoreClient
from pyspark.sql.functions import col
fs = FeatureStoreClient()
latest_model_version = fetch_model_version("credit_scoring").version
scoring_model_uri = f"models:/{model_name}/{latest_model_version}"
label_colname="default"

preds = fs.score_batch(scoring_model_uri, new_data.drop(F.col(label_colname)))
new_data = preds.join(new_data.select("default", "USER_ID"), on="USER_ID")
        
# add metadata columns to new data
new_data = new_data.withColumn(date_colname, F.lit("2023-12-17")) \
                    .withColumn(date_colname, F.to_date(date_colname, "yyyy-MM-dd")) \
                    .withColumn("project_name", F.lit(project_name)) \
                    .withColumn("model_name", F.lit(model_name)) \
                    .withColumn("PAY_AMT1", F.col("PAY_AMT1").cast(DoubleType())) \
                    .withColumn("PAY_AMT2", F.col("PAY_AMT2").cast(DoubleType())) \
                    .withColumn("PAY_AMT3", F.col("PAY_AMT3").cast(DoubleType())) \
                    .withColumn("PAY_AMT4", F.col("PAY_AMT4").cast(DoubleType())) \
                    .withColumn("PAY_AMT5", F.col("PAY_AMT5").cast(DoubleType())) \
                    .withColumn("PAY_AMT6", F.col("PAY_AMT6").cast(DoubleType()))

new_data.write.mode("append").format("delta").option("mergeSchema", "True").save(table_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Extract reference and production data from monitoring delta tables, using dates to segment the data

# COMMAND ----------

ref_start_end = ("2023-01-01", "2023-12-01")
prod_start_end = ("2023-07-01", "2023-12-31")
ref, prod = monitor.get_ref_prod_datasets(table_name, ref_start_end, prod_start_end)

# COMMAND ----------

monitor_df = spark.read.format("delta").load(table_name)

# COMMAND ----------

ref = monitor_df.where((F.col(date_colname) > ref_start_end[0]) & (F.col(date_colname) < ref_start_end[1])).toPandas()
prod = monitor_df.where((F.col(date_colname) > prod_start_end[0]) & (F.col(date_colname) < prod_start_end[1])).toPandas()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Use Evidently, an open source model monitoring package, to visualise dataset drift

# COMMAND ----------

data_drift_and_target_drift_report = Report(metrics=[
    DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.2, cat_stattest_threshold=0.2),TargetDriftPreset()
])

data_drift_and_target_drift_report.run(reference_data=ref, current_data=prod)
data_drift_and_target_drift_report.show(mode="inline")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save drift statistics into a Delta table for tracking

# COMMAND ----------

drift_profile = data_drift_and_target_drift_report.as_dict()

# COMMAND ----------

drift_profile_df = pd.DataFrame(drift_profile["metrics"][0]["result"], index=[0])
drift_profile_df["evidently_preset"] = "DataDriftPreset"

# COMMAND ----------

profile_pyspark = spark.createDataFrame(drift_profile_df)
profile_pyspark = profile_pyspark.withColumn("date", F.current_date()) \
                                         .withColumn("project_name", F.lit(project_name)) \
                                         .withColumn("model_name", F.lit(model_name))
        
output_table = f"dbfs:/{project_name}_monitoring/{model_name}_label_drift.delta"
profile_pyspark.write.format("delta").mode("append").save(output_table)
print(f"label drift data written to {output_table}")

# COMMAND ----------

display(spark.read.load("dbfs:/fairness_monitoring/credit_scoring_label_drift.delta"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Segment labels and predictions by protected variables to see bias

# COMMAND ----------

# DBTITLE 1,Defaulters segmented by SEX variable
import altair as alt
p_var = 'SEX'
tbl = pd.pivot_table(prod.drop("monitoring_date", axis=1), index=['SEX', 'default'], aggfunc=pd.Series.nunique)['USER_ID'].to_frame(name="count_defaulters")
tbl['pct_total'] = (tbl.count_defaulters / tbl.groupby(level=0).count_defaulters.transform(sum) * 100)
tbl = tbl.reset_index()
tbl['SEX'] = tbl['SEX'].apply(lambda x: 'MALE' if x == 1 else 'FEMALE')  
  
alt.Chart(tbl).mark_bar().encode(
  x=f"{p_var}",
  y="pct_total:Q",
  color=f"{p_var}:N",
  column="default:N").properties(height=300, width=300)

# COMMAND ----------

# DBTITLE 1,Predicted defaulters segmented by SEX variable
tbl = pd.pivot_table(prod.drop("monitoring_date", axis=1), index=["SEX", 'prediction'], aggfunc=pd.Series.nunique)['USER_ID'].to_frame(name="count_pred_defaulters")
tbl['pct_total'] = (tbl.count_pred_defaulters / tbl.groupby(level=0).count_pred_defaulters.transform(sum) * 100)
tbl = tbl.reset_index()
tbl['SEX'] = tbl['SEX'].apply(lambda x: 'MALE' if x == 1 else 'FEMALE')  
  
preds_plot = alt.Chart(tbl).mark_bar().encode(
  x=f"{p_var}",
  y="pct_total:Q",
  color=f"{p_var}:N",
  column="prediction:N").properties(height=300, width=300)

preds_plot


# COMMAND ----------

# DBTITLE 1,Defaulters segmented by MARRIAGE variable
import altair as alt
p_var = 'MARRIAGE'
tbl = pd.pivot_table(prod.drop("monitoring_date", axis=1), index=['MARRIAGE', 'default'], aggfunc=pd.Series.nunique)['USER_ID'].to_frame(name="count_defaulters")
tbl['pct_total'] = (tbl.count_defaulters / tbl.groupby(level=0).count_defaulters.transform(sum) * 100)
tbl = tbl.reset_index()
tbl['MARRIAGE'] = tbl['MARRIAGE'].apply(lambda x: 'MARRIED' if x == 1 else 'SINGLE')  
  
alt.Chart(tbl).mark_bar().encode(
  x=f"{p_var}",
  y="pct_total:Q",
  color=f"{p_var}:N",
  column="default:N").properties(height=300, width=300)

# COMMAND ----------

# DBTITLE 1,Predicted defaulters segmented by MARRIAGE variable
tbl = pd.pivot_table(prod.drop("monitoring_date", axis=1), index=["MARRIAGE", 'prediction'], aggfunc=pd.Series.nunique)['USER_ID'].to_frame(name="count_pred_defaulters")
tbl['pct_total'] = (tbl.count_pred_defaulters / tbl.groupby(level=0).count_pred_defaulters.transform(sum) * 100)
tbl = tbl.reset_index()
tbl['MARRIAGE'] = tbl['MARRIAGE'].apply(lambda x: 'MARRIED' if x == 1 else 'SINGLE')  
  
preds_plot = alt.Chart(tbl).mark_bar().encode(
  x=f"{p_var}",
  y="pct_total:Q",
  color=f"{p_var}:N",
  column="prediction:N").properties(height=300, width=300)

preds_plot

# COMMAND ----------


