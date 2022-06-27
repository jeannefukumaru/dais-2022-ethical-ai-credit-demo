from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd
from pyspark.sql import SparkSession
import os
from pyspark.dbutils import DBUtils
import json
import pickle

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

"""
Fairness Metrics Tracking Utility Methods
"""
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

"""
Validation Notebook Utilities
"""
def download_artifacts(tmp_path, run_id):
    local_dir = tmp_path + run_id
    if not os.path.exists(local_dir):
      dbutils.fs.mkdirs(local_dir)
    
    client = MlflowClient()
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
    client = MlflowClient()
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

# source: adapted from https://github.com/chengyin38/dais_2021_drifting_away
"""
MLflow Registry Utility Methods
""" 
def transition_model(model_version, stage):
    """
    Transition a model to a specified stage in MLflow Model Registry using the associated 
    mlflow.entities.model_registry.ModelVersion object.

    :param model_version: mlflow.entities.model_registry.ModelVersion. ModelVersion object to transition
    :param stage: (str) New desired stage for this model version. One of "Staging", "Production", "Archived" or "None"

    :return: A single mlflow.entities.model_registry.ModelVersion object
    """
    client = MlflowClient()
    
    model_version = client.transition_model_version_stage(
        name=model_version.name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=True
    )

    return model_version  
  

def fetch_model_version(registry_model_name, stage="Staging"):
    """
    For a given registered model, return the MLflow ModelVersion object
    This contains all metadata needed, such as params logged etc

    :param registry_model_name: (str) Name of MLflow Registry Model
    :param stage: (str) Stage for this model. One of "Staging" or "Production"

    :return: mlflow.entities.model_registry.ModelVersion
    """
    client = MlflowClient()
    filter_string = f'name="{registry_model_name}"'
    registered_model = client.search_registered_models(filter_string=filter_string)[0]

    if len(registered_model.latest_versions) == 1:
        model_version = registered_model.latest_versions[0]

    else:
        model_version = [model_version for model_version in registered_model.latest_versions if model_version.current_stage == stage][0]

    return model_version

  
def get_run_from_registered_model(registry_model_name, stage="Staging"):
    """
    Get Mlflow run object from registered model

    :param registry_model_name: (str) Name of MLflow Registry Model
    :param stage: (str) Stage for this model. One of "Staging" or "Production"

    :return: mlflow.entities.run.Run
    """
    model_version = fetch_model_version(registry_model_name, stage)
    run_id = model_version.run_id
    run = mlflow.get_run(run_id)

    return run  

"""
MLflow Tracking Utility Methods
"""

def get_delta_version(delta_path):
  """
  Function to get the most recent version of a Delta table give the path to the Delta table
  
  :param delta_path: (str) path to Delta table
  :return: Delta version (int)
  """
  # DeltaTable is the main class for programmatically interacting with Delta tables
  delta_table = DeltaTable.forPath(spark, delta_path)
  # Get the information of the latest commits on this table as a Spark DataFrame. 
  # The information is in reverse chronological order.
  delta_table_history = delta_table.history() 
  
  # Retrieve the lastest Delta version - this is the version loaded when reading from delta_path
  delta_version = delta_table_history.first()["version"]
  
  return delta_version


def load_delta_table_from_run(run):
  """
  Given an MLflow run, load the Delta table which was used for that run,
  using the path and version tracked at tracking time.
  Note that by default Delta tables only retain a commit history for 30 days, meaning
  that previous versions older than 30 days will be deleted by default. This property can
  be updated using the Delta table property delta.logRetentionDuration.
  For more information, see https://docs.databricks.com/delta/delta-batch.html#data-retention
  
  :param run: mlflow.entities.run.Run
  :return: Spark DataFrame
  """
  # remove feature store tables and include only raw delta tables
#   delta_metadata = [metadata for metadata in run.data.tags["sparkDatasourceInfo"].split(sep="\n") if ".db" not in metadata]
#   delta_paths = [path.split(sep=",")[0].replace("path=", "") for path in delta_metadata]
  delta_paths = run.data.tags["source_data"].split(",")
  print(f"Loading Delta table from paths: {delta_paths}")
  if len(delta_paths) < 2:
    df = spark.read.format("delta").load(delta_paths[0])
    return df
  else:
    dfs = [spark.read.format("delta").load(d).toPandas() for d in delta_paths]
    dfs = pd.concat(dfs).reset_index(drop=True)
    return spark.createDataFrame(dfs)
  
def cleanup_registered_model(registry_model_name):
  """
  Utilty function to delete a registered model in MLflow model registry.
  To delete a model in the model registry all model versions must first be archived.
  This function thus first archives all versions of a model in the registry prior to
  deleting the model
  
  :param registry_model_name: (str) Name of model in MLflow model registry
  """
  client = MlflowClient()

  filter_string = f'name="{registry_model_name}"'

  model_versions = client.search_model_versions(filter_string=filter_string)
  
  if len(model_versions) > 0:
    print(f"Deleting following registered model: {registry_model_name}")
    
    # Move any versions of the model to Archived
    for model_version in model_versions:
      try:
        model_version = client.transition_model_version_stage(name=model_version.name,
                                                              version=model_version.version,
                                                              stage="Archived")
      except mlflow.exceptions.RestException:
        pass

    client.delete_registered_model(registry_model_name)
    
  else:
    print("No registered models to delete") 