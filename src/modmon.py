import json as js

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import RegressionPreset
from evidently.metric_preset import ClassificationPreset
from evidently.metric_preset import TargetDriftPreset

import json
import pandas as pd
import numpy as np
import plotly.offline as py #working offline
import plotly.graph_objs as go
import os

from pyspark.sql import functions as F
from utils import *

from pyspark.sql import SparkSession
from databricks.feature_store import FeatureStoreClient
from mlflow.tracking import MlflowClient
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

class ModMon():
    def __init__(self, project_name, model_name):
        self.project_name = project_name
        self.model_name = model_name
        self.delta_savepath = f"dbfs:/FileStore/{self.project_name}_monitoring/{self.model_name}.delta"
    
    def register_baseline_dataset(self, score_data=True, label_colname=None, date_colname="monitoring_date"):
        """
        log data used to train the model as a comparison baseline
        """
        run = get_run_from_registered_model(registry_model_name=self.model_name)
        print(f"{run}")
        baseline_df = load_delta_table_from_run(run)

        if date_colname in baseline_df.schema.names:  # add timestamp col for later analysis
            return print("monitoring timestamp colname clashes with existing colname, please choose another one")
        
        elif score_data == True:
            print(f"{baseline_df.columns}")
            # log baseline dfs predictions
            fs = FeatureStoreClient()
            latest_model_version = fetch_model_version("credit_scoring").version
            scoring_model_uri = f"models:/{self.model_name}/{latest_model_version}"
            preds = fs.score_batch(scoring_model_uri, baseline_df.drop(F.col(label_colname)))
            print(f"scored predictions")
            baseline_df = preds.join(baseline_df.select("default", "USER_ID"), on="USER_ID")
            
            # set the schema of the baseline dataset as an attribute for future reference
            setattr(self, "schema", baseline_df.schema)
            
            # add metadata information
            baseline_df = baseline_df.withColumn(date_colname, F.current_date()) \
                                    .withColumn("project_name", F.lit(self.project_name)) \
                                    .withColumn("model_name", F.lit(self.model_name))
            
        else:
            baseline_df = baseline_df.withColumn(date_colname, F.current_date()) \
                                    .withColumn("project_name", F.lit(self.project_name)) \
                                    .withColumn("model_name", F.lit(self.model_name))
            setattr(self, "schema", baseline_df.schema)

        if os.path.exists(self.delta_savepath):
            dbutils.fs.rm(self.delta_savepath)   

        baseline_df.write.mode("overwrite").option("overwriteSchema", "True").format("delta").save(self.delta_savepath)
        print(f"baseline data saved to {self.delta_savepath}")
        return self.delta_savepath
      
    def add_new_data(self, new_data, label_colname, date_colname="monitoring_date"):
        # score new data 
        from databricks.feature_store import FeatureStoreClient
        fs = FeatureStoreClient()
        latest_model_version = fetch_model_version("credit_scoring").version
        scoring_model_uri = f"models:/{self.model_name}/{latest_model_version}"
        preds = fs.score_batch(scoring_model_uri, new_data.drop(F.col(label_colname)))
        new_data = preds.join(new_data.select("default", "USER_ID"), on="USER_ID")
        
        # add metadata columns to new data
        new_data = new_data.withColumn(date_colname, F.lit("2022-12-17")) \
                    .withColumn(date_colname, F.to_date(date_colname, "yyyy-MM-dd")) \
                    .withColumn("project_name", F.lit(self.project_name)) \
                    .withColumn("model_name", F.lit(self.model_name))

        new_data.write.mode("append").format("delta").option("mergeSchema", "True").save(self.delta_savepath)
        print(f"new data saved to {self.delta_savepath}")
        print(f"added {new_data.count()} rows")
    
    def get_ref_prod_datasets(self, ref_start_end, prod_start_end, sample_fraction=1.0, date_colname="monitoring_date"):
        """
        collate reference and production datasets for drift comparison
        ref_start_end: tuple of start and end dates used to slice reference data
        prod_start_end: tuple of start and end dates used to slice incoming data 
        returns tuple of ref and prod datasets as pandas df
        """
        monitor_df = spark.read.format("delta").load(self.delta_savepath)
        ref_df = monitor_df.where((F.col(date_colname) > ref_start_end[0]) & (F.col(date_colname) < ref_start_end[1])).toPandas()
        prod_df = monitor_df.where((F.col(date_colname) > prod_start_end[0]) & (F.col(date_colname) < prod_start_end[1])).toPandas()
        return ref_df, prod_df

class ModMonLabelDrift:
    def __init__(self, ref, prod):
        self.ref = ref
        self.prod = prod
        
    def report_label_drift(self, target_drift_type, colmap, html_filepath=None, save_html=False):
        """
        create label drift report
        
        Param
        -----
        target_drift_type: either NumTargetDriftTab or CatTargetDriftTab
        colmap: ColumnMapping saving column type information
        html_filepath: path to save report as html
        save_html: choose to save report as html
        """
        dashboard = Dashboard(tabs=[target_drift_type(verbose_level=1)])
        dashboard.calculate(self.ref, self.prod, column_mapping=colmap)
        if save_html == True:
            dashboard.save(html_filepath)
        return dashboard

    def save_label_drift_data(self, target_profile_section, colmap, project_name, model_name):
        """
        save label drift report data into Delta table
        
        Param
        -----
        target_profile_section: either NumTargetProfileSection or CatTargetProfileSection
        colmap: ColumnMapping saving column type information
        project_name: used as database name when saving data
        model_name: used as table_name when saving data
        """
        profile = Profile(sections=[target_profile_section])
        profile.calculate(self.ref, self.prod, column_mapping=colmap)
        profile_js = js.loads(profile.json())
        profile_pyspark = spark.createDataFrame(pd.DataFrame(profile_js['cat_target_drift']['data']['metrics'], index=[0]).reset_index())
        profile_pyspark = profile_pyspark.withColumn("date", F.current_date()) \
                                         .withColumn("project_name", F.lit(project_name)) \
                                         .withColumn("model_name", F.lit(project_name))
        
        output_table = f"dbfs:/{project_name}_monitoring/{model_name}_label_drift.delta"
        profile_pyspark.write.format("delta").mode("append").save(output_table)
        print(f"label drift data written to {output_table}")

class ModMonDatasetDrift:
    def __init__(self, ref, prod):
        self.ref = ref
        self.prod = prod
        
    def report_dataset_drift(self, colmap, html_filepath=None, save_html=False):
        """
        create dataset drift report
        
        Param
        -----
        colmap: ColumnMapping saving column type information
        html_filepath: path to save report as html
        save_html: choose to save report as html
        """
        dashboard = Dashboard(tabs=[DataDriftTab()])
        dashboard.calculate(self.ref, self.prod, column_mapping=colmap)
        if save_html == True:
            dashboard.save(html_filepath)
        return dashboard

    def save_dataset_drift_data(self, colmap, project_name, model_name):
        """
        save label drift report data into Delta table
        
        Param
        -----
        colmap: ColumnMapping saving column type information
        project_name: used as database name when saving data
        model_name: used as table_name when saving data
        """
        from collections import namedtuple
        profile = Profile(sections=[DataDriftProfileSection()])
        profile.calculate(self.ref, self.prod, column_mapping=colmap)
        profile_js = js.loads(profile.json())
        
        DriftMetrics = namedtuple("DriftMetrics", "feature drift_detected drift_score feature_type stattest_name")
        feat_names = profile_js['data_drift']['data']['cat_feature_names'] + profile_js['data_drift']['data']['num_feature_names']
        datadrift_df = []
        drift_cols = ['drift_detected', 'drift_score', 'feature_type', 'stattest_name']
        for f in feat_names:
            data = {k:v for k,v in profile_js['data_drift']['data']['metrics'][f].items() if k in drift_cols}
            feature_drift_data = DriftMetrics(f, data['drift_detected'], data['drift_score'], data['feature_type'], data['stattest_name'])
            datadrift_df.append(feature_drift_data)
        
        profile_pyspark = spark.createDataFrame(pd.DataFrame(datadrift_df))
        profile_pyspark = profile_pyspark.withColumn("date", F.current_date()) \
                                         .withColumn("project_name", F.lit(project_name)) \
                                         .withColumn("model_name", F.lit(project_name))
        
        output_table = f"dbfs:/FileStore/{project_name}_monitoring/{model_name}_dataset_drift.delta"
        profile_pyspark.write.format("delta").mode("append").save(output_table)
        print(f"label drift data written to {output_table}")


class ModMonDatasetQuality:
    def __init__(self, ref, prod):
        self.ref = ref
        self.prod = prod
        
    def report_dataset_drift(self, colmap, html_filepath=None, save_html=False):
        """
        create dataset quality report
        
        Param
        -----
        colmap: ColumnMapping saving column type information
        html_filepath: path to save report as html
        save_html: choose to save report as html
        """
        dashboard = Dashboard(tabs=[DataQualityTab()])
        dashboard.calculate(self.ref, self.prod, column_mapping=colmap)
        if save_html == True:
            dashboard.save(html_filepath)
        return dashboard

    def save_dataset_quality_data(self, colmap, project_name, model_name):
        """
        save label drift report data into Delta table
        
        Param
        -----
        colmap: ColumnMapping saving column type information
        project_name: used as database name when saving data
        model_name: used as table_name when saving data
        """
        from collections import namedtuple
        profile = Profile(sections=[DataDriftProfileSection()])
        profile.calculate(self.ref, self.prod, column_mapping=colmap)
        profile_js = js.loads(profile.json())
        
        DriftMetrics = namedtuple("DriftMetrics", "feature drift_detected drift_score feature_type stattest_name")
        feat_names = profile_js['data_drift']['data']['cat_feature_names'] + profile_js['data_drift']['data']['num_feature_names']
        datadrift_df = []
        drift_cols = ['drift_detected', 'drift_score', 'feature_type', 'stattest_name']
        for f in feat_names:
            data = {k:v for k,v in profile_js['data_drift']['data']['metrics'][f].items() if k in drift_cols}
            feature_drift_data = DriftMetrics(f, data['drift_detected'], data['drift_score'], data['feature_type'], data['stattest_name'])
            datadrift_df.append(feature_drift_data)
        
        profile_pyspark = spark.createDataFrame(pd.DataFrame(datadrift_df))
        profile_pyspark = profile_pyspark.withColumn("date", F.current_date()) \
                                         .withColumn("project_name", self.project_name) \
                                         .withColumn("model_name", self.project_name)
        
        output_table = f"{project_name}.{model_name}_dataset_drift"
        profile_pyspark.write.format("delta").mode("append").saveAsTable(output_table)
        print(f"label drift data written to {output_table}")