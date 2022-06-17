# Databricks notebook source
import json
registry_event = json.loads(dbutils.widgets.get('event_message'))

# COMMAND ----------

registry_event
