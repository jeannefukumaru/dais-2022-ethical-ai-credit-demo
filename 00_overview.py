# Databricks notebook source
# MAGIC %md 
# MAGIC # Demo overview

# COMMAND ----------

# MAGIC %md
# MAGIC ![features](/files/jeanne/system_features.png)

# COMMAND ----------

# MAGIC %md 
# MAGIC ![workflow](/files/jeanne/demo_workflow.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # Use case
# MAGIC 
# MAGIC Our training dataset is the default of credit card clients data set taken from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
# MAGIC 
# MAGIC The dataset's independent variables include credit payment history from August 2005 to September 2005 with demographic features such as SEX and EDUCATION. The response variable is a binary indicator (1 or 0) stating whether or not a customer will default on his or her credit payment in the subsequent month (October 2005)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Dataset attribute information
# MAGIC source: [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
# MAGIC 
# MAGIC - **ID**: ID of each client
# MAGIC - **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# MAGIC - **SEX**: Gender (1=male, 2=female)
# MAGIC - **EDUCATION**: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# MAGIC - **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
# MAGIC - **AGE**: Age in years
# MAGIC - **PAY_0**: Repayment status in September, 2005 (-2=line of credit not used, -1=pay duly, 0=Use of revolving credit, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above) [source]: (https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/discussion/34608)
# MAGIC - **PAY_2**: Repayment status in August, 2005 (scale same as above)
# MAGIC - **PAY_3** : Repayment status in July, 2005 (scale same as above)
# MAGIC - **PAY_4**: Repayment status in June, 2005 (scale same as above)
# MAGIC - **PAY_5**: Repayment status in May, 2005 (scale same as above)
# MAGIC - **PAY_6**: Repayment status in April, 2005 (scale same as above)
# MAGIC - **BILL_AMT1**: Amount of bill statement in September, 2005 (NT dollar)
# MAGIC - **BILL_AMT2**: Amount of bill statement in August, 2005 (NT dollar)
# MAGIC - **BILL_AMT3**: Amount of bill statement in July, 2005 (NT dollar)
# MAGIC - **BILL_AMT4**: Amount of bill statement in June, 2005 (NT dollar)
# MAGIC - **BILL_AMT5**: Amount of bill statement in May, 2005 (NT dollar)
# MAGIC - **BILL_AMT6**: Amount of bill statement in April, 2005 (NT dollar)
# MAGIC - **PAY_AMT1**: Amount of previous payment in September, 2005 (NT dollar)
# MAGIC - **PAY_AMT2**: Amount of previous payment in August, 2005 (NT dollar)
# MAGIC - **PAY_AMT3**: Amount of previous payment in July, 2005 (NT dollar)
# MAGIC - **PAY_AMT4**: Amount of previous payment in June, 2005 (NT dollar)
# MAGIC - **PAY_AMT5**: Amount of previous payment in May, 2005 (NT dollar)
# MAGIC - **PAY_AMT6**: Amount of previous payment in April, 2005 (NT dollar)
# MAGIC - **default**: Default on payment in the next month (1=yes, 0=no)

# COMMAND ----------


