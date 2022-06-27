# Databricks notebook source
from bitarray import test
import pandas as pd
import uuid
import numpy as np 

train_03 = pd.read_csv("./data/credit_train_01.csv")
test_03 = pd.read_csv("./data/credit_test_01.csv")
dat = pd.concat([train_03, test_03])

from numpy.random import randint
dat['PAY_AMT1'] = 0
dat['PAY_AMT2'] = 0
dat['PAY_AMT3'] = 0
dat['PAY_AMT4'] = 0
dat['PAY_AMT5'] = 0
dat['PAY_AMT6'] = 0

dat['default'] = randint(0,2, dat.shape[0])

dat.iloc[0:2501, :].to_csv("./data/credit_train_02.csv", index=False)
dat.iloc[2501:, :].to_csv("./data/credit_test_02.csv", index=False)


