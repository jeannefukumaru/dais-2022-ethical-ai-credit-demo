# Databricks notebook source
from bitarray import test
import pandas as pd
import uuid
import numpy as np 

train_03 = pd.read_csv("./data/credit_train_01.csv")
test_03 = pd.read_csv("./data/credit_test_01.csv")
dat = pd.concat([train_03, test_03])

from numpy.random import randint
dat['SEX'] = randint(1, 3, dat.shape[0])
dat['MARRIAGE'] = randint(1, 3, dat.shape[0])
dat['EDUCATION'] = randint(1, 4, dat.shape[0])
dat['AGE'] = randint(1, 100, dat.shape[0])
dat['USER_ID'] = [uuid.uuid4() for _ in range(len(dat.index))]
dat['LIMIT_BAL'] = randint(10000, 100000, dat.shape[0])

dat.iloc[0:2501, :].to_csv("./data/credit_train_02.csv", index=False)
dat.iloc[2500:, :].to_csv("./data/credit_test_02.csv", index=False)


