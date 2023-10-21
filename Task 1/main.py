import os
base_dir = os.getcwd()
import json
import pandas as pd

from snowflake.snowpark import functions as F
from snowflake.snowpark import version as v
from snowflake.snowpark.session import Session

from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.preprocessing import KBinsDiscretizer, OneHotEncoder
from snowflake.ml.modeling.impute import SimpleImputer

from snowflake.snowpark.functions import col
import streamlit as st
import snowflake.connector.pandas_tools as sfpd

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import io

import snowflake.snowpark.dataframe


with open('creds.json') as f:
    data = json.load(f)
    USERNAME = data['user']
    PASSWORD = data['password']
    SF_ACCOUNT = data['account']
    DATABASE= data['database']
    SCHEMA=data['schema'] 

CONNECTION_PARAMETERS = {
   "account": SF_ACCOUNT,
   "user": USERNAME,
   "password": PASSWORD,
   "database":DATABASE,
   "schema":SCHEMA
}

session = Session.builder.configs(CONNECTION_PARAMETERS).create()