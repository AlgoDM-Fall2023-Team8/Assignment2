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
session.sql('USE WAREHOUSE AD_FORECAST_DEMO_WH').collect()

st.title("Orginal Dataframe with daily impressions and counts")
original_dataframe=session.sql('select * from daily_impressions;').collect()
st.write(original_dataframe)


session.sql('CALL impressions_forecast!FORECAST(FORECASTING_PERIODS => 14)').collect()
query = '''
SELECT day AS ts, impression_count AS actual, NULL AS forecast, NULL AS lower_bound, NULL AS upper_bound 
FROM daily_impressions 
UNION ALL 
SELECT ts, NULL AS actual, forecast, lower_bound, upper_bound 
FROM TABLE(RESULT_SCAN(-1))
'''
result = session.sql(query)




pandas_dataframe = result.toPandas()
st.title("Dataframe")
st.dataframe(pandas_dataframe)






# Function to create the time series graph
def create_time_series_plot(data):
    print(data)
    df = data
    df['TS'] = pd.to_datetime(df['TS'])
    df.set_index('TS', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['ACTUAL'], label='Actual', color='blue')
    ax.plot(df.index, df['FORECAST'], label='Forecast', color='red')
    ax.fill_between(df.index, df['LOWER_BOUND'], df['UPPER_BOUND'], color='gray', alpha=0.5, label='Prediction Interval')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Graph')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.gcf().autofmt_xdate()
    ax.legend()
    ax.grid(True)
    
    return fig

# Streamlit app
st.title("Time Series Data")

# Display the plot in Streamlit
st.pyplot(create_time_series_plot(pandas_dataframe))