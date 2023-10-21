# Snowpark
import json

# Plotting
import matplotlib.pyplot as plt
import numpy as np
# Pandas & json
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
# Models
from sklearn.model_selection import train_test_split
from snowflake.snowpark import functions as F
from snowflake.snowpark.functions import pandas_udf
from snowflake.snowpark.session import Session
from snowflake.snowpark.types import *
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.dataframe import DataFrame


# Read credentials
# Read credentials
with open('creds.json') as f:
    connection_parameters = json.load(f)    
session = Session.builder.configs(connection_parameters).create()
session.sql('CREATE DATABASE IF NOT EXISTS anarsak_DEV_rf').collect()
session.use_database('anarsak_DEV_rf')


session.sql('CREATE SCHEMA IF NOT EXISTS SNOWPARK_CUSTOMER_SPEND_rf').collect()
session.sql("create or replace warehouse anarsak_WH with warehouse_size='3X-LARGE'").collect()

#session.add_packages("scikit-learn==1.2.2", "pandas", "numpy")

snowpark_version = VERSION
print('Database                    : {}'.format(session.get_current_database()))
print('Schema                      : {}'.format(session.get_current_schema()))
print('Warehouse                   : {}'.format(session.get_current_warehouse()))
print('Role                        : {}'.format(session.get_current_role()))
print('Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_version[2]))

customers = pd.read_csv("./EcommerceCustomers.csv")
customers.head()
print(customers.head())
# Create a Snowpark DF from the pandas DF
snowdf = session.createDataFrame(customers)

snowdf.show(2)

# Loading customer data from Snowpark DF to a Snowflake internal table

snowdf.write.mode("overwrite").saveAsTable("customers_news") 

session.table("customers_news").limit(3).show(5)

# Create a pandas data frame from the Snowflake table
custdf = session.table('customers_news').toPandas() 

print(f"'custdf' local dataframe created. Number of records: {len(custdf)} ")

# Start by understanding the correlation matrix for the new data frame
f, ax = plt.subplots(figsize=(10, 8))
ax.set_title('Encoded Correlation Heatmap for Used Vehicles Dataset', pad=12)
sns.heatmap(custdf.corr(), vmin=-1, vmax=1, annot=True, cmap='Spectral')


# Define X and Y for modeling
X = custdf[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
Y = custdf['Yearly Amount Spent']

# Split into training & Testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                 test_size=0.3, random_state=101)

# Create an instance of Linear Regression and Fit the training datasets
rf = RandomForestRegressor()
rf.fit(X_train,y_train)

# Creating a User Defined Function within Snowflake to do the scoring there
def predict_pandas_udf(df: pd.DataFrame) -> pd.Series:
    return pd.Series(rf.predict(df))  

linear_model_vec = pandas_udf(func=predict_pandas_udf,
                                return_type=FloatType(),
                                input_types=[FloatType(),FloatType(),FloatType(),FloatType()],
                                session=session,
                                packages = ("pandas","scikit-learn==1.2.1"), max_batch_size=200)

                                # Calling the UDF to do the scoring (pushing down to Snowflake)
output = session.table('customers_news').select(*list(X.columns),
                    linear_model_vec(list(X.columns)).alias('PREDICTED_SPEND'), 
                    (F.col('Yearly Amount Spent')).alias('ACTUAL_SPEND')
                    )

output.show(5)



output.write.mode("overwrite").saveAsTable("PREDICTED_CUSTOMER_SPEND") 
