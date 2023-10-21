import os
base_dir = os.getcwd()
print(base_dir)
import json
import pandas as pd

from snowflake.snowpark import functions as F
from snowflake.snowpark import version as v
from snowflake.snowpark.session import Session

from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.preprocessing import KBinsDiscretizer, OneHotEncoder
from snowflake.ml.modeling.impute import SimpleImputer

from snowflake.snowpark.functions import col

# Ensure that your credentials are stored in creds.json
with open('creds.json') as f:
    data = json.load(f)
    USERNAME = data['user']
    PASSWORD = data['password']
    SF_ACCOUNT = data['account']

CONNECTION_PARAMETERS = {
   "account": SF_ACCOUNT,
   "user": USERNAME,
   "password": PASSWORD,
}

session = Session.builder.configs(CONNECTION_PARAMETERS).create()

session.sql('CREATE DATABASE IF NOT EXISTS tpcds_xgboost').collect()
session.sql('CREATE SCHEMA IF NOT EXISTS tpcds_xgboost.demo').collect()
session.sql("create or replace warehouse FE_AND_INFERENCE_WH with warehouse_size='3X-LARGE'").collect()
session.sql("create or replace warehouse snowpark_opt_wh with warehouse_size = 'MEDIUM' warehouse_type = 'SNOWPARK-OPTIMIZED'").collect()
session.sql("alter warehouse snowpark_opt_wh set max_concurrency_level = 1").collect()
session.sql("CREATE OR REPLACE STAGE TPCDS_XGBOOST.DEMO.ML_MODELS").collect()
session.use_warehouse('FE_AND_INFERENCE_WH')
session.use_database('tpcds_xgboost')
session.use_schema('demo')
print("S-1")
TPCDS_SIZE_PARAM = 10
SNOWFLAKE_SAMPLE_DB = 'SNOWFLAKE_SAMPLE_DATA' # Name of Snowflake Sample Database might be different...

if TPCDS_SIZE_PARAM == 100: 
    TPCDS_SCHEMA = 'TPCDS_SF100TCL'
elif TPCDS_SIZE_PARAM == 10:
    TPCDS_SCHEMA = 'TPCDS_SF10TCL'
else:
    raise ValueError("Invalid TPCDS_SIZE_PARAM selection")
    
store_sales = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.store_sales')
catalog_sales = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.catalog_sales') 
web_sales = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.web_sales') 
date = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.date_dim')
dim_stores = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.store')
customer = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.customer')
address = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.customer_address')
demo = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.customer_demographics')

store_sales_agged = store_sales.group_by('ss_customer_sk').agg(F.sum('ss_sales_price').as_('total_sales'))
web_sales_agged = web_sales.group_by('ws_bill_customer_sk').agg(F.sum('ws_sales_price').as_('total_sales'))
catalog_sales_agged = catalog_sales.group_by('cs_bill_customer_sk').agg(F.sum('cs_sales_price').as_('total_sales'))
store_sales_agged = store_sales_agged.rename('ss_customer_sk', 'customer_sk')
web_sales_agged = web_sales_agged.rename('ws_bill_customer_sk', 'customer_sk')
catalog_sales_agged = catalog_sales_agged.rename('cs_bill_customer_sk', 'customer_sk')


total_sales = store_sales_agged.union_all(web_sales_agged)
total_sales = total_sales.union_all(catalog_sales_agged)
total_sales = total_sales.group_by('customer_sk').agg(F.sum('total_sales').as_('total_sales'))
customer = customer.select('c_customer_sk','c_current_hdemo_sk', 'c_current_addr_sk', 'c_customer_id', 'c_birth_year')
customer = customer.join(address.select('ca_address_sk', 'ca_zip'), customer['c_current_addr_sk'] == address['ca_address_sk'] )
customer = customer.join(demo.select('cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_credit_rating', 'cd_education_status', 'cd_dep_count'),
                                customer['c_current_hdemo_sk'] == demo['cd_demo_sk'] )
customer = customer.rename('c_customer_sk', 'customer_sk')

print("S0")
print(customer.limit(5).to_pandas())

final_df = total_sales.join(customer, on='customer_sk')
print(final_df.count()
)

session.use_database('tpcds_xgboost')
session.use_schema('demo')
final_df.write.mode('overwrite').save_as_table('feature_store')


session.use_warehouse('snowpark_opt_wh')
session.use_database('tpcds_xgboost')
session.use_schema('demo')


snowdf = session.table("feature_store")
snowdf = snowdf.drop(['CA_ZIP','CUSTOMER_SK', 'C_CURRENT_HDEMO_SK', 'C_CURRENT_ADDR_SK', 'C_CUSTOMER_ID', 'CA_ADDRESS_SK', 'CD_DEMO_SK'])
print(snowdf.limit(5).to_pandas())

from snowflake.snowpark.functions import col

snowdf = snowdf.withColumn("C_BIRTH_YEAR", col("C_BIRTH_YEAR").cast("float"))
snowdf = snowdf.withColumn("CD_DEP_COUNT", col("CD_DEP_COUNT").cast("float"))





cat_cols = ['CD_GENDER', 'CD_MARITAL_STATUS', 'CD_CREDIT_RATING', 'CD_EDUCATION_STATUS']
num_cols = ['C_BIRTH_YEAR', 'CD_DEP_COUNT']

from snowflake.ml.modeling.impute import SimpleImputer

# my_imputer = SimpleImputer(
#     input_cols=num_cols,    # Specify the input column with missing values
#     output_cols=num_cols,  # Specify the output column to store imputed values
#     strategy='constant',           # Specify the imputation strategy (in this case, constant)
#     fill_value='OTHER'             # Specify the constant value to fill missing values with
# )

# my_imputer.fit(final_df)
# my_sdf = my_imputer.transform(final_df)
# Imputation of Numeric Cols
my_imputer = SimpleImputer(input_cols= num_cols,
                           output_cols= num_cols,
                           strategy='median')
sdf_prepared = my_imputer.fit(snowdf).transform(snowdf)

# OHE of Categorical Cols
my_ohe_encoder = OneHotEncoder(input_cols=cat_cols, output_cols=cat_cols, drop_input_cols=True)
sdf_prepared = my_ohe_encoder.fit(sdf_prepared).transform(sdf_prepared)
print("S1")
print(sdf_prepared.limit(5).to_pandas())
print(sdf_prepared.columns)
print("S2")
# Cleaning column names to make it easier for future referencing
import re

cols = sdf_prepared.columns
for old_col in cols:
    new_col = re.sub(r'[^a-zA-Z0-9_]', '', old_col)
    new_col = new_col.upper()
    sdf_prepared = sdf_prepared.withColumnRenamed(old_col, new_col)

    # Prepare Data for modeling
session.use_warehouse('snowpark_opt_wh')

feature_cols = sdf_prepared.columns
feature_cols.remove('TOTAL_SALES')
target_col = 'TOTAL_SALES'
# Save the train and test sets as time stamped tables in Snowflake
snowdf_train, snowdf_test = sdf_prepared.random_split([0.8, 0.2], seed=82) 
snowdf_train.write.mode("overwrite").save_as_table("tpcds_xgboost.demo.tpc_TRAIN")
snowdf_test.write.mode("overwrite").save_as_table("tpcds_xgboost.demo.tpc_TEST")

xgbmodel = XGBRegressor(random_state=123, input_cols=feature_cols, label_cols=target_col, output_cols='PREDICTION')

print("S")
print(snowdf_train)
print("S")
snowdf_train = snowdf_train.sample(n=1000)
xgbmodel.fit(snowdf_train)
print(snowdf_train.columns)
print("S100")
sdf_scored = xgbmodel.predict(snowdf_test)
print(snowdf_test.columns)
print(sdf_scored.limit(5).to_pandas)

snowdf_test = session.table('tpc_TEST')
# Predicting with sample dataset
sample_data = snowdf_test.limit(100)
session.use_database('tpcds_xgboost')
session.use_schema('demo')

snowdf_test = session.table('tpc_TEST')
# Predicting with sample dataset
sample_data = snowdf_test.limit(100)

sample_data.write.mode("overwrite").save_as_table("temp_test")
test_sdf = session.table('temp_test')

import joblib
import cachetools
xgb_file = xgbmodel.to_xgboost()
xgb_file
MODEL_FILE = 'model.joblib.gz'
joblib.dump(xgb_file, MODEL_FILE)

session.file.put(MODEL_FILE, "@ML_MODELS", auto_compress=False, overwrite=True)


from snowflake.snowpark.functions import udf
import snowflake.snowpark.types as T
# Define a simple scoring function
from cachetools import cached

@cached(cache={})
def load_model(model_path: str) -> object:
    from joblib import load
    model = load(model_path)
    return model

def udf_score_xgboost_model_vec_cached(df: pd.DataFrame) -> pd.Series:
    import os
    import sys
    # file-dependencies of UDFs are available in snowflake_import_directory
    IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
    import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
    model_name = 'model.joblib.gz'
    model = load_model(import_dir+model_name)
    df.columns = feature_cols
    scored_data = pd.Series(model.predict(df))
    return scored_data

# Register UDF
# Define the list of expected feature names
# expected_feature_names = [
#     'CD_GENDER_F', 'CD_GENDER_M', 'CD_MARITAL_STATUS_D', 'CD_MARITAL_STATUS_M', 
#     'CD_MARITAL_STATUS_S', 'CD_MARITAL_STATUS_U', 'CD_MARITAL_STATUS_W', 
#     'CD_CREDIT_RATING_GOOD', 'CD_CREDIT_RATING_HIGH_RISK', 'CD_CREDIT_RATING_LOW_RISK', 
#     'CD_CREDIT_RATING_UNKNOWN', 'CD_EDUCATION_STATUS_2_YR_DEGREE', 
#     'CD_EDUCATION_STATUS_4_YR_DEGREE', 'CD_EDUCATION_STATUS_ADVANCED_DEGREE', 
#     'CD_EDUCATION_STATUS_COLLEGE', 'CD_EDUCATION_STATUS_PRIMARY', 
#     'CD_EDUCATION_STATUS_SECONDARY', 'CD_EDUCATION_STATUS_UNKNOWN', 
#     'C_BIRTH_YEAR', 'CD_DEP_COUNT'
# ]

# # Rename the feature names in feature_cols to match the expected feature names
# modified_feature_cols = [
#     'CD_GENDER_F', 'CD_GENDER_M', 'CD_MARITAL_STATUS_D', 'CD_MARITAL_STATUS_M', 
#     'CD_MARITAL_STATUS_S', 'CD_MARITAL_STATUS_U', 'CD_MARITAL_STATUS_W', 
#     'CD_CREDIT_RATING_GOOD', '"CD_CREDIT_RATING_LOW_RISK"', 
#     'CD_CREDIT_RATING_UNKNOWN', 'CD_EDUCATION_STATUS_2_YR_DEGREE', 
#     'CD_EDUCATION_STATUS_4_YR_DEGREE', 'CD_EDUCATION_STATUS_ADVANCED_DEGREE', 
#     'CD_EDUCATION_STATUS_COLLEGE', 'CD_EDUCATION_STATUS_PRIMARY', 
#     'CD_EDUCATION_STATUS_SECONDARY', 'CD_EDUCATION_STATUS_UNKNOWN', 
#     'C_BIRTH_YEAR', 'CD_DEP_COUNT'
# ]

# Register the UDF with the modified feature names
udf_clv = session.udf.register(func=udf_score_xgboost_model_vec_cached, 
                               name="TPCDS_PREDICT_CLV", 
                               stage_location='@ML_MODELS',
                               input_types=[T.FloatType()]*len(feature_cols),
                               return_type = T.FloatType(),
                               replace=True, 
                               is_permanent=True, 
                               imports=['@ML_MODELS/model.joblib.gz'],
                               packages=['pandas',
                                         'xgboost',
                                         'joblib',
                                         'cachetools'], 
                               session=session)
print("Aaaaaaaa")
print(feature_cols)
print("bbbb")
# Apply the UDF with modified feature names to the Snowflake DataFrame
test_sdf_w_preds = test_sdf.with_column('PREDICTED', udf_clv(*feature_cols))



# test_sdf_w_preds = test_sdf.with_column('PREDICTED', udf_clv(*feature_cols))
print(test_sdf_w_preds.limit(2).to_pandas())

test_sdf_w_preds = test_sdf.with_column('PREDICTED',F.call_udf("TPCDS_PREDICT_CLV",
                                                               [F.col(c) for c in feature_cols]))
print(test_sdf_w_preds.limit(2).to_pandas())








