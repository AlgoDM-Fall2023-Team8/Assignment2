import streamlit as st
import pandas as pd
from snowflake.connector import connect
from snowflake.ml.modeling.preprocessing import OneHotEncoder

# Streamlit interface
st.title("Customer Life Time Value")

# Input fields
import streamlit as st
import pandas as pd

# Create columns to organize the input fields
col1, col2 = st.columns(2)

# Input fields

# Gender
CD_GENDER = col1.selectbox("CD_GENDER", ["Male", "Female"])

if (CD_GENDER =="Male"):
    CD_GENDER="M"
else:
    CD_GENDER="F"

# Marital Status
CD_MARITAL_STATUS = col1.selectbox("CD_MARITAL_STATUS", ["D", "M", "S", "U", "W"])

# Credit Rating
CD_CREDIT_RATING = col1.selectbox("CD_CREDIT_RATING", ["Good", "High Risk", "Low Risk"])

# Education Status
CD_EDUCATION_STATUS = col1.selectbox("CD_EDUCATION_STATUS", ["2 yr Degree", "4 yr Degree", "Advanced Degree", "College", "Primary", "Secondary", "Unknown"])

# Birth Year
C_BIRTH_YEAR = col2.number_input("C_BIRTH_YEAR", min_value=1900, max_value=2023, value=1990)

# Dependency Count
CD_DEP_COUNT = col2.number_input("CD_DEP_COUNT", min_value=0, value=1)

# Total Sales
TOTAL_SALES = col2.number_input("Income", min_value=0, value=20000)

if st.button("Predict"):
    # Create a dictionary for one-hot encoding
    input_data_dict = {
        "CD_GENDER_F": [1.0 if CD_GENDER == "F" else 0.0],
        "CD_GENDER_M": [1.0 if CD_GENDER == "M" else 0.0],
        "CD_MARITAL_STATUS_D": [1.0 if CD_MARITAL_STATUS == "D" else 0.0],
        "CD_MARITAL_STATUS_M": [1.0 if CD_MARITAL_STATUS == "M" else 0.0],
        "CD_MARITAL_STATUS_S": [1.0 if CD_MARITAL_STATUS == "S" else 0.0],
        "CD_MARITAL_STATUS_U": [1.0 if CD_MARITAL_STATUS == "U" else 0.0],
        "CD_MARITAL_STATUS_W": [1.0 if CD_MARITAL_STATUS == "W" else 0.0],
        "CD_CREDIT_RATING_GOOD": [1.0 if CD_CREDIT_RATING == "Good" else 0.0],
        "CD_CREDIT_RATING_HIGHRISK": [1.0 if CD_CREDIT_RATING == "High Risk" else 0.0],
        "CD_CREDIT_RATING_LOWRISK": [1.0 if CD_CREDIT_RATING == "Low Risk" else 0.0],
        "CD_EDUCATION_STATUS_2YRDEGREE": [1.0 if CD_EDUCATION_STATUS == "2 yr Degree" else 0.0],
        "CD_EDUCATION_STATUS_4YRDEGREE": [1.0 if CD_EDUCATION_STATUS == "4 yr Degree" else 0.0],
        "CD_EDUCATION_STATUS_ADVANCEDDEGREE": [1.0 if CD_EDUCATION_STATUS == "Advanced Degree" else 0.0],
        "CD_EDUCATION_STATUS_COLLEGE": [1.0 if CD_EDUCATION_STATUS == "College" else 0.0],
        "CD_EDUCATION_STATUS_PRIMARY": [1.0 if CD_EDUCATION_STATUS == "Primary" else 0.0],
        "CD_EDUCATION_STATUS_SECONDARY": [1.0 if CD_EDUCATION_STATUS == "Secondary" else 0.0],
        "CD_EDUCATION_STATUS_UNKNOWN": [1.0 if CD_EDUCATION_STATUS == "Unknown" else 0.0],
        "C_BIRTH_YEAR": [C_BIRTH_YEAR],
        "CD_DEP_COUNT": [CD_DEP_COUNT],
        "TOTAL_SALES": [TOTAL_SALES]
    }

    # Create a DataFrame from the input data
    input_data_df = pd.DataFrame(input_data_dict)


    # Create a DataFrame from the input data
    input_data_df = pd.DataFrame(input_data_dict)

    # Connect to Snowflake


    import json
    from snowflake.connector import connect

    # Load connection parameters from creds.json
    # with open('creds.json') as f:
    #     data = json.load(f)

    # user = data['user']
    # password = data['password']
    # account = data['account']


    # Create the Snowflake connection
    connection = connect(
        user='ARYUCK01',
        password='Pass@123',
        account="otqkqbf-uob82367",
        warehouse='FE_AND_INFERENCE_WH',
        database='tpcds_xgboost',
        schema='demo'

    )

    # Now, you have a connection to Snowflake using the parameters from creds.json
    input_data_df = pd.DataFrame(input_data_dict)

        # Call Snowflake UDF
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT TPCDS_PREDICT_CLV({','.join(map(str, input_data_df.values[0]))})")
        prediction = cursor.fetchone()[0]

    st.write(f"Predicted Total Sales: {prediction}")
