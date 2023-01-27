import pandas as pd
import sqlite3

#Code within this file was adapted from https://github.com/hfhoffman1144/Heart-Disease-Prediction/blob/main/notebooks/csv_to_sql.ipynb

# Path to the csv file containing the data
CSV_PATH = 'Data/diabetes_binary_health_indicators_BRFSS2015.csv'

# Path to sqlite databse that will be created based on csv file
SQL_PATH = 'Data/database.db'

# Read csv
data = pd.read_csv(CSV_PATH)
print(data.shape)
data.head()

# Creates a connection to the sqlite database and returns it
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except:
        print("Create connection failed")
    return conn

#Creates a table based on the create_table_sql statement in the sqlite database connected to by conn
def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except:
        print("Create table failed")

create_table_str = """CREATE TABLE IF NOT EXISTS diabetes (
                                    Diabetes_binary integer,
                                    HighBP integer,
                                    HighChol integer,
                                    CholCheck integer,
                                    BMI integer,
                                    Smoker integer,
                                    Stroke integer,
                                    HeartDiseaseorAttack integer,
                                    PhysActivity integer,
                                    Fruits integer,
                                    Veggies integer,
                                    HvyAlcoholConsump integer,
                                    AnyHealthcare integer,
                                    NoDocbcCost integer,
                                    GenHlth integer,
                                    MentHlth integer,
                                    PhysHlth integer,
                                    DiffWalk integer,
                                    Sex integer,
                                    Age integer,
                                    Education integer,
                                    Income integer
                                );"""

# Create connection and the associated table for the data
conn = create_connection(SQL_PATH)
if conn is not None:
    create_table(conn, create_table_str)
conn.close()

# Writes the data to the database
conn = create_connection(SQL_PATH)
data.to_sql('diabetes', conn, if_exists='append', index=False)
conn.close()

# Test read
conn = create_connection(SQL_PATH)
data_sql = pd.read_sql("SELECT * FROM diabetes", conn)
conn.close()

data_sql.head()