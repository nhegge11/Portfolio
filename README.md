# Portfolio
Public Portfolio


References
Apache Spark Official Documentation: https://spark.apache.org/docs/latest/
Apache Airflow Official Documentation: https://airflow.apache.org/docs/apache-airflow/stable/
AWS Data Engineering: https://aws.amazon.com/big-data/
Snowflake Documentation: https://docs.snowflake.com/en/

# Data Engineering Tools Cheat Sheet

## dbt (Data Build Tool)

### 1. Basic Workflow

1. **Initialize a dbt Project**
   ```bash
   dbt init [PROJECT_NAME]
Creates a starter project with sample model files and a profiles.yml template.

Configure Profiles

Edit profiles.yml (often in ~/.dbt/ directory) to specify connection details.
Example for a Postgres profile:
yaml
Copy
Edit
my_postgres:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      user: username
      password: password
      port: 5432
      dbname: my_database
      schema: public
Run Models

bash
Copy
Edit
dbt run
Executes SQL transformations defined in your models.

Test Models

bash
Copy
Edit
dbt test
Runs data tests (schema and custom) to ensure data quality and correctness.

Documentation

bash
Copy
Edit
dbt docs generate
dbt docs serve
Generates project documentation and hosts it locally for browsing.

2. Folder Structure
models/: Contains .sql files for each data transformation.
tests/: Stores custom tests (e.g., not_null, unique) in .yml or .sql format.
macros/: Holds custom macros for reusable logic.
3. Best Practices
Use one source file per data source to manage references easily.
Employ schema naming conventions aligned with business domains.
Leverage incremental models for large datasets to reduce overhead:
sql
Copy
Edit
{{
  config(
    materialized='incremental',
    unique_key='id'
  )
}}
SELECT * FROM source_table
WHERE {{ dbt_utils.last_inserted_row() }}
PySpark
1. Initializing Spark
python
Copy
Edit
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CheatSheetApp") \
    .getOrCreate()
Creates a Spark application context to process data in parallel.

2. Reading Data
CSV
python
Copy
Edit
df = spark.read \
  .option("header", "true") \
  .csv("/path/to/file.csv")
JSON
python
Copy
Edit
df = spark.read.json("/path/to/file.json")
Parquet
python
Copy
Edit
df = spark.read.parquet("/path/to/file.parquet")
3. Common DataFrame Operations
Selecting Columns
python
Copy
Edit
df.select("col1", "col2").show()
Filtering Data
python
Copy
Edit
df.filter(df.col1 > 100).show()
Aggregations
python
Copy
Edit
from pyspark.sql.functions import avg, count

df.groupBy("category").agg(avg("value"), count("*")).show()
Joins
python
Copy
Edit
df1.join(df2, df1.key == df2.key, "inner")
WithColumn
python
Copy
Edit
from pyspark.sql.functions import col, lit

df = df.withColumn("new_column", col("existing_col") * 100)
4. Writing Data
CSV
python
Copy
Edit
df.write \
  .option("header", "true") \
  .mode("overwrite") \
  .csv("/output/path/")
Parquet
python
Copy
Edit
df.write.mode("append").parquet("/output/path/")
5. Performance Tips
Partitioning: Repartition or coalesce for optimal shuffle behavior:
python
Copy
Edit
df.repartition(10)
Caching/Persisting: Cache repeatedly used DataFrames to memory/disk:
python
Copy
Edit
df.cache()
Broadcast Joins: Broadcast smaller tables to avoid shuffles:
python
Copy
Edit
from pyspark.sql.functions import broadcast

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 100 * 1024 * 1024)  # 100 MB
df_large.join(broadcast(df_small), "key")
