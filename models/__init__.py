import os
from pyspark.sql import SparkSession

_path = os.path.dirname(os.path.abspath(__file__))

spark = SparkSession.builder \
        .appName("app") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
