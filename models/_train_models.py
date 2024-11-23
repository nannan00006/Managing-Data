if __name__ == "__main__":
    """
    This script is independent from any part of the project.
    It's solely used to train and build machine learning models.
    """
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StringIndexerModel, OneHotEncoderModel
    from pyspark.sql.functions import split, col
    from pyspark.ml.regression import LinearRegression

    spark = SparkSession.builder \
        .appName("DataCleaner") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .config("spark.mongodb.input.uri", "mongodb://localhost:27017/brewery.data") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    df = spark.read.format("mongo").load()


    def indexer_and_encoder(data):
        """
        This function is used to develop necessary model for converting categorical variables
        into dummy variable, and save into models file.
        """
        indexer = StringIndexer(inputCols=["Beer_Style", "SKU"], outputCols=["style_index", "sku_index"]).fit(data)
        indexer.save("indexer")
        data = indexer.transform(data).drop("Beer_Style", "SKU")
        encoder = OneHotEncoder(inputCols=["style_index", "sku_index"], outputCols=["beer_style", "sku"]).fit(data)
        encoder.save("encoder")


    def clean_data(data):
        indexer = StringIndexerModel.load("indexer")
        encoder = OneHotEncoderModel.load("encoder")
        data = indexer.transform(data).drop("Beer_Style", "SKU")
        data = encoder.transform(data).drop("style_index", "sku_index")
        return data.withColumn("ingredient1", split(col("Ingredient_Ratio"), ":").getItem(1).cast("float")) \
            .withColumn("ingredient2", split(col("Ingredient_Ratio"), ":").getItem(2).cast("float")) \
            .drop("Ingredient_Ratio")


    def quality_linear_regression(data):
        """
        This function is used to train a linear regression model for quality score,
        and save the model for later prediction.
        """
        feature_columns = ["beer_style", "sku", "Fermentation_Time", "Temperature",
                           "pH_Level", "Gravity", "Alcohol_Content", "Bitterness",
                           "Color", "ingredient1", "ingredient2", "Volume_Produced"]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        data = assembler.transform(data).select("features", 'Quality_Score')
        lr = LinearRegression(featuresCol='features', labelCol='Quality_Score')
        lr_model = lr.fit(data)
        lr_model.save("quality_lr")


    def sales_linear_regression(data):
        """
        This function is used to train a linear regression model for total sales,
        and save the model for later prediction.
        """
        feature_columns = ["beer_style", "sku", "Fermentation_Time", "Temperature",
                           "pH_Level", "Gravity", "Alcohol_Content", "Bitterness",
                           "Color", "ingredient1", "ingredient2", "Volume_Produced"]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        data = assembler.transform(data).select("features", 'Total_Sales')
        lr = LinearRegression(featuresCol='features', labelCol='Total_Sales')
        lr_model = lr.fit(data)
        lr_model.save("sales_lr")


    df = clean_data(df)

    features = ["beer_style", "sku", "Fermentation_Time", "Temperature",
                "pH_Level", "Gravity", "Alcohol_Content", "Bitterness",
                "Color", "ingredient1", "ingredient2", "Volume_Produced",
                "Quality_Score", "Total_Sales"]

    df = df.select(features)

    spark.stop()
