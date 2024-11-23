from . import spark, _path
import os
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler, StringIndexerModel, OneHotEncoderModel
from pyspark.sql.functions import split, col


class BreweryFeatures:
    """
    This Class contain all the features for training model,
    as well as make prediction.
    """
    def __init__(self, beer_style, sku, fermentation_time, temperature, ph_level, gravity,
                 alcohol_content, bitterness, color, ingredient1, ingredient2,
                 volume_produced):
        self.beer_style = beer_style
        self.sku = sku
        self.fermentation_time = fermentation_time
        self.temperature = temperature
        self.ph_level = ph_level
        self.gravity = gravity
        self.alcohol_content = alcohol_content
        self.bitterness = bitterness
        self.color = color
        self.ingredient1 = ingredient1
        self.ingredient2 = ingredient2
        self.volume_produced = volume_produced

    @staticmethod
    def features_list():
        return ["beer_style", "sku", "Fermentation_Time", "Temperature",
                "pH_Level", "Gravity", "Alcohol_Content", "Bitterness",
                "Color", "ingredient1", "ingredient2", "Volume_Produced"]

    def to_tuple(self):
        return (self.beer_style, self.sku, self.fermentation_time, self.temperature,
                self.ph_level, self.gravity, self.alcohol_content, self.bitterness,
                self.color, self.ingredient1, self.ingredient2, self.volume_produced)


def clean_data(beer_style, sku, fermentation_time, temperature, ph_level, gravity,
               alcohol_content, bitterness, color, ingredient_ratio, volume_produced):
    """
    This function is used to clean data:
    (make beer_style and sku into dummy variables,
    split ingredient ratio into separate variables).
    It will return BreweryFeatures class for later use.
    """
    data = spark.createDataFrame([(beer_style, sku, ingredient_ratio)],
                                 ["Beer_Style", "SKU", "Ingredient_Ratio"])
    indexer = StringIndexerModel.load(os.path.join(_path, "indexer"))
    encoder = OneHotEncoderModel.load(os.path.join(_path, "encoder"))
    data = indexer.transform(data).drop("Beer_Style", "SKU")
    data = encoder.transform(data).drop("style_index", "sku_index")
    data = data.withColumn("ingredient1", split(col("Ingredient_Ratio"), ":").getItem(1).cast("float")) \
        .withColumn("ingredient2", split(col("Ingredient_Ratio"), ":").getItem(2).cast("float")) \
        .drop("Ingredient_Ratio").collect()
    beer_style = data[0]["beer_style"]
    sku = data[0]["sku"]
    ingredient1 = data[0]["ingredient1"]
    ingredient2 = data[0]["ingredient2"]
    return BreweryFeatures(beer_style, sku, fermentation_time, temperature, ph_level, gravity,
                           alcohol_content, bitterness, color, ingredient1, ingredient2, volume_produced)


def lr_predict(beer_style, sku, fermentation_time, temperature, ph_level, gravity,
               alcohol_content, bitterness, color, ingredient_ratio, volume_produced):
    """
    This function will load the trained model and
    make prediction for quality score and total sales.
    Return value will be a list of predicted quality score and total sales.
    i.e. [quality_score, total_sales]
    """
    brewery_features = clean_data(beer_style, sku, fermentation_time, temperature, ph_level, gravity,
                                  alcohol_content, bitterness, color, ingredient_ratio, volume_produced)
    data = spark.createDataFrame([brewery_features.to_tuple()], brewery_features.features_list())
    quality_model = LinearRegressionModel.load(os.path.join(_path, "quality_lr"))
    sales_model = LinearRegressionModel.load(os.path.join(_path, "sales_lr"))
    assembler = VectorAssembler(inputCols=brewery_features.features_list(), outputCol="features")
    data = assembler.transform(data).select("features")
    quality = quality_model.transform(data).select("prediction").collect()
    sales = sales_model.transform(data).select("prediction").collect()
    return [round(quality[0][0], 2), round(sales[0][0], 2)]
