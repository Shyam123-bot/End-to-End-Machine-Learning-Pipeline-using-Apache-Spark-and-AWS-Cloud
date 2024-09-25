import os
import sys
import shutil
import logging
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from dotenv import load_dotenv

from constants import BUCKET_NAME, TEMP_MODEL_PATH
from utils import get_spark, upload

load_dotenv()

def train_model(df):
    train, test = df.randomSplit([0.7, 0.3], seed=2018)

    # Initialize RandomForestClassifier
    rf_clf = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='Class')

    # Create ParamGrid for hyperparameter tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(rf_clf.numTrees, [50, 100, 150]) \
        .addGrid(rf_clf.maxDepth, [5, 10, 15]) \
        .build()

    # Define the evaluator for accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="accuracy")

    # CrossValidator for model selection with hyperparameter tuning
    crossval = CrossValidator(estimator=rf_clf,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)  # 3-fold cross-validation

    # Fit the cross-validator model on training data
    cvModel = crossval.fit(train)

    # Use the best model from cross-validation
    bestModel = cvModel.bestModel

    # Make predictions on test data
    predictions = bestModel.transform(test)

    # Evaluate accuracy on the test set
    accuracy = evaluator.evaluate(predictions)

    print(f"Training Pipeline: Best Model's accuracy: {accuracy}")

    # Save the best model
    bestModel.write().overwrite().save(TEMP_MODEL_PATH)
    
    # Upload the model to S3 bucket
    upload(BUCKET_NAME, TEMP_MODEL_PATH, "model")

    return accuracy

def main():
    spark = get_spark()
    preprocessed_df = spark.read.parquet(os.path.join(os.environ["HDFS_FILE_PATH"], sys.argv[1]))
    train_model(preprocessed_df)

main()
