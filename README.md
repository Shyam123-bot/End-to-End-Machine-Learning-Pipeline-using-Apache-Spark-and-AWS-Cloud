# End-to-End-Maching-Learning-Pipeline-using-Apache-Spark-and-AWS-Cloud.
This project implements an end-to-end machine learning pipeline for detecting credit card fraud using distributed computing with Apache Spark and cloud technologies on AWS. It involves data preprocessing, model training, and inferencing using PySpark, and employs various AWS services for storage and deployment.

In this project, we have done the following things:
**1. Data Preprocessing and Model Building with Apache Spark:** The whole data processing workflow—from basic preparation to model training and inference—was powered by Apache Spark. Spark is perfect for activities like feature engineering, scalability, and model training because of its distributed architecture, which makes processing enormous datasets efficient. We were able to easily implement machine learning algorithms like RandomForest thanks to Spark MLlib, which ensured scalability and good performance even with massive data volumes.
**2. Data Storage on AWS S3:** The primary data storage for the raw datasets was AWS S3. We were able to securely and scalable store and retrieve big datasets by using S3. S3 also offers an affordable long-term data storage option, guaranteeing that datasets are easily accessible for upcoming research or model retraining.
**3. Intermediate Data Storage with HDFS:** HDFS, or Hadoop Distributed File System, was used to store data that had already been processed. The processed data was put on HDFS in a structured format called Parquet, which makes it easy to retrieve for upcoming research or model building. By utilizing HDFS, we may save time and computational resources by minimizing the need to repeat preprocessing procedures each time fresh data is examined. This makes it possible to perform analytics in the future—particularly when working with huge datasets—without having to recompute the transforms.
**4. Storing Inferences in AWS DynamoDB:** Predictions or inferences produced by the model are saved in the quick and scalable NoSQL database AWS DynamoDB. We provide real-time data retrieval by storing the model inferences in DynamoDB. This enables external systems to consume the predictions over REST APIs. Because of this, downstream applications like dashboards and real-time fraud detection systems can simply access the model's outputs.
**5. Deployment of Hadoop and Spark on AWS EC2:** AWS EC2 instances were used to build up Apache Spark and Hadoop for the distributed computing environment. These instances offer the flexibility and processing capacity required to grow the system according to the volume of data and processing demands. High availability and cost-effectiveness are ensured by our ability to dynamically modify the infrastructure based on workload by utilizing EC2. To effectively manage data processing and model training duties, the whole big data ecosystem—including Hadoop and Spark—was implemented in a cloud-based environment.

We have implemented machine learning pipeline for credit card fraud detection. The dataset used for this project can be found here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The machine learning algorithm used to detect credit card fraus is RandomForestClassifier.

The data and the model obtained after training is stored in AWS S3.

# Dependencies to install in local
1. HADOOP
2. Apache Spark
3. AWS CLI
4. Python
5. Java

# Steps to setup the project
1. Clone project in your local using:
```
git clone https://github.com/nikitashrestha/ds_project.git
```

2. Create a virtualenv:
```
virtualenv venv
```

3. Activate virtualenv and install following dependecies:
```
source venv/bin/activate
pip install pyspark, pandas, dotenv, boto3
```

# Steps to run the project

1. Go to project root directory.

## Running pre-processing pipeline
python src/preprosess.py {aws_s3_bucket_name} {file_name_to_pre_process}

## Running training pipeline
python src/train.py {name_of_parquet_file_to_preprocess}

## Running inderencing pipeline
python src/test.py {name_of_file_to_inference} {model_path}
