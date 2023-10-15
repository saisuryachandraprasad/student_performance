import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass

class DataTransformerConfig:
    preprocessor_obj_file_path = os.path.join("artifact","preprocessor.pkl")

class DataTransformer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()

    def get_data_transformer_object(self):
        """ THIS FUNCTION IS REPONSIBLE TO GET DATA FOR TRANSFORMER"""

        try:
            """ NUMERIC AND CATEGORICAL FEATURES IN DATA"""

            numerical_feature = ["reading_score","writing_score"]
            categorical_feature = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            """ THIS PIPELINE HOLD THE STEPS TO FOLLOW FOR DATA MANIPULATION NUMERIC FEATURES"""
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            """ THIS PIPELINE HOLD THE STEPS TO FOLLOW FOR DATA MANIPULATION CATEGORICAL FEATUES"""
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy= "most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("standardscaling doen for numerical feature")
            logging.info("one hot encoding doen for categorical feature")

            preprocessing = ColumnTransformer(
                [
                    ("numeric_pipeline", numeric_pipeline,numerical_feature),
                    ("categorical_pipeline", categorical_pipeline,categorical_feature)

                ]
            )

            return preprocessing

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_set_path, test_set_path):
        logging.info("data transformation is initiated")
        try:

            train_df = pd.read_csv(train_set_path)
            test_df = pd.read_csv(test_set_path)

            logging.info("Read the data set ")
            logging.info("obtaining teh data processing")

            processing_obj = self.get_data_transformer_object()

            target_column = "math_score"
            numerical_column = ["reading_score","writing_score"]

            input_feature_train_df = train_df.drop(columns= [target_column],axis=1)
            traget_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns= [target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("applying preprocessing object on train and test data")

            input_feature_train_arr = processing_obj.fit_transform(input_feature_train_df)
            inpurt_feature_test_arr = processing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(traget_feature_train_df)]
            test_arr = np.c_[inpurt_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saved preprocessing object")

            save_object(

                file_path = self.data_transformer_config.preprocessor_obj_file_path,
                obj = processing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
