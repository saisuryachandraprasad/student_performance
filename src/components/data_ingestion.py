from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass

class DataIngestionConfig():
    train_set_path = os.path.join("artifact","train.csv")
    test_set_path = os.path.join("artifact","test.csv")
    raw_set_path = os.path.join("artifact","raw.csv")

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")

        try:
            df = pd.read_csv('notebook\stud.csv')
            logging.info("reading data frame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_set_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_set_path,index= False,header=True)

            logging.info("iniatiated train test split")
            train_set, test_set = train_test_split(df, test_size= 0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_set_path,index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_set_path,index= False, header = True)

            logging.info("train test split is completed")

            return(
                self.ingestion_config.test_set_path,
                self.ingestion_config.train_set_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
