import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")

class ModelTriner:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr, test_arr):
        try:
            logging.info("Split of train and test data")

            x_train,y_train, x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1,],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "LinearRegression":LinearRegression(),
                "DecisionTree":DecisionTreeRegressor(),
                "KNeighbors":KNeighborsRegressor(),
                "RandomForest":RandomForestRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose= False),
                "GradientBoosting":GradientBoostingRegressor()
            }
            params={
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_model(x_train = x_train, y_train = y_train,x_test=x_test, y_test=y_test, models=models,param=params)

            """ TO GET BEST MODEL SCORE"""
            best_model_score = max(sorted(model_report.values()))

            """ TO GET BEST MODEL"""

            best_model = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_name = models[best_model]

            if best_model_score < 0.75:
                raise CustomException(" No best model found")
            logging.info("found best model")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model_name.predict(x_test)
            predicted_score = r2_score(y_test,predicted)

            return predicted_score
        

        except Exception as e:
            raise CustomException(e,sys)