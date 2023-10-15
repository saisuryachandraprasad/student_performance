import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):

    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException (e,sys)
    

def evaluate_model(x_train,y_train,x_test,y_test,models,param):

    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para =param[list(models.keys())[i]]

            gscv = GridSearchCV(model,para,cv=3)
            gscv.fit(x_train,y_train)

            model.set_params(**gscv.best_params_)
            model.fit(x_train,y_train)

            # model.fit(x_train,y_train) #model training

            y_train_predict = model.predict(x_train)
            y_test_predcit = model.predict(x_test)

            train_model_r2_score = r2_score(y_train,y_train_predict)
            test_model_r2_score = r2_score(y_test,y_test_predcit)

            report[list(models.keys())[i]] = test_model_r2_score

            return report
        
    except Exception as e:
        raise CustomException(e,sys)
    
""" this method is responsible for dumping from pickle file"""
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)