import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        path = os.path.dirname(file_path)
        
        os.makedirs(path, exist_ok=True)
        
        with open(file_path, "wb") as obj_file:
            dill.dump(obj, obj_file)
            
    except Exception as e:
        raise CustomException
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    evaluation = dict()
    
    for model_name, model in models.items():
        
        # fit current regressor
        model.fit(X_train, y_train)
        
        y_test_pred = model.predict(X_test)
        
        # calculate r2 score and store in evalutation
        score = r2_score(y_test_pred, y_test)
        evaluation[model_name] = score
        
    return evaluation
        