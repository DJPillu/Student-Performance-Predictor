import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def train(self, array_train, array_test):
        try:
            logging.info("Creating Train and Test split")
            
            X_train,y_train,X_test,y_test = (
                array_train[:, :-1],
                array_train[:, -1],
                array_test[:, :-1],
                array_test[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsClassifier(),
                "XGB": XGBRegressor(),
                "Catboosting": CatBoostRegressor(verbose=False),
                "Adaboost": AdaBoostRegressor()
            }
            
            logging.info("Evaluating various Classifiers")
            evaluation = evaluate_models(X_train, y_train, X_test, y_test, models)
                    
            # get the best model based on highest r2 score
            score, best_model_name = max(zip(evaluation.values(), evaluation.keys()))
            logging.info(f"Best Model found: {best_model_name} with an r2 score of {score}")
            
            # save the best model as a pickle file
            save_object(
                file_path = self.model_trainer_config.model_file_path,
                obj = models[best_model_name]
            )
            
            return score
            
        except Exception as e:
            raise CustomException(e, sys)
        
        