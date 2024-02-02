import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_column_transformer(self):
        '''
        This function creates and returns a column based transformer
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            continuous_transformer = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scalar", StandardScaler())
                ]
            )
            logging.info("Numerical columns preprocessing steps created")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            categorical_transformer = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    # ("scalar", StandardScaler())
                ]
            )
            logging.info("Categorical columns preprocessing steps created")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                transformers = [("continuous", continuous_transformer, numerical_columns),
                ("categorical", categorical_transformer, categorical_columns)]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logging.info("Train and Test data read")
            
            preprocessor = self.get_column_transformer()
            logging.info("Preprocessor Object obtained")            
            
            target_column = "math_score"
            # numerical_columns = ["writing_score", "reading_score"]
            
            df_input_train = df_train[[col for col in df_train.columns if col != target_column]] # confirm correctness
            # df_input_train = df_train.drop(columns=[target_column], axis=1)
            df_target_train = df_train[[target_column]]
            
            df_input_test = df_test[[col for col in df_test.columns if col != target_column]] # confirm correctness
            # df_input_test = df_test.drop(columns=[target_column], axis=1)
            df_target_test = df_test[[target_column]]
            
            logging.info("Applying column transformer preprocessor on data")
            
            # print(df_input_train.head(5))
            df_input_train_transformed = preprocessor.fit_transform(df_input_train)
            df_target_test_transformed = preprocessor.transform(df_input_test)
            
            array_train = np.c_[df_input_train_transformed, np.array(df_target_train)]
            array_test = np.c_[df_target_test_transformed, np.array(df_target_test)]
            
            logging.info("Saving preprocessor object")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_file_path,
                obj = preprocessor
            )
            
            return (
                array_train,
                array_test,
                self.data_transformation_config.preprocessor_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)