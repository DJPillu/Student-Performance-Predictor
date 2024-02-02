import os
import sys
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        path = os.path.dirname(file_path)
        
        os.makedirs(path, exist_ok=True)
        
        with open(file_path, "wb") as obj_file:
            dill.dump(obj, obj_file)
            
    except Exception as e:
        raise CustomException