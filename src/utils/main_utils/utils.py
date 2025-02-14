import yaml
from src.exception.exception import CustomException
from src.logging.logger import logging
import os,sys
import numpy as np
#import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e