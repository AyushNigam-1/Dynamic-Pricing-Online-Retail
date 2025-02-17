import pandas as pd
import numpy as np
import os
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file
from src.constants.training_pipeline import SCHEMA_FILE_PATH

class DataCleaning:
    def __init__(self, raw_data_path: str, cleaned_data_path: str):
        try:
            self.raw_data_path = raw_data_path
            self.cleaned_data_path = cleaned_data_path
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
           
        except Exception as e:
            raise CustomException(e)

    def read_data(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.raw_data_path)
        except Exception as e:
            raise CustomException(e)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for column, dtype in self._schema_config.items():
                if column in df.columns:
                    if dtype == "float" or dtype == "int":
                        df[column].fillna(df[column].median(), inplace=True)  # Median imputation for numerical
                    else:
                        df[column].fillna(df[column].mode()[0], inplace=True)  # Mode imputation for categorical
            return df
        except Exception as e:
            raise CustomException(e)
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for column in df.select_dtypes(include=[np.number]).columns:
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
            return df
        except Exception as e:
            raise CustomException(e)

    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for column, dtype in self._schema_config.items():
                if column in df.columns:
                    if dtype == "int":
                        df[column] = df[column].astype(int)
                    elif dtype == "float":
                        df[column] = df[column].astype(float)
                    elif dtype == "object":
                        df[column] = df[column].astype(str)
            return df
        except Exception as e:
            raise CustomException(e)
    
    def save_cleaned_data(self, df: pd.DataFrame):
        try:
            os.makedirs(os.path.dirname(self.cleaned_data_path), exist_ok=True)
            df.to_csv(self.cleaned_data_path, index=False, header=True)
        except Exception as e:
            raise CustomException(e)
