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
            logging.info("Initialized DataCleaning with raw_data_path: %s and cleaned_data_path: %s", 
                         raw_data_path, cleaned_data_path)
        except Exception as e:
            logging.error("Error initializing DataCleaning: %s", str(e))
            raise CustomException(e)

    def read_data(self) -> pd.DataFrame:
        try:
            logging.info("Reading raw data from %s", self.raw_data_path)
            df = pd.read_csv(self.raw_data_path)
            logging.info("Successfully read data with shape: %s", df.shape)
            return df
        except Exception as e:
            logging.error("Error reading data: %s", str(e))
            raise CustomException(e)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Handling missing values...")
            missing_before = df.isnull().sum().sum()
            for column, dtype in self._schema_config.items():
                if column in df.columns:
                    if dtype in ["float", "int"]:
                        df[column].fillna(df[column].median(), inplace=True)
                    else:
                        df[column].fillna(df[column].mode()[0], inplace=True)
            missing_after = df.isnull().sum().sum()
            logging.info("Missing values before: %d, after: %d", missing_before, missing_after)
            return df
        except Exception as e:
            logging.error("Error handling missing values: %s", str(e))
            raise CustomException(e)
        
    def handle_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Handling duplicates...")
            duplicates_before = df.duplicated().sum()
            df.drop_duplicates(inplace=True)
            duplicates_after = df.duplicated().sum()
            logging.info("Duplicates before: %d, after: %d", duplicates_before, duplicates_after)
            return df
        except Exception as e:
            logging.error("Error handling duplicates: %s", str(e))
            raise CustomException(e)
        
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Handling outliers...")
            for column in df.select_dtypes(include=[np.number]).columns:
                q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers_before = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
                outliers_after = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                logging.info("Column: %s, Outliers before: %d, after: %d", column, outliers_before, outliers_after)
            return df
        except Exception as e:
            logging.error("Error handling outliers: %s", str(e))
            raise CustomException(e)

    def convert_data_types(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        expected_dtypes = self._schema_config.get("columns", {})

        for column, expected_dtype in expected_dtypes.items():
            if column in dataframe.columns:
                actual_dtype = str(dataframe[column].dtype)

                # Handle string vs object
                if expected_dtype == "string" and actual_dtype == "object":
                    logging.info(f"Column {column}: 'object' treated as 'string', no conversion needed.")
                    continue  # Skip unnecessary conversion
                
                if "int" in expected_dtype:
                    dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").astype("Int64")
                elif "float" in expected_dtype:
                    dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").astype("float64")
                elif "datetime" in expected_dtype:
                    dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce")
                elif expected_dtype == "string":
                    dataframe[column] = dataframe[column].astype("string")

                logging.info(f"Converted Column: {column} | From: {actual_dtype} → To: {expected_dtype}")

        return dataframe




    
    def save_cleaned_data(self, df: pd.DataFrame):
        try:
            logging.info("Saving cleaned data to %s", self.cleaned_data_path)
            os.makedirs(os.path.dirname(self.cleaned_data_path), exist_ok=True)
            df.to_csv(self.cleaned_data_path, index=False, header=True)
            logging.info("Successfully saved cleaned data.")
        except Exception as e:
            logging.error("Error saving cleaned data: %s", str(e))
            raise CustomException(e)  
