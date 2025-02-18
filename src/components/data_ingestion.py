from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception.exception import CustomException
from src.logging.logger import logging
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_URI")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)
        
    def export_collection_as_dataframe(self):
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=self.mongo_client[database_name][collection_name]

            df=pd.DataFrame(list(collection.find().limit(10000)))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            self.export_data_into_feature_store(dataframe)
            return DataIngestionArtifact(feature_store_path=self.data_ingestion_config.feature_store_file_path)
        except Exception as e:
            raise CustomException(e, sys)

        except Exception as e:
            raise CustomException(e,sys)    
        