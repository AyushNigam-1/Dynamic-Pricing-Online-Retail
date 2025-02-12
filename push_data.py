import os
import sys
import json
import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from dotenv import load_dotenv
import certifi
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_URI")

ca=certifi.where()

class DataExtraction():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def excel_to_json_convertor(self,file_path):
        try:
            data=pd.read_excel(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
if __name__=='__main__':
    FILE_PATH="notebook/data/Online Retail.xlsx"
    DATABASE="MLData"
    Collection="DynamicPricing"
    extobj=DataExtraction()
    records=extobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records=extobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)
        


