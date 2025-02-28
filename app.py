import numpy as np
import pandas as pd
import certifi
import os
from src.exception.exception import CustomException 
from src.pipeline.train_pipeline import TrainingPipeline
from fastapi import FastAPI, File, UploadFile,Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run
from fastapi.responses import RedirectResponse , JSONResponse
from fastapi.templating import Jinja2Templates
from src.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from src.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME
import pandas as pd
import sys
from src.utils.main_utils.utils import load_object
from src.utils.ml_utils.model.estimator import MLModel
from dotenv import load_dotenv
import pymongo

load_dotenv()
ca = certifi.where()
templates = Jinja2Templates(directory="./templates")
mongo_db_url = os.getenv("MONGODB_URL_KEY")


client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return JSONResponse(content={"message": "Training is successful"})
    except Exception as e:
        raise CustomException(e,sys)
    
@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
        preprocesor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        network_model = MLModel(preprocessor=preprocesor,model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    app_run(host="localhost",port=8080,debug=True)


