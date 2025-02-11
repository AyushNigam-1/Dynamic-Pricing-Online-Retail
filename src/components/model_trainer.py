import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models , save_object
@dataclass
class ModelTrainerConfig:
    trained_model_file_paths = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        X_train,y_train,X_test,y_test = (
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1]
        )
        models = {
            "Random Forest":RandomForestRegressor(),
            "Decision Tree":DecisionTreeRegressor(),
            "Gradient Regression":GradientBoostingRegressor(),
            "Linear Regression":LinearRegression(),
            "K_Neighbors Classifier":KNeighborsRegressor(),
            "XGBRegressor":XGBRegressor(),
            "AdaBoost Regressor":AdaBoostRegressor()
        }
        params= {
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'splitter': ['best', 'random']
    },
    "Gradient Regression": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "Linear Regression": {
        'fit_intercept': [True, False],
    },
    "K_Neighbors Classifier": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]  # Distance metric (1 = Manhattan, 2 = Euclidean)
    },
    "XGBRegressor": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.3]
    },
    "AdaBoost Regressor": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'loss': ['linear', 'square', 'exponential']
    }
}
        model_report = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[model_report.values()].index(best_model_score)

        best_model = models[best_model_name]

        if best_model_score<0.6:
            raise CustomException("No best model found")
        logging.info("Best found Model on both training and test dataset")

        save_object(
            file_path = self.model_trainer_config.trained_model_file_paths,
            obj = best_model
        )

        predicted = best_model.predict(X_test)

        r2score = r2_score(y_test,predicted)
        return r2_score


