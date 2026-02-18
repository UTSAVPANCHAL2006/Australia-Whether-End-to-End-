import os
import sys
import joblib
import pandas as pd 
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Rain_Prediction_Project")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_function import read_yaml,load_data
from config.path_config import * 



logger = get_logger(__name__)

class ModelTraining:
    
    def __init__(self,train_path,test_path,model_output_path):
        
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
    
    def load_and_split_data(self):
        
        try:
            logger.info(f"Loading Train Data From {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"Loading Test Data From {self.test_path}")
            test_df = load_data(self.test_path)
            
            X_train = train_df.drop(columns=["RainTomorrow"])
            y_train = train_df["RainTomorrow"]
            X_test = test_df.drop(columns=["RainTomorrow"])
            y_test = test_df["RainTomorrow"]
            
            logger.info("Data Sucessfully Spiltted For Model Training")
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error("Error While Splitting Data")
            raise CustomException("Failed While Spiltting Data" , sys)
        
    def train_xgb(self, X_train , y_train):
        
        try:
            
            logger.info("Intializing Data For Training")
            
            model = XGBClassifier()
            
            logger.info("Training The Model")
            
            model.fit(X_train , y_train)
                
            return model
            
        except Exception as e:
            logger.error("Error While Training The Model")
            raise CustomException("Failed While Training The Model", sys)    
        
    def evaluate_model(self,model,X_test,y_test):
        try:
            
            logger.info("Evaluating The Model ")        
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test, y_pred)

            
            logger.info(f"Accuracy Score : {accuracy}")
            logger.info(f"Precision Score : {precision}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"f1_Score : {f1}")
            
            return { "Accuracy" : accuracy , "Precision" :precision , "Recall" : recall , "f1_Score" : f1}
            
        except Exception as e:
            logger.error("Error While Evaluating Model")
            raise CustomException("Failed While Evaluating Model", sys)
        
    def save_model(self,model):
        
        try:
            
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            
            logger.info("Saving The Model")
            
            joblib.dump(model, self.model_output_path)
            
            logger.info(f"Model Saved At {self.model_output_path}")
            
        except Exception as e:
            logger.error("Error While Saving The Model")
            raise CustomException("Failed While Saving The Model", sys)
        
        
    def run(self):
        try:
            logger.info("Starting our MLFLOW experimentation")

            with mlflow.start_run() as run:

                logger.info("Loading processed train and test data")

                train_df = load_data(self.train_path)
                test_df = load_data(self.test_path)

        
                logger.info("Logging the training and testing dataset to MLFLOW")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

            
                logger.info("Splitting data into training and testing sets")

                X_train = train_df.drop(columns=["RainTomorrow"])
                y_train = train_df["RainTomorrow"]
                X_test = test_df.drop(columns=["RainTomorrow"])
                y_test = test_df["RainTomorrow"]

                
                logger.info("Training XGBoost Classifier")
                model = self.train_xgb(X_train, y_train)

            
                logger.info("Evaluating model")
                metrics = self.evaluate_model(model, X_test, y_test)

        
                logger.info("Logging metrics to MLFLOW")
                mlflow.log_metric("accuracy", metrics["Accuracy"])
                mlflow.log_metric("precision", metrics["Precision"])
                mlflow.log_metric("recall", metrics["Recall"])
                mlflow.log_metric("f1_score", metrics["f1_Score"])


                logger.info("Logging the model into MLFLOW")
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_artifact(self.model_output_path, artifact_path="model")

                logger.info("Logging encoders to MLFLOW")

                encoder_dir = "artifacts/encoders"

                if os.path.exists(encoder_dir):
                    mlflow.log_artifacts(encoder_dir, artifact_path="encoders")


                logger.info("Saving trained model")
                self.save_model(model)

                client = MlflowClient()

                run_id = run.info.run_id
                model_uri = f"runs:/{run_id}/model"
                model_name = "RainPredictionModel"

                try:
                    client.create_registered_model(model_name)
                except:
                    pass  

                model_details = mlflow.register_model(model_uri, model_name)

                print(
                    f"Registered Model: {model_details.name}, "
                    f"Version: {model_details.version}"
                )

                logger.info("Model training pipeline completed successfully")

        except Exception as e:
            logger.error("Error While Model Training Pipeline")
            raise CustomException("Model Training Pipeline Failed", sys)

        
        
if __name__=="__main__":
    
    trainer = ModelTraining(PROCESSED_TRAIN_FILE_PATH,PROCESSED_TEST_FILE_PATH,MODEL_OUTPUT_PATH)
    trainer.run()