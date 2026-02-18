import os
import sys
import pandas as pd 
import numpy as np
import joblib
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_function import read_yaml , load_data
from config.path_config import * 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


logger = get_logger(__name__)

class DataProcessor:
    
    def __init__(self,train_path,test_path,processed_dir,config_path):
        
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
    def process_data(self,df):
        
        try:
            
            logger.info("Starting Our Data Preprocessing")
            
            logger.info("Seprating Num & Cat Cols")
            
            numerical_cols = []
            categorical_cols = []
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
                
            logger.info("Filling The Numerical Missing Value")
            
            for col in numerical_cols:
                df[col]= df[col].fillna(df[col].mean())
                
            logger.info("Filling The Categorical Missing Value")
                
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode().iloc[0]) 
                
            logger.info("Converting Date Str To Real DateTime")
            
            df['Date'] = pd.to_datetime(df['Date'])
            
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            
            logger.info("Droping The Date Column")
            
            df.drop("Date", axis=1 , inplace=True)
            
            return df
            
        except Exception as e:
            logger.error("Error While Doing Preprocessing")
            raise CustomException("Failed While Doing Processing Data", sys)
        
    def save_data(self,df,file_path):
        
        try:
            
            logger.info("Saving Our Data")
            
            df.to_csv(file_path, index=False)
            
            
            logger.info(f"Data Saved Succesfully At {file_path}")
            
        except Exception as e:
            logger.error("Error While Saving The Data {e}")
            raise CustomException("Error While Saving The Data ", sys)
        
    def process(self):
        
        try:
            
            logger.info("Loading Data From RAW Directory")
            
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            
            train_df = self.process_data(train_df)
            test_df = self.process_data(test_df)
        
            X_train = train_df.drop("RainTomorrow", axis=1)
            y_train = train_df["RainTomorrow"]

            X_test = test_df.drop("RainTomorrow", axis=1)
            y_test = test_df["RainTomorrow"]

            logger.info("Appling Label Encoding and saving encoders")
            
            le_cols = ['Location','WindGustDir','WindDir9am','WindDir3pm','RainToday']
            encoder_dir = os.path.join("artifacts", "encoders")
            os.makedirs(encoder_dir, exist_ok=True)
                        
            for col in le_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                joblib.dump(le, os.path.join(encoder_dir, f"{col}.pkl"))
            
            y_le = LabelEncoder()
            y_train = y_le.fit_transform(y_train.astype(str))
            y_test = y_le.transform(y_test.astype(str))
            joblib.dump(y_le, os.path.join(encoder_dir, "RainTomorrow.pkl"))
            
            train_processed = pd.concat([X_train, pd.Series(y_train, name="RainTomorrow")], axis=1)
            test_processed = pd.concat([X_test, pd.Series(y_test, name="RainTomorrow")], axis=1)
            
            self.save_data(train_processed, PROCESSED_TRAIN_FILE_PATH)
            self.save_data(test_processed,PROCESSED_TEST_FILE_PATH)
            
            logger.info("Data Processing Completed Succesfully")
        
        except Exception as e:
            logger.error("Error During Processing Pipeling Step {e}")
            raise CustomException("Error While Data Preprocessing Pipeline",  e)
        
        
if __name__=="__main__":
    
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_FILE)
    processor.process()
    
    
