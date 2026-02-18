import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import * 
from utils.common_function import read_yaml 
from google.cloud import storage

logger = get_logger(__name__)

class DataIngestion():
    
    def __init__(self,config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train-ratio"]
        
        os.makedirs(RAW_DIR,exist_ok=True)
        
        logger.info(f"Data Ingestion Start With {self.bucket_name} And File Is {self.bucket_file_name}")
        
    def download_data_from_gcp(self):
        
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            
            blob.download_to_filename(RAW_FILE_PATH)
            
            logger.info(f"CSV File Download At {RAW_FILE_PATH}")
            
        except Exception as e:
            logger.error("Error While Downloading CSV File")
            raise CustomException("Failed Downloading CSV File", sys)
        
    def split_data(self):
        
        try:
            
            logger.info("Starting The Splitting Data")
            data = pd.read_csv(RAW_FILE_PATH)
            
            train_data , test_data = train_test_split(data , test_size=1-self.train_test_ratio ,random_state=42)
            
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            
            logger.info(f"Successfully Saved Train Data At {TRAIN_FILE_PATH}")
            logger.info(f"Successfully Saved Test Data At {TEST_FILE_PATH}")
            
        except Exception as e:
            logger.error("Error While Splitting The Data")
            raise CustomException("Failed While Splitting The Data", sys)
        
    def run(self):
        
        try:
            
            logger.info("Starting The Data Ingestion ")
            
            self.download_data_from_gcp()
            self.split_data()
            
            logger.info("Data Ingestion Completed")
            
        except Exception as e:
            logger.error(f"Customexception : {str(e)}")
            raise e
    
if __name__ == "__main__":
    
    ingestion = DataIngestion(read_yaml(CONFIG_FILE))
    ingestion.run()