from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from src.data_ingestion import DataIngestion    
from utils.common_function import read_yaml
from config.path_config import *

if __name__=="__main__":
    
    ### 1. Data Ingestion
    ingestion = DataIngestion(read_yaml(CONFIG_FILE))
    ingestion.run()
    
    ### 2. Data Processing
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_FILE)
    processor.process()
    
    
    ### 3. Model Training
    trainer = ModelTraining(PROCESSED_TRAIN_FILE_PATH,PROCESSED_TEST_FILE_PATH,MODEL_OUTPUT_PATH)
    trainer.run()