import os

################## DATA INGESTION ###################

RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_FILE = "config/config.yml"

################# DATA PREPROCESSING ###############

PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_FILE_PATH = os.path.join(PROCESSED_DIR, "train.csv")
PROCESSED_TEST_FILE_PATH = os.path.join(PROCESSED_DIR,"test.csv")

################### MODEL TRAINING ##################

MODEL_OUTPUT_PATH = "artifacts/model/model.pkl"