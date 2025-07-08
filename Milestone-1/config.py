# Configuration for file paths and model settings

DATA_DIR = "data"
RESULT_DIR = "data/milestone1_result"

SENTIMENT_INPUT = f"{DATA_DIR}/milestone-1-eng_sentiment.csv"
TOXICITY_INPUT = f"{DATA_DIR}/milestone-1-eng_toxicity.csv"

SENTIMENT_RESULT = f"{RESULT_DIR}/sentiment_analysis_results.csv"
TOXICITY_RESULT = f"{RESULT_DIR}/toxicity_detect_results.csv"

# Model configuration
MODEL_ID = "ibm-granite/granite-3.2-2b-instruct"
DEVICE_MAP = "cuda"  # or "cpu" if you don't have a GPU
MAX_NEW_TOKENS = 200