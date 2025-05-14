import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración general
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_RECORDS = 1000

# Configuración de HuggingFace
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Configuración de ChromaDB
CHROMA_DB_PATH = "./insurance_chroma_db"
COLLECTION_NAME = "insurance"

# Dataset
DATASET_NAME = "bitext/Bitext-insurance-llm-chatbot-training-dataset"
