import json
from pathlib import Path
import os

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
# huggingface mirror
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # Uncomment this line if you want to use a specific Hugging Face mirror
# os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache/")

# OpenAI Configuration
REMOTE_URL = "https://api.openai.com/v1/chat/completions"
# TOKEN = "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"
TOKEN="tmp"
DEFAULT_CHATAGENT_MODEL = "gpt-4o-mini"
ADVANCED_CHATAGENT_MODEL = "gpt-4o"

# Gemini Configuration
load_dotenv(Path(".env"))
GEMINI_API_KEY = os.getenv("API_KEY") 
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
ADVANCED_GEMINI_MODEL = "gemini-2.5-pro"

# LLM Provider Selection ("openai" or "gemini")
LLM_PROVIDER = "gemini"

# Model mapping between providers
MODEL_MAPPING = {
    "openai_to_gemini": {
        "gpt-4o": "gemini-2.5-pro",
        "gpt-4o-mini": "gemini-2.5-flash",
        "gpt-4": "gemini-2.5-pro",
        "gpt-3.5-turbo": "gemini-2.5-flash",
    },
    "gemini_to_openai": {
        "gemini-2.5-pro": "gpt-4o",
        "gemini-2.5-flash": "gpt-4o-mini",
        "gemini-pro": "gpt-4o",
        "gemini-flash": "gpt-4o-mini",
    }
}

def get_equivalent_model(model: str, target_provider: str) -> str:
    """Get equivalent model for the target provider."""
    if target_provider == "gemini":
        return MODEL_MAPPING["openai_to_gemini"].get(model, DEFAULT_GEMINI_MODEL)
    else:
        return MODEL_MAPPING["gemini_to_openai"].get(model, DEFAULT_CHATAGENT_MODEL)

LOCAL_URL = "LOCAL_URL"
LOCAL_LLM = "LOCAL_LLM"
DEFAULT_EMBED_LOCAL_MODEL = "DEFAULT_EMBED_LOCAL_MODEL"

## for embedding model
DEFAULT_EMBED_ONLINE_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_REMOTE_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_TOKEN = "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"
SPLITTER_WINDOW_SIZE = 6
SPLITTER_CHUNK_SIZE = 2048

## for preprocessing
CRAWLER_BASE_URL = ""
CRAWLER_GOOGLE_SCHOLAR_SEND_TASK_URL = ""
DEFAULT_DATA_FETCHER_ENABLE_CACHE = True
CUT_WORD_LENGTH = 10
MD_TEXT_LENGTH = 20000
ARXIV_PROJECTION = (
    "_id, title, authors, detail_url, abstract, md_text, reference, detail_id, image"
)

## Iteration and paper pool limits
DEFAULT_ITERATION_LIMIT = 3
DEFAULT_PAPER_POOL_LIMIT = 1024

## llamaindex OpenAI
DEFAULT_LLAMAINDEX_OPENAI_MODEL = "gpt-4o"
# DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
CHAT_AGENT_WORKERS = 4

## survey generation
COARSE_GRAINED_TOPK = 200
MIN_FILTERED_LIMIT = 150
NUM_PROCESS_LIMIT = 10

## fig retrieving
FIG_RETRIEVE_URL = ""
ENHANCED_FIG_RETRIEVE_URL = ""
FIG_CHUNK_SIZE = 8192
MATCH_TOPK = 3
FIG_RETRIEVE_Authorization = ""
FIG_RETRIEVE_TOKEN = ""
