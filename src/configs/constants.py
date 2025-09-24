from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent

DATASET_DIR = Path(f"{BASE_DIR}/datasets")
RESOURCE_DIR = Path(f"{BASE_DIR}/resources")
OUTPUT_DIR = Path(f"{BASE_DIR}/outputs")
CACHE_DIR = Path(f"{BASE_DIR}/cache")

CONFIG_PATH = Path(f"{BASE_DIR}/configuration/config.json")
TMP_CONFIG_PATH = Path(f"{OUTPUT_DIR}/tmp/tmp_config.json")

AVAILABLE_DATA_SOURCES = ["google_scholar", "arxiv"]
DEFAULT_SPLITTER_TYPE = "sentence"
