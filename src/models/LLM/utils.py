import base64
import os
from pathlib import Path

import tiktoken

from src.configs.config import CUT_WORD_LENGTH
from src.configs.logger import get_logger

logger = get_logger("src.modules.LLM.utils")


def load_prompt(file_path: Path, **kwargs):
    """读取prompt模板"""
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            return f.read().format(**kwargs)
    else:
        logger.error(f"Prompt template not found at {file_path}")
        return ""


def num_token_from_string(text: str, model: str = "gpt-4o-mini") -> int:
    """Return token nums of a string."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_text = encoding.encode(text)
    return len(encoded_text)


def cut_text_by_token(text: str, max_tokens: int, model: str = "gpt-4o-mini"):
    """Cut text by token num."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        encoded_text = encoding.encode(text)
        cut_text = encoding.decode(encoded_text[:max_tokens])
    except Exception as e:
        logger.error(e)
        cut_text = text[: CUT_WORD_LENGTH * max_tokens]
    return cut_text


# 图片转base64函数
def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
