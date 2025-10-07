import sys 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (containing 'writing/') to sys.path
sys.path.insert(0, os.path.join(current_dir, '../'))
import json

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
# from tqdm import tqdm
import json
from pathlib import Path
from typing import Union

import requests
import google.generativeai as genai
from tqdm import tqdm

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent


from src.models.LLM.ChatAgent import ChatAgent


# TOPICS = """TRANSFORMERS IN VISION
# A SURVEY ON IN-CONTEXT LEARNING
# FEDERATED LEARNING IN MOBILE EDGE NETWORKS
# SURVEY OF HALLUCINATION IN NATURAL LANGUAGE GENERATION
# DEEP LEARNING FOR DEEPFAKES CREATION AND DETECTION
# EFFICIENT TRANSFORMERS
# A SURVEY ON ADVERSARIAL RECOMMENDER SYSTEMS
# THE RISE AND POTENTIAL OF LARGE LANGUAGE MODEL BASED AGENTS
# A COMPREHENSIVE SURVEY ON GRAPH NEURAL NETWORKS
# DEEP LEARNING FOR IMAGE SUPER-RESOLUTION
# """.splitlines()
TOPICS = ["knowledge graph embedding"]
svx_path = Path(f"{BASE_DIR}/data/svx")
# print(remote_chat("hello"))

def load_file_as_string(path: Union[str, Path]) -> str:
    if isinstance(path, str):
        with open(path, "r", encoding="utf-8") as fr:
            return fr.read()
    elif isinstance(path, Path):
        with path.open("r", encoding="utf-8") as fr:
            return fr.read()
    else:
        raise ValueError(path)
def extract_citation_abstracts(json_data):
    """
    Trích xuất citation_key và abstract từ dữ liệu JSON
    
    Args:
        json_data: Có thể là string JSON, dict, hoặc list
    
    Returns:
        dict: {citation_key: abstract}
    """
    result = {}
    
    # Nếu input là string JSON, parse nó
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            print(f"Lỗi parse JSON: {e}")
            return result
    else:
        data = json_data
    
    # Nếu data là list (như trong trường hợp của bạn)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                citation_key = item.get('citation_key')
                abstract = item.get('metadata', {}).get('abstract', '')
                if abstract == '':
                    abstract = item.get('metadata', {}).get('title', '')
                if citation_key:
                    result[citation_key] = abstract
                    
    # Nếu data là dict đơn lẻ
    elif isinstance(data, dict):
        citation_key = data.get('citation_key')
        abstract = data.get('metadata', {}).get('abstract', '')
        if abstract == '':
            abstract = item.get('metadata', {}).get('title', '')
        if citation_key:
            result[citation_key] = abstract
    
    return result

def extract_from_file(file_path):
    """
    Đọc file JSON và trích xuất citation_key và abstract
    
    Args:
        file_path: Đường dẫn tới file JSON
    
    Returns:
        dict: {citation_key: abstract}
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return extract_citation_abstracts(data)
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        return {}
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return {}


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def nli(claim: str, source: str):
    chat_agent = ChatAgent()
    # genai.configure(api_key="AIzaSyCL-Ew4orxHGjglw7g0LjBhGsHQVPXPbro")
    # model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = """---
Claim:
{claim}
---
Source: 
{source}
---
Claim:
{claim}
---
Is the Claim faithful to the Source? 
A Claim is faithful to the Source if the core part in the Claim can be supported by the Source.\n
Only reply with 'Yes' or 'No':""".format(
        claim=claim, source=source
    )
    res = chat_agent.gemini_chat(prompt, temperature=0.3, model="gemini-2.5-pro")
    if "no" in res.lower():
        return False
    else:
        return True

def extract_cite_and_text(text: str) -> list[list[str], str]:
    cites = re.findall(r"\\cite\{([^}]+)\}", text)
    cite_list = []
    for cite in cites:
        for single_cite in cite.split(","):
            cite_list.append(single_cite.strip())
    cleaned_text = re.sub(r"\\cite\{[^}]+\}", "", text)

    return cite_list, cleaned_text.strip()


def parse_a_paper(paper_path: Path, bibname2abs: dict) -> dict:
    content = load_file_as_string(paper_path)
    claim2source = {}
    sentences = re.split(r"[.\n]+", content)
    for sentence in sentences:
        if r"\cite{" in sentence:
            sources, claim = extract_cite_and_text(sentence)
            # print(len(bibname2abs.keys()))
            # abss = [bibname2abs[source] for source in sources]
            abss = []
            for source in sources:
                try: 
                    abss.extend(bibname2abs[source])
                except:
                    pass
            claim2source[claim] = abss
    return claim2source


if __name__ == "__main__":
    # get claim and sources per paper
    res_per_paper = []
    for topic in tqdm(TOPICS):
        # ref_dir = Path(f"{BASE_DIR}/data/ref/{topic}")
        file_path = f"paper_data/{topic.replace(' ', '_')}/literature_review_output/literature_review.tex"
        mainbody_path = Path(file_path)
        bibname2abs = extract_from_file(f"paper_data/{topic.replace(' ', '_')}/keywords/processed_checkpoint.json")
        
        claim2source = parse_a_paper(mainbody_path, bibname2abs)

        claim_TF = {}
        times = 0
        for claim, sources in claim2source.items():
            source_TF = []
            for source in sources:
                times += 1
                source_TF.append(nli(claim, source))
            claim_TF[claim] = source_TF
        print(times)
        # res_per_paper.append([claim2source, claim_TF, mainbody_path])

    # calculate recall and precision
    recall_l = []
    precision_l = []
    for claim2source, claim_TF, path in res_per_paper:
        supported_claim = 0
        claim_num = len(claim2source)
        claim_source_pair_num = 0
        claim_source_pair_supported_num = 0

        for claim in claim2source.keys():
            source = claim2source[claim]
            tf = claim_TF[claim]
            t_count = tf.count(True)

            if t_count > 0:
                supported_claim += 1
            claim_source_pair_num += len(tf)
            if t_count == 0:
                claim_source_pair_supported_num += len(tf)
            else:
                claim_source_pair_supported_num += t_count

        eval_dict = {
            "claim_num": claim_num,
            "supported_claim": supported_claim,
            "source_num": claim_source_pair_num,
            "supported_source_num": claim_source_pair_supported_num,
        }

        recall = eval_dict["supported_claim"] / eval_dict["claim_num"]
        precision = eval_dict["supported_source_num"] / eval_dict["source_num"]
        recall_l.append(recall)
        precision_l.append(precision)
    print("recall: ", np.mean(recall_l))
    print("precision: ", np.mean(precision_l))