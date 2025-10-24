import sys 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '../'))
import json
import time
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import hashlib

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent

from src.models.LLM.ChatAgent import ChatAgent

TOPICS = ["A survey on Visual Transformer", "Hallucination in Large Language Models", "Graph Neural Networks" ]
CACHE_FILE = BASE_DIR / "nli_cache.json"

# ==================== CACHE SYSTEM ====================
class CacheSystem:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = self.load_cache()
    
    def load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def get_cache_key(self, *args):
        """Tạo cache key từ arguments"""
        content = '|||'.join(str(arg)[:200] for arg in args)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        self.cache[key] = value
        # Save cache periodically
        if len(self.cache) % 5 == 0:
            self.save_cache()

cache_system = CacheSystem(CACHE_FILE)

# ==================== HELPER FUNCTIONS ====================
def load_file_as_string(path):
    if isinstance(path, str):
        with open(path, "r", encoding="utf-8") as fr:
            return fr.read()
    elif isinstance(path, Path):
        with path.open("r", encoding="utf-8") as fr:
            return fr.read()
    else:
        raise ValueError(path)

def extract_citation_abstracts(json_data):
    """Trích xuất citation_key và abstract từ dữ liệu JSON"""
    result = {}
    
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            print(f"Lỗi parse JSON: {e}")
            return result
    else:
        data = json_data
    
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                citation_key = item.get('citation_key')
                infor = item.get('metadata', {}).get('summary')
                
                if citation_key and infor:
                    result[citation_key] = infor
                    
    elif isinstance(data, dict):
        citation_key = data.get('citation_key')
        abstract = data.get('metadata', {}).get('abstract')
        
        if citation_key and abstract:
            result[citation_key] = abstract
    
    return result

def extract_from_file(file_path):
    """Đọc file JSON và trích xuất citation_key và abstract"""
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

def extract_cite_and_text(text):
    """Extract citation keys and clean text from a sentence"""
    cites = re.findall(r"\\cite\{([^}]+)\}", text)
    cite_list = []
    
    for cite in cites:
        for single_cite in cite.split(","):
            cite_list.append(single_cite.strip())
    
    cleaned_text = re.sub(r"\\cite\{[^}]+\}", "", text)
    
    return cite_list, cleaned_text.strip()

def parse_a_paper(paper_path, bibname2abs):
    """Parse a LaTeX paper and extract claims with their cited sources"""
    content = load_file_as_string(paper_path)
    claim2source = {}
    
    sentences = re.split(r'\.(?:\s+|(?=\\))|(?:\n\s*){2,}', content)
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        if not sentence:
            continue
            
        if r"\cite{" in sentence:
            sources, claim = extract_cite_and_text(sentence)
            
            if sources and claim:
                try:
                    abss = [bibname2abs[source] for source in sources if source in bibname2abs]
                    
                    if abss:
                        claim2source[claim] = abss
                except KeyError as e:
                    print(f"Warning: Citation key not found: {e}")
                    continue
    
    return claim2source

# ==================== NLI FUNCTIONS WITH CACHING ====================
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30)
)
def nli(claim, source):
    """NLI với caching"""
    # Check cache first
    cache_key = cache_system.get_cache_key("nli", claim, source)
    cached_result = cache_system.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Rate limiting đơn giản
    time.sleep(1)  # Delay 1 giây giữa các request
    
    chat_agent = ChatAgent()
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
A Claim is faithful to the Source if the core part in the Claim can be supported by the Source.

Only reply with 'Yes' or 'No':""".format(claim=claim, source=source)
    
    try:
        res = chat_agent.gemini_chat(prompt, temperature=0.3, model="gemini-2.5-flash")
        result = "yes" in res.lower()
        
        # Cache result
        cache_system.set(cache_key, result)
        
        return result
    except Exception as e:
        print(f"\n[API Error in NLI]: {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print(f"Rate limit hit, waiting 30s...")
            time.sleep(30)
        raise

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30)
)
def check_relevance(target_source, other_sources, claim):
    """Kiểm tra relevance với caching"""
    if not other_sources.strip():
        return True
    
    # Check cache first
    cache_key = cache_system.get_cache_key("relevance", target_source, other_sources, claim)
    cached_result = cache_system.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Rate limiting đơn giản
    time.sleep(1)  # Delay 1 giây giữa các request
    
    chat_agent = ChatAgent()
    prompt = """---
Claim:
{claim}
---
Target Source:
{target_source}
---
Other Sources:
{other_sources}
---
Question: Does the Target Source provide NEW or ESSENTIAL information to support the Claim that is NOT already provided by the Other Sources?

In other words, is the Target Source necessary and non-redundant?

Only reply with 'Yes' or 'No':""".format(
        claim=claim,
        target_source=target_source,
        other_sources=other_sources
    )
    
    try:
        res = chat_agent.gemini_chat(prompt, temperature=0.3, model="gemini-2.5-flash")
        result = "yes" in res.lower()
        
        # Cache result
        cache_system.set(cache_key, result)
        
        return result
    except Exception as e:
        print(f"\n[API Error in Relevance]: {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print(f"Rate limit hit, waiting 30s...")
            time.sleep(30)
        raise

# ==================== MAIN (SINGLE-THREADED) ====================
if __name__ == "__main__":
    print("="*60)
    print("Citation Quality Evaluation (Single-threaded)")
    print("="*60)
    
    res_per_paper = []
    
    for topic in TOPICS:
        print(f"\n{'='*60}")
        print(f"Processing topic: {topic}")
        print(f"{'='*60}")
        
        file_path = f"paper_data/{topic.replace(' ', '_')}/literature_review_output/literature_review.tex"
        mainbody_path = Path(file_path)
        bibname2abs = extract_from_file(
            f"paper_data/{topic.replace(' ', '_')}/keywords/processed_checkpoint.json"
        )
        
        print(f"Loaded {len(bibname2abs)} references")
        
        claim2source = parse_a_paper(mainbody_path, bibname2abs)
        print(f"Found {len(claim2source)} claims with citations")
        
        if len(claim2source) == 0:
            print("No claims found, skipping...")
            continue
        
        claims = list(claim2source.keys())
        sources_list = list(claim2source.values())
        
        # ===== RECALL CALCULATION =====
        print(f"\n{'='*60}")
        print("STEP 1: Calculating RECALL")
        print(f"{'='*60}")
        
        scores = []
        
        # for i, (claim, sources) in enumerate(zip(claims, sources_list)):
            # print(f"Processing claim {i+1}/{len(claims)}: {claim[:100]}...")
        for i, (claim, sources) in enumerate(tqdm(zip(claims, sources_list), total=len(claims), desc="Processing claims")):
            # Concatenate tất cả sources
            sources_concat = '\n'.join(sources)
            
            try:
                result = nli(claim, sources_concat)
                scores.append(1 if result else 0)
                # print(f"  Result: {'Supported' if result else 'Not supported'}")
            except Exception as e:
                print(f"  Error: {e}")
                scores.append(0)
        
        supported_claims = sum(scores)
        recall = np.array(scores).mean() if scores else 0
        print(f"\n✓ Recall completed: {supported_claims}/{len(claims)} claims supported ({recall:.4f})")
        
        # ===== PRECISION CALCULATION =====
        print(f"\n{'='*60}")
        print("STEP 2: Calculating PRECISION")
        print(f"{'='*60}")
        
        citation_num = sum(len(sources) for sources in sources_list)
        print(f"Total citations to evaluate: {citation_num}")
        
        precisions = [0] * len(claims)
        total_relevant = 0
        processed_citations = 0
        
        # for j, (claim, sources) in enumerate(zip(claims, sources_list)):
        for j, (claim, sources) in enumerate(tqdm(zip(claims, sources_list), total=len(claims), desc="Processing claims")):
            # Chỉ tính precision cho claims đã được verify
            if scores[j] == 1:
                # print(f"Processing claim {j+1} with {len(sources)} citations...")
                
                # Kiểm tra từng source
                for idx, target_source in enumerate(sources):
                    processed_citations += 1
                    # print(f"  Citation {processed_citations}/{citation_num}")
                    
                    # Lấy tất cả sources KHÁC
                    other_sources = [s for k, s in enumerate(sources) if k != idx]
                    other_sources_concat = '\n'.join(other_sources)
                    
                    try:
                        result = check_relevance(target_source, other_sources_concat, claim)
                        if result:
                            precisions[j] += 1
                            total_relevant += 1
                        print(f"    Result: {'Relevant' if result else 'Not relevant'}")
                    except Exception as e:
                        print(f"    Error: {e}")
        
        precision = total_relevant / citation_num if citation_num > 0 else 0
        print(f"\n✓ Precision completed: {total_relevant}/{citation_num} citations relevant ({precision:.4f})")
        
        # Save results
        res_per_paper.append({
            'topic': topic,
            'claims': claims,
            'sources': sources_list,
            'scores': scores,
            'precisions': precisions,
            'citation_num': citation_num,
            'recall': recall,
            'precision': precision,
            'path': mainbody_path
        })
        
        # Save cache
        cache_system.save_cache()
        print(f"\n✓ Cache saved to {CACHE_FILE}")
    
    # ===== FINAL RESULTS =====
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    if len(res_per_paper) == 0:
        print("No papers processed!")
    else:
        recall_list = [r['recall'] for r in res_per_paper]
        precision_list = [r['precision'] for r in res_per_paper]
        
        print(f"\nPer-paper results:")
        for i, result in enumerate(res_per_paper, 1):
            print(f"\n{i}. {result['topic']}")
            print(f"   Recall:    {result['recall']:.4f} ({sum(result['scores'])}/{len(result['scores'])} claims)")
            print(f"   Precision: {result['precision']:.4f} ({sum(result['precisions'])}/{result['citation_num']} citations)")
        
        print(f"\n{'='*60}")
        print(f"Average Recall:    {np.mean(recall_list):.4f}")
        print(f"Average Precision: {np.mean(precision_list):.4f}")
        print(f"{'='*60}")
        
        # Save final cache
        cache_system.save_cache()
        print(f"\n✓ Final cache saved to {CACHE_FILE}")