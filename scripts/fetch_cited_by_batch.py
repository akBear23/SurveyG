import sys 
import os 
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import requests
import json
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

load_dotenv(Path(".env"))
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY") 
SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "paperId,title,authors,year,citationCount,abstract,url,venue,publicationDate,externalIds,openAccessPdf"
PRESTIGIOUS_VENUES = ['nature', 'science', 'cell', 'neurips', 'icml', 'iclr', 'aaai', 'ijcai', 'acl', 'emnlp']
def load_cited_by_paper_ids(json_path):
    main_paper_id = load_main_paper_id(json_path)
    with open(json_path, 'r') as f:
        papers = json.load(f)
    # Support both dict and list formats
    if isinstance(papers, list) and len(papers) > 0:
        cited_by = []
        for paper in papers:
            if 'cited_by' in paper:
                cited_by.extend(paper['cited_by'])
    else:
        cited_by = []
    # Only keep those with paperId
    return list({p['paperId'] for p in cited_by if 'paperId' in p and p['paperId'] not in main_paper_id})

def load_main_paper_id(json_path):
    with open(json_path, 'r') as f:
        papers = json.load(f)
    if isinstance(papers, list) and len(papers) > 0:
        return papers, list({p['id'] for p in papers if 'id' in p})
    return None

def fetch_batch_info(paper_ids):
    if not paper_ids:
        print("No paper IDs found in cited_by.")
        return []
    payload = {
        "ids": paper_ids
    }
    params = {
        "fields": FIELDS
    }
    response = requests.post(SEMANTIC_SCHOLAR_BATCH_URL, json=payload, params=params, timeout=60)
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        print(f"Batch API error: {response.status_code}")
        print(response.text)
        return []


def fetch_batch_info_batched(paper_ids, batch_size=500):
    all_results = []
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i+batch_size]
        print(f"Fetching batch {i//batch_size+1} ({len(batch)} papers)...")
        # Use the working query format for Semantic Scholar
        headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
        response = requests.post(
            SEMANTIC_SCHOLAR_BATCH_URL,
            params={'fields': FIELDS},
            json={"ids": batch},
            timeout=120, 
            headers=headers
        )

        if response.status_code == 200:
            resp_json = response.json()
            if isinstance(resp_json, dict) and 'data' in resp_json:
                all_results.extend(resp_json['data'])
            elif isinstance(resp_json, list):
                all_results.extend(resp_json)
            else:
                print("Unexpected response format.")
        else:
            print(f"Batch API error: {response.status_code}")
            print(response.text)
    return all_results


def extract_intro_method(text):
    """Extract introduction and methodology sections from text."""
    import re
    # Lowercase for section matching
    text_lower = text.lower()
    # Find introduction
    intro_start = text_lower.find('introduction')
    method_start = text_lower.find('methodology')
    method_alt_start = text_lower.find('methods')
    # Find next section after introduction
    next_section = re.search(r'\n[a-z ]{3,20}\n', text_lower[intro_start+12:])
    intro_end = intro_start + 12 + next_section.start() if next_section else None
    # Find next section after methodology/methods
    if method_start != -1:
        method_end_search = re.search(r'\n[a-z ]{3,20}\n', text_lower[method_start+10:])
        method_end = method_start + 10 + method_end_search.start() if method_end_search else None
        method_section = text[method_start:method_end] if method_end else text[method_start:]
    elif method_alt_start != -1:
        method_end_search = re.search(r'\n[a-z ]{3,20}\n', text_lower[method_alt_start+7:])
        method_end = method_alt_start + 7 + method_end_search.start() if method_end_search else None
        method_section = text[method_alt_start:method_end] if method_end else text[method_alt_start:]
    else:
        method_section = ''
    intro_section = text[intro_start:intro_end] if intro_start != -1 and intro_end else text[intro_start:] if intro_start != -1 else ''
    return intro_section.strip() + '\n' + method_section.strip()

def get_pdf_link(paper):
    pdf_url = None
    # Try arXiv if available
    arxiv_id = None
    external_ids = paper.get('externalIds', {})
    if isinstance(external_ids, dict):
        arxiv_id = external_ids.get('ArXiv')
    if arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # Try DOI (not always direct PDF)
    doi = external_ids.get('DOI') if isinstance(external_ids, dict) else None
    if doi and not pdf_url:
        pdf_url = f"https://doi.org/{doi}"
    return pdf_url

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/fetch_cited_by_batch.py \"your research query\"")
        print("Example: python scripts/fetch_cited_by_batch.py \"federated learning privacy\"")
        return
    
    query = sys.argv[1]
    json_path = f"paper_data/{query.replace(' ', '_').replace(':', '')}/info/crawl_papers.json"

    crawl_papers, main_paper_ids = load_main_paper_id(json_path)
    print(f"Found {len(main_paper_ids)} main paper IDs.")
    # if paper in papers already has externalIds, skip
    if crawl_papers and 'externalIds' in crawl_papers[0]:
        print("Papers already have externalIds, skipping fetch.")
    else:
        print("Fetching batch info for main papers...")
        batch_info = fetch_batch_info_batched(main_paper_ids)
        print(f"Fetched info for {len(batch_info)} main papers.")
        for paper in crawl_papers:
            pid = paper.get('id')
            for info in batch_info:
                if info.get('paperId') == pid:
                    paper['externalIds'] = info.get('externalIds', {})
                    paper['pdf_link'] = get_pdf_link(info)
        with open(json_path, 'w') as f:
            json.dump(crawl_papers, f, indent=2)
    
    paper_ids = load_cited_by_paper_ids(json_path)
    print(f"Found {len(paper_ids)} cited_by paper IDs.")
    batch_info = fetch_batch_info_batched(paper_ids)
    print(f"Fetched info for {len(batch_info)} papers.")
    cited_papers = []
    
    for paper in batch_info:
        if paper is None:
            continue
        citation_count = paper.get('citationCount', None)
        abstract = paper.get('abstract', '')
        title = paper.get('title', '')
        url = paper.get('url', '')
        venue = paper.get('venue', '')
        year = paper.get('year', 0)
        if year == None:
            continue
        if any(v in venue for v in PRESTIGIOUS_VENUES):
            in_prestigious_venues = True
        else: in_prestigious_venues = False
        if int(year) < 2025 and in_prestigious_venues == False:
            continue
        pdf_link = get_pdf_link(paper)
        if abstract is not None:
            cited_papers.append({
                'id': paper.get('paperId', ''),
                'title': title,
                'authors': [author.get('name', '') for author in paper.get('authors', [])],
                'year': year,
                'citationCount': citation_count,
                'abstract': abstract,
                'url': url,
                'pdf_link': pdf_link,
                'venue': paper.get('venue', ''),
                'publicationDate': paper.get('publicationDate', ''),
                'paper_type': paper.get('paper_type', ''),
                'externalIds': paper.get('externalIds', {}),
                'openAccessPdf': paper.get('openAccessPdf', {})
            })
    for paper in cited_papers:
        # Calculate score
        year = int(paper.get('year', 2025))
        citations = paper.get('citationCount', 0)
        score = citations * (1 / max(1, (2025 - year)))
        paper['score'] = score
    papers = crawl_papers + cited_papers
    papers.sort(key=lambda x: x.get('score', 0), reverse=True)
    selected_papers = []
    if len(papers) > 120:
        # get papers after 2025 and big venues 
        before_2025_big_venues_papers = [paper for paper in papers if paper.get('year', 0) < 2025 and paper.get('venue', '') in PRESTIGIOUS_VENUES]
        other_papers = [paper for paper in papers if paper not in before_2025_big_venues_papers]
        before_2025_papers = [paper for paper in other_papers if paper.get('year', 0) < 2025]
        after_2025_papers = [paper for paper in other_papers if paper.get('year', 0) >= 2025]
        if len(before_2025_big_venues_papers) <= 120: 
            selected_papers.extend(before_2025_big_venues_papers)
            x = (120 - len(before_2025_big_venues_papers)) / (len(before_2025_papers) + len(after_2025_papers))
            selected_papers.extend(before_2025_papers[:int(x * len(before_2025_papers))])
            selected_papers.extend(after_2025_papers[:int(x * len(after_2025_papers))])
        else:
            x = 120 / (len(before_2025_papers) + len(after_2025_papers))     
            selected_papers.extend(after_2025_papers[:int(x*len(after_2025_papers))])
            selected_papers.extend(before_2025_big_venues_papers[:120-len(selected_papers)])                                                                                                                                                                                                                                                                                                                                                                                                                                          
        # papers = papers[:max(1, int(len(papers) * 0.2))]
    else:
        selected_papers = papers
    # Save results
    output_file = f"paper_data/{query.replace(' ', '_').replace(':', '')}/info/selected_papers.json"
    with open(output_file, "w") as f:
        json.dump(selected_papers, f, indent=2)
    print(f"Saved processed info to {output_file}")
    print(f" Collected {len(selected_papers)} cited papers ready for survey generation")

if __name__ == "__main__":
    main()
