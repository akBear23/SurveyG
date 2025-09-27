import sys 
from pdf_downloader import download_paper
import json 
import os 



def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/summarize.py \"your research query\"")
        print("Example: python scripts/summarize.py \"federated learning privacy\"")
        return
    query = sys.argv[1]
    save_dir = f'paper_data/{query.replace(' ', '_')}'
    save_dir_core = f'paper_data/{query.replace(' ', '_')}/core_papers'
    summaries = f"paper_data/{query.replace(' ', '_')}/info/summaries.json"

    query = sys.argv[1]
    crawl_paper_json_path = f"paper_data/{query.replace(' ', '_')}/info/crawl_papers.json"
    cited_paper_json_path = f"paper_data/{query.replace(' ', '_')}/info/cited_papers.json"
    all_papers = []
    with open(crawl_paper_json_path, 'r') as f:
        crawl_papers = json.load(f)
    with open(cited_paper_json_path, 'r') as f:
        cited_papers = json.load(f)
    all_papers.extend(crawl_papers)
    all_papers.extend(cited_papers)
    
    for paper in all_papers:
        id = paper.get('id', '')
        save_path = os.path.join(save_dir, f"{id}.pdf")
        if os.path.exists(save_path):
            print(f"PDF for paper ID {id} already exists, skipping download.")
            continue
        download_paper(save_path, paper.get('pdf_link'))



if __name__ == "__main__":
    main()