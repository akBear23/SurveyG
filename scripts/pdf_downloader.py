import requests
from bs4 import BeautifulSoup
import os
import re
import urllib.parse
import sys 
import json 
from selenium_crawler import download_with_selenium
def _download_pdf_content(url, file_path, headers):
    """
    Internal helper function to download a PDF file from a given URL.
    It checks for a successful response and streams the content to a file.
    """

    try:    
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # if 'application/pdf' not in response.headers.get('content-type', ''):
        #     print(f"Error: The URL does not point to a PDF. Content-Type is {response.headers.get('content-type')}")
        #     return False

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {file_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF from {url}: {e}")
        return False

def download_paper(filename, url):
    """
    Downloads a PDF paper from either a direct URL or a DOI link.

    Args:
        url (str): The URL of the paper, which can be a direct PDF link or a DOI.
        filename (str, optional): The name to save the PDF file as. 
                                  If None, a name is generated from the URL.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', 
        'Referer': 'https://www.google.com/',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    
    # Generate a default filename if none is provided
    # if filename is None:
    #     filename = os.path.basename(url)
    #     if not filename or not filename.endswith('.pdf'):
    #         filename = re.sub(r'[^a-zA-Z0-9\s_.-]', '', url).replace('httpsdoi.org', '').replace('http://', '').replace('/', '_').strip('_') + '.pdf'
    if url is None: 
        return
    try:
        if url.lower().endswith('.pdf') or 'arxiv.org/pdf/' in url.lower():
            _download_pdf_content(url, filename, headers)
            print(f"Downloaded directly from PDF link {filename}.")
            return
        # Step 1: Resolve the URL to its final destination
        print(f"Resolving URL: {url}...")
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=10)
        response.raise_for_status()
        final_url = response.url
        print(f"URL resolved to: {final_url}")
    
        # Step 3: If not a direct PDF link, scrape the page for a PDF download link (DOI case).
        # print("Final URL is not a direct PDF. Scraping for a PDF link...")
        if "mdpi" in final_url:
            print("Trying to download with selenium crawler")
            download_with_selenium(final_url, filename)
            return
            
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_link_element = None

        # Robustly search for common PDF link patterns
        if not pdf_link_element:
            pdf_link_element = soup.find('a', {'class': 'obj_galley_link pdf'})  # Common for OJS journals
        if not pdf_link_element:
            pdf_link_element = soup.find('a', string=re.compile(r'PDF', re.I))  # Find by case-insensitive text "PDF"
        if not pdf_link_element:
            pdf_link_element = soup.find('a', href=re.compile(r'\.pdf$', re.I))  # Find by href ending in .pdf
        
        if "ieeexplore.ieee.org" in final_url:
            pdf_link_element = soup.find('a', {'class': 'document-access-icon-pdf'})
            if pdf_link_element:
                pdf_url_path = pdf_link_element.get('href')
                if pdf_url_path:
                    # Construct the full PDF URL
                    pdf_url = urllib.parse.urljoin(final_url, pdf_url_path)
                    print(f"Found IEEE PDF URL: {pdf_url}")
                    _download_pdf_content(pdf_url, filename, headers)
                    return
                
        if not pdf_link_element or 'href' not in pdf_link_element.attrs:
            print(f"Could not find a PDF link on the page for {final_url}.")
            return

        pdf_link = pdf_link_element['href']

        # Step 4: Handle relative URLs by joining them with the base URL
        if not pdf_link.startswith('http'):
            pdf_url = requests.compat.urljoin(final_url, pdf_link)
        else:
            pdf_url = pdf_link
        
        print(f"Found PDF URL: {pdf_url}")
        
        # Step 5: Download the PDF from the found URL
        _download_pdf_content(pdf_url, filename, headers)

    except requests.exceptions.RequestException as e:
        print(f"Error resolving URL or downloading PDF: {e}")

# # Example usage
# if __name__ == "__main__":
#     # Example 1: Direct PDF link
#     direct_pdf_url = "https://arxiv.org/pdf/1811.01399.pdf"
#     download_paper('Graph_Attention_Networks.pdf', direct_pdf_url)

#     print("\n" + "="*50 + "\n")

#     # Example 2: DOI link resolving to a journal landing page
#     doi_landing_page = "https://doi.org/10.1609/aaai.v28i1.8870"
#     download_paper('Knowledge_Graph_Embedding.pdf', doi_landing_page)



def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/pdf_downloader.py \"your research query\"")
        print("Example: python scripts/pdf_downloader.py \"federated learning privacy\"")
        return
    query = sys.argv[1]
    save_dir = f"paper_data/{query.replace(' ', '_').replace(':', '')}"
    save_dir_core = f"paper_data/{query.replace(' ', '_').replace(':', '')}/core_papers"
    metadata_path = f"paper_data/{query.replace(' ', '_').replace(':', '')}/info/metadata.json"
    metadata = {}
    query = sys.argv[1]
    # crawl_paper_json_path = f"paper_data/{query.replace(' ', '_').replace(':', '')}/info/crawl_papers.json"
    # cited_paper_json_path = f"paper_data/{query.replace(' ', '_').replace(':', '')}/info/cited_papers.json"
    # all_papers = []
    # with open(crawl_paper_json_path, 'r') as f:
    #     crawl_papers = json.load(f)
    # with open(cited_paper_json_path, 'r') as f:
    #     cited_papers = json.load(f)
    # all_papers.extend(crawl_papers)
    # all_papers.extend(cited_papers)
    selected_papers_path = f"paper_data/{query.replace(' ', '_').replace(':', '')}/info/paper_citation_graph.json"
    with open(selected_papers_path, "r") as f:
        all_papers = json.load(f)
    all_papers = all_papers['nodes']
    for paper in all_papers:
        id = paper.get('id', '')
        save_path = os.path.join(save_dir, f"{id}.pdf")
        if os.path.exists(save_path):
            print(f"PDF for paper ID {id} already exists, skipping download.")
        else: download_paper(save_path, paper.get('pdf_link'))
        filename = f"{id}.pdf"
        paper_metadata = {
            "title": paper.get('title'),
            "authors": paper.get('authors', []),
            "published_date": str(paper.get('year', '')),
            "abstract": paper.get('abstract'),
            "file_path": save_path, 
            "venue": paper.get('venue', ''),
            "citationCount": paper.get('citationCount', 0), 
            "score": paper.get('score', 0)
        }
        metadata[filename] = paper_metadata

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved {len(metadata)} papers to {metadata_path}")
if __name__ == "__main__":
    main()