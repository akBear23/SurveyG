import json
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys 

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_abstract(abstract):
    if not abstract:
        return ""
    return re.sub(r'\s+', ' ', abstract.strip())

def detect_experiment_mention(abstract):
    experiment_keywords = [
        r'\bexperiment(s|al)?\b', r'\bevaluation\b', r'\bresults?\b',
        r'\btest(s|ing)?\b', r'\bvalidat(e|ion)\b', r'\bperformanc(e|es)\b',
        r'\bimplementation\b', r'\bdemonstrat(e|ion|ed)\b'
    ]
    if not abstract:
        return False
    pattern = '|'.join(experiment_keywords)
    return bool(re.search(pattern, abstract.lower()))

def create_paper_graph(crawl_papers_path, cited_papers_path):
    cited_papers = load_json_data(cited_papers_path)
    crawl_papers = load_json_data(crawl_papers_path)
    survey_papers = crawl_papers + cited_papers

    layers = assign_layers(survey_papers, K=20)
    
    paper_abstracts = {}
    for paper in survey_papers:
        paper_id = paper.get('id', None)
        paper_abstracts[paper_id] = preprocess_abstract(paper.get('abstract', ''))
    
    valid_papers = {pid: abstract for pid, abstract in paper_abstracts.items() if abstract.strip()}
    paper_ids = list(valid_papers.keys())
    abstracts = [valid_papers[pid] for pid in paper_ids]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(abstracts)
    G = nx.DiGraph()
    for paper in survey_papers:
        pid = paper.get('id', None)
        G.add_node(
            pid,
            title=paper.get('title', ''),
            abstract=paper.get('abstract', ''),
            authors=paper.get('authors', []),
            year=paper.get('year', ''),
            citation_count=paper.get('citationCount', 0),
            layer=layers.get(pid, 3),
            new_direction=paper.get('new_direction', 0),
            url=paper.get('url', ''), 
            pdf_link=paper.get('pdf_link', ''),
            venue=paper.get('venue', ''), 
            paper_type=paper.get('paper_type', ''),
            summary=paper.get('summary', ''),
            keywords=paper.get('keywords', []),
            is_new_direction = paper.get('is_new_direction', '0')
        )
    for paper in crawl_papers:
        paper_id = paper['id']
        if 'cited_by' not in paper:
            continue
        for cited in paper['cited_by']:
            if cited['paperId'] is None:
                continue
            cited_id = cited['paperId']
            if paper_id not in valid_papers or cited_id not in valid_papers:
                continue
            try:
                idx1 = paper_ids.index(paper_id)
                idx2 = paper_ids.index(cited_id)
                similarity = cosine_similarity(tfidf_matrix[idx1:idx1+1], tfidf_matrix[idx2:idx2+1])[0][0]
                experiment_bonus = 0.2 if detect_experiment_mention(valid_papers[cited_id]) else 0
                G.add_edge(paper_id, cited_id, weight=float(similarity + experiment_bonus))
            except ValueError:
                continue
    return G

def save_graph(G, output_path):
    # Save graph as JSON
    graph_data = {
        "nodes": [
            {"id": n, **G.nodes[n]} for n in G.nodes()
        ],
        "edges": [
            {"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges()
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f"Graph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average in-degree: {sum(dict(G.in_degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Average out-degree: {sum(dict(G.out_degree()).values()) / G.number_of_nodes():.2f}")

def assign_layers(survey_papers, K=20):
    foundation_scores = []
    for paper in survey_papers:
        paper_id = paper.get('id', None)
        year = paper.get('year', 2025)
        score = paper.get('score', 0)
        foundation_scores.append((score, paper_id))
    # Get top K papers for foundation layer
    foundation_scores.sort(reverse=True)
    foundation_ids = set([pid for _, pid in foundation_scores[:K]])
    # Layer assignment
    layers = {}
    for paper in survey_papers:
        year = paper.get('year', 2025)
        pid = paper.get('id', None)
        if pid in foundation_ids:
            layers[pid] = 1
        elif year >= 2024:
            layers[pid] = 3
        else:
            layers[pid] = 2
    return layers

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/pdf_downloader.py \"your research query\"")
        print("Example: python scripts/pdf_downloader.py \"federated learning privacy\"")
        return
    query = sys.argv[1]
    save_dir = f"paper_data/{query.replace(' ', '_')}/info"
    metadata_path = f"{save_dir}/metadata.json"
    crawl_papers_path = f"{save_dir}/crawl_papers.json"
    cited_papers_path = f"{save_dir}/cited_papers.json"
    output_path = f"{save_dir}/paper_citation_graph.json"
    try:
        G = create_paper_graph(crawl_papers_path, cited_papers_path)
        save_graph(G, output_path)
        print(f"Graph saved successfully to {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()