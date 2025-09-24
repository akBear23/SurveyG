"""
Paper search via Semantic Scholar API and other sources for hierarchical knowledge graph construction.
"""
import requests
from typing import List, Dict, Any

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def search_papers(topic: str, limit: int = 60) -> List[Dict[str, Any]]:
    """Search papers using Semantic Scholar API and return metadata for graph construction."""
    params = {
        "query": topic,
        "limit": limit,
        "fields": "title,authors,abstract,year,venue,fieldsOfStudy,citationCount,influentialCitations"
    }
    response = requests.get(SEMANTIC_SCHOLAR_API_URL, params=params)
    response.raise_for_status()
    data = response.json()
    papers = []
    for paper in data.get("data", []):
        # Normalize fields for graph construction
        papers.append({
            "id": paper.get("paperId"),
            "title": paper.get("title", ""),
            "authors": [a.get("name", "") for a in paper.get("authors", [])],
            "abstract": paper.get("abstract", ""),
            "year": paper.get("year", 0),
            "venue": paper.get("venue", ""),
            "fieldsOfStudy": paper.get("fieldsOfStudy", []),
            "citationCount": paper.get("citationCount", 0),
            "influentialCitations": paper.get("influentialCitations", 0),
            # Optionally add more fields as needed
        })
    return papers
