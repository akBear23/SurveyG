"""
Hierarchical Knowledge Graph Builder
Builds a graph from paper metadata, citations, and attributes.
"""
import networkx as nx
from typing import List, Dict

class KnowledgeGraphBuilder:
    def build_graph(self, papers: List[Dict]) -> nx.DiGraph:
        """Builds a directed graph from a list of papers."""
        G = nx.DiGraph()
        for paper in papers:
            paper_id = paper.get("paperId") or paper.get("id")
            G.add_node(paper_id, **paper)
            # Example: add citation edges if available
            for cited_id in paper.get("citations", []):
                G.add_edge(paper_id, cited_id)
        return G
