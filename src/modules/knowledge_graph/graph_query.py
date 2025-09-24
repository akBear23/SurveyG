"""
Graph Query Module
Provides functions to query the knowledge graph for relevant papers/info.
"""
import networkx as nx
from typing import List, Dict

class GraphQueryEngine:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def query_relevant(self, topic: str, top_k: int = 10) -> List[Dict]:
        """Query the graph for the most relevant papers to a topic (simple keyword match)."""
        # Example: rank nodes by keyword match in title/abstract
        scored = []
        for node, data in self.graph.nodes(data=True):
            score = 0
            if topic.lower() in (data.get("title", "").lower()):
                score += 2
            if topic.lower() in (data.get("abstract", "").lower()):
                score += 1
            if score > 0:
                scored.append((score, data))
        scored.sort(reverse=True)
        return [d for s, d in scored[:top_k]]
