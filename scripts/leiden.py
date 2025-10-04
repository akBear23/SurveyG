import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import igraph as ig
from networkx.algorithms.community import quality
import re
import sys
import os

from numpy import sqrt 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import json
import networkx as nx
from src.models.LLM.ChatAgent import ChatAgent

import igraph as ig
import networkx as nx
import json
from collections import Counter

import igraph as ig
import networkx as nx
import json
from collections import Counter

class Leiden_summarizer:
    def __init__(self, graph_path):
        self.G = self.load_graph(graph_path)
        self.g_ig = self.nx_to_ig(self.G)

    def load_graph(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        G = nx.DiGraph()  # Load as directed graph
        for node in data['nodes']:
            G.add_node(node['id'], **node)
        for edge in data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge)
        return G

    def nx_to_ig(self, nx_graph):
        """Convert NetworkX graph to igraph"""
        # Get edges and nodes
        edges = list(nx_graph.edges())
        nodes = list(nx_graph.nodes())
        
        # Create mapping from node ID to index
        node_mapping = {node: idx for idx, node in enumerate(nodes)}
        
        # Create igraph as undirected for Leiden algorithm
        g_ig = ig.Graph(directed=False)  # Changed to undirected
        g_ig.add_vertices(len(nodes))
        
        # Store original node IDs as 'name' attribute
        g_ig.vs["name"] = nodes
        
        # Add node attributes
        for node, idx in node_mapping.items():
            for attr_name, attr_value in nx_graph.nodes[node].items():
                if attr_name != 'id':  # Skip 'id' since we already stored it as 'name'
                    g_ig.vs[idx][attr_name] = attr_value
        
        # Convert edges to use integer indices
        indexed_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges]
        g_ig.add_edges(indexed_edges)
        
        # Add edge attributes
        for u, v, data in nx_graph.edges(data=True):
            # Since we're making it undirected, we need to check if edge exists
            try:
                idx = g_ig.get_eid(node_mapping[u], node_mapping[v])
                for attr_name, attr_value in data.items():
                    if attr_name not in ('source', 'target'):  # Skip source/target as they're already in the edge
                        g_ig.es[idx][attr_name] = attr_value
            except igraph.InternalError:
                # Edge might not exist if it was a duplicate in undirected graph
                pass
        
        return g_ig
    
    def leiden_algorithm(self, layer=None):
        """Run Leiden algorithm, optionally on a specific layer
        
        Args:
            layer: If specified, only process nodes with this layer value
        """
        if layer is not None:
            # Get node IDs from NetworkX graph that have the specified layer
            layer_node_ids = [node_id for node_id, attrs in self.G.nodes(data=True) 
                             if attrs.get('layer') == layer]
            
            if not layer_node_ids:
                print(f"No nodes found with layer '{layer}'")
                return
            
            # Map node IDs to igraph vertex indices using the 'name' attribute
            # Create a mapping from node ID to index in igraph
            name_to_index = {name: idx for idx, name in enumerate(self.g_ig.vs["name"])}
            
            # Get igraph indices for the layer nodes
            layer_indices = [name_to_index[node_id] for node_id in layer_node_ids 
                            if node_id in name_to_index]
            
            if not layer_indices:
                print(f"No matching nodes found in igraph graph for layer '{layer}'")
                return
            
            # Create subgraph with only nodes from this layer
            subgraph = self.g_ig.subgraph(layer_indices)
            print(f"Running Leiden on layer '{layer}' with {len(layer_indices)} nodes")
        else:
            subgraph = self.g_ig
            print("Running Leiden on entire graph")
        
        # Run Leiden algorithm
        leiden_communities = subgraph.community_leiden(
            objective_function='modularity',
            resolution=1.0,
            n_iterations=5,
            beta=0.01
        )

        # Get community membership
        leiden_membership = leiden_communities.membership

        # Create a mapping from igraph vertex index to original node ID
        node_mapping = {idx: node_id for idx, node_id in enumerate(subgraph.vs["name"])}
        
        # Add community labels to NetworkX graph
        for idx, community_id in enumerate(leiden_membership):
            node_id = node_mapping[idx]
            self.G.nodes[node_id]['community'] = community_id

        # Calculate modularity score
        modularity_score = leiden_communities.modularity
        print(f"Leiden algorithm found {len(set(leiden_membership))} communities")
        print(f"Modularity score: {modularity_score:.4f}")

        # Count nodes per community
        community_sizes = Counter(leiden_membership)
        print("\nCommunity sizes:")
        for comm_id, size in sorted(community_sizes.items()):
            print(f"Community {comm_id}: {size} nodes")
        
        # Return communities
        communities = []
        for comm_id in sorted(set(leiden_membership)):
            # Get all node indices in this community
            node_indices = [idx for idx, cid in enumerate(leiden_membership) if cid == comm_id]
            # Convert indices to node IDs
            community_nodes = [node_mapping[idx] for idx in node_indices]
            communities.append(community_nodes)
        
        return communities
        
    
def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/traversal.py \"your research query\"")
        print("Example: python scripts/traversal.py \"federated learning privacy\"")
        return
    query = sys.argv[1]
    info_dir = f"paper_data/{query.replace(' ', '_').replace(':', '')}/info"
    graph_path = f"{info_dir}/paper_citation_graph.json"
    ls = Leiden_summarizer(graph_path)
    # ls.leiden_algorithm(layer=2)
    # ls.leiden_algorithm(layer=3)
    community = ls.leiden_algorithm()
    print(community[0])
    print(len(community))
if __name__ == "__main__":
    main()
