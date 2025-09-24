import json
import networkx as nx

def load_graph(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    G = nx.DiGraph()
    for node in data['nodes']:
        G.add_node(node['id'], **node)
    for edge in data.get('edges', []):
        G.add_edge(edge['source'], edge['target'], **edge)
    return G

def check_layer1_connections(G):
    layer1_nodes = [n for n, attr in G.nodes(data=True) if attr.get('layer') == 1]
    connected_pairs = []
    for u in layer1_nodes:
        for v in layer1_nodes:
            if u != v and G.has_edge(u, v):
                connected_pairs.append((u, v))
    return connected_pairs

def main():
    graph_path = '../paper_citation_graph.json'
    G = load_graph(graph_path)
    connected_pairs = check_layer1_connections(G)
    if connected_pairs:
        print('Layer 1 nodes with connections:')
        for u, v in connected_pairs:
            print(f'{u} -> {v}')
    else:
        print('No layer 1 nodes are connected to each other.')

if __name__ == '__main__':
    main()