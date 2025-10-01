import re
import sys
import os

from numpy import sqrt 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import json
import networkx as nx
from src.models.LLM.ChatAgent import ChatAgent
import requests
from pdf_downloader import download_paper
# Load the graph from JSON
def load_graph(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    G = nx.DiGraph()
    for node in data['nodes']:
        G.add_node(node['id'], **node)
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'], **edge)
    return G

# Collect papers with new_direction=1 for each layer
def get_layer_seeds(G, layer):
    return [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer]

# BFS 1-hop traversal from seeds, collect info
def bfs_one_hop(G, seeds, max_hop_papers=10):
    visited = set()
    info = []
    path = []
    for seed in seeds:
        if seed not in visited:
            visited.add(seed)
            attr = G.nodes[seed]
            info.append({'title': attr.get('title', ''), 'abstract': attr.get('abstract', ''), 'year': attr.get('year', 0)})
            path.append(seed)
        # Get outgoing edges sorted by weight (descending)
        neighbors = list(G.successors(seed))
        weighted_neighbors = [(neighbor, G.edges[seed, neighbor]['weight']) for neighbor in neighbors]
        weighted_neighbors.sort(key=lambda x: x[1], reverse=True)
        # Take up to max_hop_papers highest-weight neighbors
        for neighbor, _ in weighted_neighbors[:max_hop_papers]:
            if neighbor not in visited:
                visited.add(neighbor)
                attr = G.nodes[neighbor]
                info.append({'title': attr.get('title', ''), 'abstract': attr.get('abstract', ''), 'year': attr.get('year', 0)})
                path.append(neighbor)
    return info, path

def bfs_one_hop_from_seed(G, seed, max_hop_papers=10):
    visited = set()
    info = []
    path = []
    # Add the seed itself
    if seed not in visited:
        visited.add(seed)
        attr = G.nodes[seed]
        info.append({'title': attr.get('title', ''), 'abstract': attr.get('abstract', ''), 'year': attr.get('year', 0)})
        path.append(seed)
    # Get outgoing edges sorted by weight (descending)
    neighbors = list(G.successors(seed))
    weighted_neighbors = [(neighbor, G.edges[seed, neighbor]['weight']) for neighbor in neighbors]
    weighted_neighbors.sort(key=lambda x: x[1], reverse=True)
    # Take up to max_hop_papers highest-weight neighbors
    for neighbor, _ in weighted_neighbors[:max_hop_papers]:
        if neighbor not in visited:
            visited.add(neighbor)
            attr = G.nodes[neighbor]
            info.append({'title': attr.get('title', ''), 'abstract': attr.get('abstract', ''), 'year': attr.get('year', 0)})
            path.append(neighbor)
    return info, path

# Call Gemini LLM to summarize
def summarize_with_gemini(query,paper_infos, path, layer_2_summary=''):
    # Compose a prompt for taxonomy of a direction, showing the traversal path
    # prompt += "Summarize the following papers for taxonomy of a development direction, where each paper in the path cites the previous one. Make sure to highlight the development in the topic and how each paper contributes to the direction based on their methodology and publication time.\nPaper traversal path: "
    # prompt += " -> ".join([info['title'] for info in paper_infos]) + "\n\n"
    path_text = ''
    for info in paper_infos:
        try:
            if info['summary'] != info['abstract'] and info['summary'] != '':
                path_text += f"Title: {info['title']}\nAbstract: {info['abstract']}\nSummary: {info['summary']}\nPublication Year: {info['year']}\n\n"
        except: path_text += f"Title: {info['title']}\nAbstract: {info['abstract']}\nPublication Year: {info['year']}\n\n-->"
    
    prompt = f"""
    For the research topic {query}, You are given a sequence of papers forming a citation path, where each paper cites the previous one. For each paper, describe:
    - Its main methodological contribution.
    - How it builds on the previous paper(s).
    - The role of its publication time in shaping the progression.
    Finally, organize the summary as a coherent development path, highlighting the evolution of ideas and how the sequence of works collectively contributes to the research direction.
    Paper traversal path: {path_text}
    """
    if layer_2_summary != '':
        prompt += f"Given the taxonomy of earlier papers in the research direction {layer_2_summary}\n"

    chat_agent = ChatAgent()
    try:
        summary = chat_agent.gemini_chat(prompt, temperature=0.3)
    except Exception as e:
        print(f"Error occurred while summarizing direction: {e}")
        summary = "LLM summary placeholder for direction."
    return summary

def generate_survey_outline(query, taxonomy, develop_direction, previous_outline='', improvement_suggestions=''):
    # Combine all taxonomy summaries into one prompt
    prompt = f"Create a comprehensive literature review outline for the topic {query} based on the following taxonomy summaries for three layers (foundational, development, and recent/trending) and development directions.\n\n"
    for layer in [1, 2, 3]:
        layer_desc = {1: 'foundational papers', 2: 'development papers', 3: 'recent and trending papers'}
        summary = taxonomy.get(f"{layer}", {}).get('summary', '')
        prompt += f"Layer {layer_desc.get(layer, '')} taxonomy summary:\n{summary}\n\n"
    # Add development direction info the develop_direction dict format {seed_id: {"seed_title": title, "summary": summary, "path": [list of paper ids]}}
    prompt += f"Development direction:\n"
    for seed_id, info in develop_direction.items():
        prompt += f"Seed ID: {seed_id}\n"
        prompt += f"Title: {info.get('seed_title', '')}\n"
        prompt += f"Summary: {info.get('summary', '')}\n"
        prompt += f"Path: {info.get('path', [])}\n\n"
    
    prompt += "Based on the above information, create a detailed outline for a literature review paper, organizing it into sections and subsections that reflect the major themes and developments in the field. "
    prompt += "Please organize the outline in a logical flow suitable for a literature review paper, do not need to add unncessary details like layer numbers. "
    prompt += "\nRespond with the outline in json format with keys: ['section_outline', 'section_focus', 'proof_ids']."
    prompt += "\nFor each section, add the section and its subsections titles in a hierarchical manner in the same 'section_outline'; "
    prompt += "For each section, add a paragraph to the key 'section_focus' to indicate the main focus of that section; "
    prompt += "For each section, add an id (taken from the information layer number or the development seed ids) to the key 'proof_ids' to indicate the proof for each section, if proof is from the taxonomy of layer 1, 2, or 3 put the layer number only, if the proof is from the development direction, put the seed(s) paper id."
    prompt += ".\n\nRespond with the outline in json format structure as the example below:\n"
    prompt += """ 
    {
      [
        "section_outline": "### 1. Introduction
    *   1.1. Background: Knowledge Graphs and Their Significance
    *   1.2. The Role of Knowledge Graph Embedding
    *   1.3. Scope and Organization of the Review",
        "section_focus": "This section introduces Knowledge Graphs (KGs) and the fundamental problem of Knowledge Graph Embedding (KGE). It highlights the importance of KGE for various AI tasks and outlines the scope and structure of the review.",
        "proof_ids": [1, 2]
      ], 
      ...
    }
    **IMPORTANT**: 
    - Each section and its subsections should be in the same string under the key 'section_outline'
    - Return ONLY valid JSON without any markdown formatting or code blocks
    - Escape all backslashes and quotes properly in JSON strings
    - Do not include any special characters that might break JSON parsing
    """
    if improvement_suggestions:
        prompt += f"Here is the previously generated outline: {previous_outline}\n"
        prompt += f"Here are some evaluation and improvement suggestions for the outline: {improvement_suggestions}\n"
        prompt += "Please regenerate the outline with the provided taxonomy summaries and development directions, addressing these suggestions.\n"
    # prompt += ".\n\nRespond with the outline in json format with keys: 'section_outline', 'section_focus', 'proof_ids'."

    # # prompt return markdown
    # prompt += "Based on the above information, create a detailed outline for a literature review paper, organizing it into sections and subsections that reflect the major themes and developments in the field. "
    # prompt += "Please organize the outline in a logical flow suitable for a literature review paper, do not need to add unncessary details like layer numbers. "
    # prompt += "In your output, first for each section list the section and subsection titles in a hierarchical manner under the key ** Section Outline **, "
    # prompt += "For each section, add a paragraph to the key ** Section Focus ** to indicate the main focus of that section; "
    # prompt += "For each section, add an id (taken from the information layer number or the development seed ids) to the key ** Proof IDs ** to indicate the proof for each section, if proof is from the taxonomy of layer 1, 2, or 3 put the layer number only, if the proof is from the development direction, put the seed(s) paper id."
    # prompt += ".\n\nRespond with the outline in markdown format."

    chat_agent = ChatAgent()
    try:
        outline = chat_agent.gemini_chat(prompt, temperature=0.3)
        json_string = outline.strip().removeprefix('```json\n').removesuffix('\n```')
        try:
            outline = json.loads(json_string)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Problematic string start: {json_string[:50]}...")
            with open (f"paper_data/{query.replace(' ', '_').replace(':', '')}/literature_review_output/survey_outline_v3.txt", "w", encoding="utf-8") as f:
                f.write(outline)
    except Exception as e:
        print(f"Error occurred while generating survey outline: {e}")
        outline = "LLM outline placeholder."
    return outline

def get_layer_subgraph(G, layer):
    nodes = [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer]
    return G.subgraph(nodes)

def summarize_layer_method_groups(query, G, layer, max_papers=50):
    # Get papers with new_direction=1 in this layer
    papers = [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer and attr.get('new_direction') == 1]
    if len(papers) == 0: 
        papers = [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer]
        papers = sorted(
            papers,
            key=lambda node: G.nodes[node].get('score', float('-inf')),
            reverse=True
        )[:50]
    print(len(papers))
    # If layer == 3, include highly cited papers in this layer?
    # if layer == 3:
    #     additional_papers = [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer and attr.get('citation_count', 0) >= 50]
    #     papers = list(set(papers) | set(additional_papers))
    
    # If more than max_papers, select those with highest citation count
    if len(papers) > max_papers:
        papers = sorted(papers, key=lambda n: G.nodes[n].get('citation_count', 0), reverse=True)[:max_papers]
    infos = []
    for n in papers:
        attr = G.nodes[n]
        infos.append({
            'title': attr.get('title', ''),
            'abstract': attr.get('abstract', ''),
            'summary': attr.get('summary', ''),
            'year': attr.get('year', 0)
        })
    # print(f"Layer {layer} - summarizing {len(infos)} papers for method group summary.")
    # Compose prompt for LLM
    layer_desc = {1: 'foundational papers', 2: 'development papers', 3: 'recent and trending papers'}
    prompt = f"Summarize the method groups for the survey topic {query} from the following {layer_desc.get(layer, '')} in layer {layer}. Divide the papers in the layer into subgroups based on their contributions and methodologies.\n\n"
    for info in infos:
        prompt += f"Title: {info['title']}\nAbstract: {info['abstract']}\nSummary: {info['summary']}\nPublication Year: {info['year']}\n\n"
    chat_agent = ChatAgent()
    try:
        summary = chat_agent.gemini_chat(prompt, temperature=0.3)
    except Exception as e:
        print(f"Error occurred while summarizing method groups for layer {layer}: {e}")
        summary = f"LLM summary placeholder for layer {layer}."
    return summary, papers

def save_papers_info_json(paper_ids, G, save_dir, output_path, full_metadata):
    core_metadata = {}

    # Process each paper entry
    for pid in paper_ids:
        pid = pid + '.pdf'
        try: 
            core_metadata[pid] = full_metadata[pid]
        except:
            pass

    # Write the new metadata to an output JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(core_metadata, f, indent=4)

    print(f"Saved {len(core_metadata)} papers to {output_path}")

from collections import deque

def bfs_from_seed(query, graph, seed, max_development_paper=20, max_frontier_paper=30):
    visited = set()
    info = []
    path = []
    queue = deque()
    
    # Separate papers by layer
    layer1_papers = []
    layer2_papers = []
    layer3_papers = []
    
    # Taxonomy results
    layer_2_summary = None
    path_taxonomy = None
    
    # Process layer 1 (seed) separately
    visited.add(seed)
    path.append(seed)
    
    # Get seed attributes
    attr = graph.nodes[seed]
    paper_info = {
        'title': attr.get('title', ''),
        'abstract': attr.get('abstract', ''),
        'summary': attr.get('summary', ''),
        'year': attr.get('year', 0)
    }
    info.append(paper_info)
    layer1_papers.append(paper_info)
    
    # Get neighbors for layer 1 and add to queue as layer 2
    neighbors = list(graph.successors(seed))
    neighbors = sorted(neighbors, key=lambda n: graph.edges[seed, n].get('weight', 0), reverse=True)
    
    for neighbor in neighbors:
        attr = graph.nodes[neighbor]
        layer = attr.get('layer', None)
        if neighbor not in visited and layer == 2:
            visited.add(neighbor)
            queue.append((neighbor, 2))  # All these neighbors are at layer 2
    
    # Process layers 2 and 3
    while queue:
        node, layer = queue.popleft()
        path.append(node)
        
        # Get node attributes
        attr = graph.nodes[node]
        paper_info = {
            'title': attr.get('title', ''),
            'abstract': attr.get('abstract', ''),
            'year': attr.get('year', 0)
        }
        info.append(paper_info)
        
        # Process layer 2 nodes
        if layer == 2:
            layer2_papers.append(paper_info)
            
            # Check if we should call LLM for layer 2
            if not layer_2_summary and len(layer2_papers) >= max_development_paper:
                layer_2_summary = summarize_with_gemini(query, layer1_papers + layer2_papers, path)
                continue
            
            # Get neighbors for layer 3
            neighbors = list(graph.successors(node))
            neighbors = sorted(neighbors, key=lambda n: graph.edges[node, n].get('weight', 0), reverse=True)
            
            for neighbor in neighbors:
                attr = graph.nodes[neighbor]
                layer = attr.get('layer', None)
                if neighbor not in visited and layer == 3:
                    visited.add(neighbor)
                    queue.append((neighbor, 3))  # These neighbors are at layer 3
        
        # Process layer 3 nodes
        elif layer == 3:
            layer3_papers.append(paper_info)
            
            # Check if we should to travel
            if len(layer3_papers) >= max_frontier_paper:
                break
    if not layer_2_summary: layer_2_summary = summarize_with_gemini(query, layer1_papers + layer2_papers, path)
    if len(layer3_papers): path_taxonomy = summarize_with_gemini(query,layer3_papers, path, layer_2_summary)
    else: 
        path_taxonomy = summarize_with_gemini(query, layer1_papers + layer2_papers + layer3_papers, path)
 
    return [
        info,
        path,
        layer1_papers,
        layer2_papers,
        layer3_papers,
        layer_2_summary,
        path_taxonomy
    ]

def dfs_from_seed(graph, seed, last_layer, max_paper_number=50):
    visited = set()
    info = []
    path = []

    def dfs(node):
        if len(info) >= max_paper_number:
            return
        if node in visited:
            return
        visited.add(node)
        attr = graph.nodes[node]
        info.append({
            'title': attr.get('title', ''),
            'abstract': attr.get('abstract', ''),
            'year': attr.get('year', 0)
        })
        path.append(node)
        # Stop if node is in the last layer
        if attr.get('layer', None) == last_layer:
            return
        # Explore neighbors (successors)
        neighbors = list(graph.successors(node))
        # Optionally sort neighbors by edge weight descending
        neighbors = sorted(neighbors, key=lambda n: graph.edges[node, n].get('weight', 0), reverse=True)
        if attr.get('layer', None) == 1:
            neighbors = neighbors[:int(sqrt(max_paper_number))]  # Limit to top papers for layer 1 nodes
        if attr.get('layer', None) == 2:
            neighbors = neighbors[:int(sqrt(max_paper_number))]  # Limit to top papers for layer 2 nodes
        for neighbor in neighbors:
            dfs(neighbor)

    dfs(seed)
    summary = summarize_with_gemini(info, path)
    return info, path, summary

def evaluate_outline(query, outline, save_dir, max_iterations=3):
    """
        Evaluate the quality of a literature review section based on multiple criteria
        
        Args:
            section_title (str): Title of the section
            section_content (str): Content of the section
            section_focus (str): Expected focus/scope of the section
            
        Returns:
            Dict: Evaluation results with scores and improvement suggestions
        """
    # concat section outline text and focus to a string
    outline_text = ""
    for section in outline:
        outline_text += f"{section['section_outline']}\n{section['section_focus']}"
    evaluation_prompt = f"""
    Evaluate the quality and structure of the following literature review outline for the topic {query}. Assess whether the outline demonstrates meaningful organization of works rather than a simple concatenation of summaries. Your feedback should include:
    • Strengths of the outline
    • Weaknesses or issues (if any)
    • Specific suggestions for improvement (only if issues are found).
    Outline to evaluate: {outline_text}

    """
    # **IMPORTANT**: 
    # - Return ONLY valid JSON without any markdown formatting or code blocks
    # - Escape all backslashes and quotes properly in JSON strings
    # - Do not include any special characters that might break JSON parsing
    
    # **Response Format** (JSON only):
    #     {{
    #         "overall_score": <average_score>,
    #         "detailed_scores": {{
    #             "logical_flow": <score>,
    #             "coherence": <score>,
    #             "relevance": <score>,
    #             "clarity": <score>,
    #             "content_coverage": <score>
    #             "relational_awareness": <score>
    #         }},
    #         "improvement_suggestions": "<detailed suggestions for improvement>"
    #     }}
    # evaluation_prompt = f"""
    # Evaluate the logicality, coherence, and structural quality of the following literature review outline. Focus on whether the outline demonstrates meaningful organization of works rather than a simple concatenation of summaries. Suggest improvements if issues are found.
    # Outline to evaluate:
    # {outline_text}
    # **Evaluation Criteria** (Rate each from 1-5, where 5 is excellent):
    # 1. **Logical Flow**: Does the outline follow a clear and logical progression of ideas?
    # 2. **Coherence**: Are the sections and subsections well-connected, reflecting meaningful grouping and hierarchy of ideas?
    # 3. **Relevance**: Do the sections align with the overall topic and objectives?
    # 4. **Clarity**: Is the outline easy to interpret, with clear section and subsection labels?
    # 5. **Content Coverage**: Does the outline comprehensively address the topic’s scope?
    # 6. **Relational Awareness**: Does the outline capture relationships between works (e.g., citations, methodological connections, or progression of ideas), rather than treating them as isolated?
    
    # **IMPORTANT**: 
    # - Return ONLY valid JSON without any markdown formatting or code blocks
    # - Escape all backslashes and quotes properly in JSON strings
    # - Do not include any special characters that might break JSON parsing
    
    # **Response Format** (JSON only):
    #     {{
    #         "overall_score": <average_score>,
    #         "detailed_scores": {{
    #             "logical_flow": <score>,
    #             "coherence": <score>,
    #             "relevance": <score>,
    #             "clarity": <score>,
    #             "content_coverage": <score>
    #             "relational_awareness": <score>
    #         }},
    #         "improvement_suggestions": "<detailed suggestions for improvement>"
    #     }}
    # Consider an outline satisfactory if overall_score >= 3.5 and no individual score is below 3.0.
    # """
    # call chat agent to evaluate the input outline
    chat_agent = ChatAgent()
    evaluation = chat_agent.gemini_chat(evaluation_prompt, temperature=0.3)

    # parse the output json
    # try:
    #     json_string = evaluation.strip().removeprefix('```json\n').removesuffix('\n```')
    # except Exception as e:
    #     print(f"Error processing outline: {e}")
    #     with open(f"{save_dir}/outline_to_evaluate.txt", "w", encoding="utf-8") as f:
    #         f.write(evaluation)
    #     return {"overall_score": 0, "detailed_scores": {}, "improvement_suggestions": "Error processing outline, see outline_to_evaluate.txt"}
    # load json data

    with open(f"{save_dir}/outline_to_evaluate.txt", "a", encoding="utf-8") as f:
        f.write(evaluation)
    return evaluation

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/traversal.py \"your research query\"")
        print("Example: python scripts/traversal.py \"federated learning privacy\"")
        return
    query = sys.argv[1]
    info_dir = f"paper_data/{query.replace(' ', '_').replace(':', '')}/info"
    save_dir = f"paper_data/{query.replace(' ', '_').replace(':', '')}/paths"
    save_dir_core_paper = f"paper_data/{query.replace(' ', '_').replace(':', '')}/core_papers"

    # if dir not exist, create it
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_core_paper, exist_ok=True)
    graph_path = f"{info_dir}/paper_citation_graph.json"

    output_txt_path = f"{save_dir}/layer1_seed_taxonomy.txt"
    seed_taxonomy_output_path = f"{save_dir}/layer1_seed_taxonomy.json"
    layer_summary_output_path = f"{save_dir}/layer_method_group_summary.json"

    save_outline_dir = f"paper_data/{query.replace(' ', '_').replace(':', '')}/literature_review_output"
    os.makedirs(save_outline_dir, exist_ok=True)
    survey_outline_path = f"{save_outline_dir}/survey_outline.json"
    
    G = load_graph(graph_path)
    # Count the number of nodes in each layer
    layer_counts = {1: 0, 2: 0, 3: 0}
    for _, attr in G.nodes(data=True):
        layer = attr.get('layer')
        if layer in layer_counts:
            layer_counts[layer] += 1
    print(f"Layer counts: {layer_counts}")
    # --- Layer 1 seed direction summaries ---
    seeds = get_layer_seeds(G, 1)  # Only layer 1 seeds with new_direction=1
    all_text = []
    all_json = {}
    all_paths = []
    for seed in seeds:
        paper_infos, path, layer1_papers, layer2_papers, layer3_papers, layer2_summary, summary = bfs_from_seed(query, G, seed, max_development_paper=20, max_frontier_paper=30)
        all_paths.extend(path)
        seed_title = G.nodes[seed].get('title', '')
        seed_text = f"Seed: {seed_title}\nDevelopment direction taxonomy summary:\n{summary}\nPath: {path}\n"
        # print(seed_text)
        all_text.append(seed_text)
        all_json[seed] = {"seed_title": seed_title, "summary": summary, "path": path, "layer1_papers": layer1_papers, "layer2_papers": layer2_papers, "layer3_papers": layer3_papers, "layer2_summary": layer2_summary}
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))
    with open(seed_taxonomy_output_path, "w", encoding="utf-8") as f:
        json.dump(all_json, f, ensure_ascii=False, indent=2)
    print(f"All layer 1 seed taxonomy summaries saved to {output_txt_path} and {seed_taxonomy_output_path}")

    
    for node_id in all_paths:
        paper_attr = G.nodes[node_id]
        save_path = os.path.join(save_dir_core_paper, f"{node_id}.pdf")
        if os.path.exists(save_path):
            print(f"PDF for paper ID {node_id} already exists, skipping download.")
            continue
        download_paper(save_path, paper_attr.get('pdf_link'))

    # --- Layer method group summaries ---
    layer_method_group_txt = ""
    layer_method_group_json = {}
    for layer in [1, 2, 3]:
        layer_summary, papers = summarize_layer_method_groups(query, G, layer, max_papers=50)
        layer_method_group_txt += f"Layer {layer} method group summary:\n{layer_summary}\n\n"
        layer_method_group_json[f"{layer}"] = {
            "summary": layer_summary,
            "papers": papers
        }
        papers = [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer and attr.get('new_direction') == 1]
        if len(papers) > 50:
            papers = sorted(papers, key=lambda n: G.nodes[n].get('citation_count', 0), reverse=True)[:50]
        
        all_paths.extend(papers)
        # # Download papers in this layer's method group
        for n in papers:
            paper_attr = G.nodes[n]
            os.makedirs(save_dir_core_paper, exist_ok=True)
            save_path = os.path.join(save_dir_core_paper, f"{n}.pdf")
            if os.path.exists(save_path):
                print(f"PDF for paper ID {n} already exists, skipping download.")
                continue
            download_paper(save_path, paper_attr.get('pdf_link'))
    
    all_paths = list(set(all_paths))  # unique

    with open(layer_summary_output_path, "w", encoding="utf-8") as f:
        json.dump(layer_method_group_json, f, ensure_ascii=False, indent=2)
    print(f"Layer method group summaries saved to {layer_summary_output_path}")
    with open(f"{info_dir}/metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    save_papers_info_json(all_paths, G, info_dir, os.path.join(info_dir, "metadata_core_papers.json"), metadata)

    # Generate outline for the survey
    print('Generating survey outline...')
    layer_method_group_json = json.load(open(layer_summary_output_path, "r", encoding="utf-8"))
    develop_direction = json.load(open(seed_taxonomy_output_path, "r", encoding="utf-8"))
    outline = generate_survey_outline(query, layer_method_group_json, develop_direction, previous_outline='', improvement_suggestions='')
    iteration = 0
    # use another LLM to evaluate the logicality of the outline
    while iteration < 3:
        iteration += 1
        print(f"Ensuring outline coherence and logicality, iteration {iteration}...")
        evaluation = evaluate_outline(query, outline, save_outline_dir)
        print(evaluation)
        previous_outline = outline

        # else:
            # print("Outline needs improvement, regenerating...")
        print("Outline improving")
        outline = generate_survey_outline(query, layer_method_group_json, develop_direction, previous_outline, evaluation)
    # save outline to json file
    with open(survey_outline_path, "w", encoding="utf-8") as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)
    print(f"Survey outline saved to {survey_outline_path}")
if __name__ == "__main__":
    main()