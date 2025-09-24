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
    return [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer and attr.get('new_direction') == 1]

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
def summarize_with_gemini(paper_infos, path):
    # Compose a prompt for taxonomy of a direction, showing the traversal path
    prompt = "Summarize the following papers for taxonomy of a development direction, where each paper in the path cites the previous one. Make sure to highlight the development in the topic and how each paper contributes to the direction based on their methodology and publication time.\nPaper traversal path: "
    prompt += " -> ".join([info['title'] for info in paper_infos]) + "\n\n"
    for info in paper_infos:
        prompt += f"Title: {info['title']}\nAbstract: {info['abstract']}\nPublication Year: {info['year']}\n\n"
    chat_agent = ChatAgent()
    try:
        summary = chat_agent.gemini_chat(prompt, temperature=0.3)
    except Exception as e:
        print(f"Error occurred while summarizing direction: {e}")
        summary = "LLM summary placeholder for direction."
    return summary

def generate_survey_outline(taxonomy, develop_direction, improvement_suggestions=''):
    # Combine all taxonomy summaries into one prompt
    prompt = "Create a comprehensive literature review outline based on the following taxonomy summaries for three layers (foundational, development, and recent/trending) and development directions.\n\n"
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
        prompt += f"Here is the previously generated outline: {outline}\n"
        prompt += f"Here are some improvement suggestions for the outline: {improvement_suggestions}\n"
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
            with open ("../survey_outline_v2.txt", "w", encoding="utf-8") as f:
                f.write(outline)
    except Exception as e:
        print(f"Error occurred while generating survey outline: {e}")
        outline = "LLM outline placeholder."
    return outline

def get_layer_subgraph(G, layer):
    nodes = [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer]
    return G.subgraph(nodes)

def summarize_layer_method_groups(G, layer, max_papers=50):
    # Get papers with new_direction=1 in this layer
    papers = [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer and attr.get('new_direction') == 1]
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
    prompt = f"Summarize the method groups for the survey from the following {layer_desc.get(layer, '')} in layer {layer}. Divide the papers in the layer into subgroups based on their contributions and methodologies.\n\n"
    for info in infos:
        prompt += f"Title: {info['title']}\nAbstract: {info['abstract']}\nSummary: {info['summary']}\nPublication Year: {info['year']}\n\n"
    chat_agent = ChatAgent()
    try:
        summary = chat_agent.gemini_chat(prompt, temperature=0.3)
    except Exception as e:
        print(f"Error occurred while summarizing method groups for layer {layer}: {e}")
        summary = f"LLM summary placeholder for layer {layer}."
    return summary, papers

def save_papers_info_json(paper_ids, G, save_dir, output_path):
    metadata = {}

    # Process each paper entry
    for pid in paper_ids:
        attr = G.nodes[pid]
        # Generate a sanitized filename from the paper title
        id = pid
        filename = f"{id}.pdf"
        paper_dir = "paper_data/knowledge_graph_embedding/core_papers/"
        # Create a dictionary for the paper's metadata
        paper_metadata = {
            "title": attr.get('title'),
            "authors": attr.get('authors', []),
            "published_date": str(attr.get('year', '')),
            # "venue": attr.get('venue'),
            "journal": attr.get('journal'),
            "abstract": attr.get('abstract'),
            "keywords": attr.get("survey_keywords", []),  # Assuming no keywords in the source data
            "paper_type": attr.get('paper_type'),
            "summary": attr.get('summary'),
            "file_path": os.path.join(paper_dir, filename), 
            "journal": attr.get('venue', ''),
        }
        
        # Add the metadata to the main dictionary using the filename as the key
        metadata[filename] = paper_metadata

    # Write the new metadata to an output JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {len(metadata)} papers to {output_path}")

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
    return info, path

def evaluate_outline(outline, max_iterations=3):
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
    Evaluate the logicality and coherence of the following literature review outline. Ensure that the sections and subsections are organized in a clear and logical manner, reflecting a coherent flow of ideas. If there are any issues with the structure, suggest improvements.
    Outline to evaluate:
    {outline_text}
    **Evaluation Criteria** (Rate each from 1-5, where 5 is excellent):
    1. **Logical Flow** (1-5): Does the outline follow a clear and logical progression of ideas?
    2. **Coherence** (1-5): Are the sections and subsections coherent and well-connected?
    3. **Relevance** (1-5): Do the sections align with the overall topic and objectives of the literature review?
    4. **Clarity** (1-5): Is the outline clear and easy to understand?
    5. **Content Coverage** (1-5): Does the section comprehensively cover the expected focus area?

    **IMPORTANT**: 
    - Return ONLY valid JSON without any markdown formatting or code blocks
    - Escape all backslashes and quotes properly in JSON strings
    - Do not include any special characters that might break JSON parsing
    
    **Response Format** (JSON only):
        {{
            "overall_score": <average_score>,
            "detailed_scores": {{
                "logical_flow": <score>,
                "coherence": <score>,
                "relevance": <score>,
                "clarity": <score>,
                "content_coverage": <score>
            }},
            "improvement_suggestions": "<detailed suggestions for improvement>"
        }}
    Consider an outline satisfactory if overall_score >= 3.5 and no individual score is below 3.0.
    """
    # call chat agent to evaluate the input outline
    chat_agent = ChatAgent()
    evaluation = chat_agent.gemini_chat(evaluation_prompt, temperature=0.3)

    # parse the output json
    try:
        json_string = evaluation.strip().removeprefix('```json\n').removesuffix('\n```')
    except Exception as e:
        print(f"Error processing outline: {e}")
        with open("outline_to_evaluate.txt", "w", encoding="utf-8") as f:
            f.write(evaluation)
        return {"overall_score": 0, "detailed_scores": {}, "improvement_suggestions": "Error processing outline, see outline_to_evaluate.txt"}
    # load json data
    evaluation = json.loads(json_string)
    return evaluation

def main():
    save_dir = "paper_data/knowledge_graph_embedding"
    # if dir not exist, create it
    os.makedirs(save_dir, exist_ok=True)

    graph_path = "paper_citation_graph.json"
    output_txt_path = "layer1_seed_taxonomy.txt"
    output_json_path = "layer1_seed_taxonomy.json"

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
        paper_infos, path = dfs_from_seed(G, seed, last_layer=3, max_paper_number=30)
        all_paths.extend(path)
        summary = summarize_with_gemini(paper_infos, path)
        seed_title = G.nodes[seed].get('title', '')
        seed_text = f"Seed: {seed_title}\nDevelopment direction taxonomy summary:\n{summary}\nPath: {path}\n"
        # print(seed_text)
        all_text.append(seed_text)
        all_json[seed] = {"seed_title": seed_title, "summary": summary, "path": path}
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_json, f, ensure_ascii=False, indent=2)
    print(f"All layer 1 seed taxonomy summaries saved to {output_txt_path} and {output_json_path}")

    
    for node_id in all_paths:
        paper_attr = G.nodes[node_id]
        save_path = os.path.join(save_dir, f"{node_id}.pdf")
        if os.path.exists(save_path):
            print(f"PDF for paper ID {node_id} already exists, skipping download.")
            continue
        download_paper(save_path, paper_attr.get('pdf_link'))

    # --- Layer method group summaries ---
    layer_method_group_txt = ""
    layer_method_group_json = {}
    for layer in [1, 2, 3]:
        layer_summary, papers = summarize_layer_method_groups(G, layer, max_papers=50)
        layer_method_group_txt += f"Layer {layer} method group summary:\n{layer_summary}\n\n"
        layer_method_group_json[f"{layer}"] = {
            "summary": layer_summary,
            "papers": papers
        }
        papers = [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer and attr.get('new_direction') == 1]
        if len(papers) > 50:
            papers = sorted(papers, key=lambda n: G.nodes[n].get('citation_count', 0), reverse=True)[:50]
        
        all_paths.extend(papers)
        # Download papers in this layer's method group
        for n in papers:
            paper_attr = G.nodes[n]
            save_dir_core_paper = "/media/aiserver/New Volume/HDD_linux/bear/SurveyX/paper_data/KNOWLEDGE_GRAPH_EMBEDDING/core_papers"
            os.makedirs(save_dir_core_paper, exist_ok=True)
            save_path = os.path.join(save_dir_core_paper, f"{n}.pdf")
            if os.path.exists(save_path):
                print(f"PDF for paper ID {n} already exists, skipping download.")
                continue
            download_paper(save_path, paper_attr.get('pdf_link'))
    
    all_paths = list(set(all_paths))  # unique
    # with open("layer_method_group_summary.txt", "w", encoding="utf-8") as f:
    #     f.write(layer_method_group_txt)

    with open("layer_method_group_summary.json", "w", encoding="utf-8") as f:
        json.dump(layer_method_group_json, f, ensure_ascii=False, indent=2)
    print("Layer method group summaries saved to layer_method_group_summary.txt and layer_method_group_summary.json")
    
    save_papers_info_json(all_paths, G, save_dir, os.path.join(save_dir, "metadata.json"))

    # Generate outline for the survey
    print('Generating survey outline...')
    layer_method_group_json = json.load(open("layer_method_group_summary.json", "r", encoding="utf-8"))
    develop_direction = json.load(open("layer1_seed_taxonomy.json", "r", encoding="utf-8"))
    outline = generate_survey_outline(layer_method_group_json, develop_direction, improvement_suggestions='')
    iteration = 0
    # use another LLM to evaluate the logicality of the outline
    while iteration < 3:
        iteration += 1
        print(f"Ensuring outline coherence and logicality, iteration {iteration}...")
        evaluation = evaluate_outline(outline)
        overall_score = evaluation.get("overall_score", 0)
        detailed_scores = evaluation.get("detailed_scores", {})
        improvement_suggestions = evaluation.get("improvement_suggestions", "")
        if 'outline_to_evaluate.txt' in improvement_suggestions:
            continue
        print(f"Outline evaluation - Overall Score: {overall_score}, Detailed Scores: {detailed_scores}")
        if overall_score >= 3.5 and all(score >= 3.0 for score in detailed_scores.values()):
            print("Outline is satisfactory.")
            break
        else:
            print("Outline needs improvement, regenerating...")
            outline = generate_survey_outline(layer_method_group_json, develop_direction, improvement_suggestions)
    # save outline to json file
    with open("survey_outline_v2.json", "w", encoding="utf-8") as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)
    print("Survey outline saved to survey_outline_v2.json")
if __name__ == "__main__":
    main()