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
from leiden import Leiden_summarizer
from prompt import PromptHelper

prompt_helper = PromptHelper()

class Config:
    """Configuration constants for the literature review generation"""
    MAX_PAPERS_PER_LAYER = 50
    MAX_DEVELOPMENT_PAPERS = 20
    MAX_FRONTIER_PAPERS = 30
    MAX_CONTEXT_PAPERS = 30  # Limit papers sent to LLM at once
    MAX_OUTLINE_ITERATIONS = 3
    MIN_QUALITY_SCORE = 4.0
    REQUIRED_MIN_SCORE = 3.0
    LLM_TEMPERATURE = 0.3
    
    # Section requirements
    MIN_MAIN_SECTIONS = 5
    MAX_MAIN_SECTIONS = 8
    MIN_SUBSECTIONS = 2
    MAX_SUBSECTIONS = 5
    MAX_OUTLINE_EVALUATION = 3

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
def llm_summary_with_retry(prompt, placeholder="LLM summary placeholder", temperature=0.3):
    chat_agent = ChatAgent()
    finish_generated = False
    tries = 0
    summary = placeholder
    while finish_generated == False and tries < 3:
        tries += 1
        try:
            summary = chat_agent.gemini_chat(prompt, temperature=temperature)
            finish_generated = True
        except Exception as e:
            print(f"Error occurred while summarizing direction: {e}")
    return summary
# Call Gemini LLM to summarize
def summarize_development_path(query,paper_infos, path, previous_context=''):
    # Compose a prompt for taxonomy of a direction, showing the traversal path
    papers_text = ''
    for info in paper_infos:
        try:
            if info['summary'] != info['abstract'] and info['summary'] != '':
                papers_text += f"Title: {info['title']}\nAbstract: {info['abstract']}\nSummary: {info['summary']}\nPublication Year: {info['year']}\n\n"
        except: papers_text += f"Title: {info['title']}\nAbstract: {info['abstract']}\nPublication Year: {info['year']}\n\n"
    
    prompt = prompt_helper.generate_prompt(prompt_helper.BFS_PROMPT, 
                                           paras={
                                            'QUERY': query,
                                            'NUMBER_OF_PAPERS': str(len(paper_infos)),
                                            'PAPER_INFO': papers_text,
                                            'PREVIOUS_CONTEXT': ("PREVIOUS CONTEXT:" + chr(10) + previous_context + chr(10)) if previous_context else ""                
                                            })
    with open('test_prompts.txt', "a") as f:
        f.write(f"DEVELOPMENT PATH PROMPT:\n {prompt}")
    summary = llm_summary_with_retry(prompt, "LLM summary placeholder for direction.")
    return summary

def summarize_layer_method_groups(query, G, layer):
    # Get papers with new_direction=1 in this layer
    papers = [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer]
    papers = sorted(
        papers,
        key=lambda node: G.nodes[node].get('score', float('-inf')),
        reverse=True
    )
    
    papers = sorted(papers, key=lambda n: G.nodes[n].get('citation_count', 0), reverse=True)
    infos = []
    for n in papers:
        attr = G.nodes[n]
        infos.append({
            'title': attr.get('title', ''),
            'abstract': attr.get('abstract', ''),
            'summary': attr.get('summary', ''),
            'year': attr.get('year', 0)
        })
    # Compose prompt for LLM
    layer_desc = {1: 'foundational papers', 2: 'development papers', 3: 'recent and trending papers'}
    papers_info = ''
    for info in infos:
        papers_info += f"Title: {info['title']}\nAbstract: {info['abstract']}\nSummary: {info['summary']}\nPublication Year: {info['year']}\n\n"
    prompt = prompt_helper.generate_prompt(prompt_helper.LAYER_PROMPT, 
                                           paras={
                                               'QUERY': query,
                                               'LAYER_DESCRIPTION': layer_desc.get(layer, ''),
                                               'NUMBER_OF_PAPERS': str(len(infos)),
                                               'PAPER_INFO': papers_info
                                           })
    placeholder = f"LLM summary placeholder for layer {layer}."
    summary = llm_summary_with_retry(prompt, placeholder)
    with open('test_prompts.txt', "a") as f:
        f.write(f"LAYER PROMPT:\n {prompt}")
    return summary, papers

def summarize_community(query, G, papers):
    infos = []
    for n in papers:
        attr = G.nodes[n]
        infos.append({
            'title': attr.get('title', ''),
            'abstract': attr.get('abstract', ''),
            'summary': attr.get('summary', ''),
            'year': attr.get('year', 0)
        })
    # Compose prompt for LLM
    paper_info = ''
    for info in infos:
        paper_info += f"Title: {info['title']}\nAbstract: {info['abstract']}\nSummary: {info['summary']}\nPublication Year: {info['year']}\n\n"
    prompt = prompt_helper.generate_prompt(prompt_helper.COMMUNITY_PROMPT,
                                           paras={
                                               'QUERY': query,
                                               'PAPER_INFO': paper_info
                                           })
    summary = llm_summary_with_retry(prompt)
    with open('test_prompts.txt', "a") as f:
        f.write(f"COMMUNITY PROMPT:\n {prompt}")
    return summary, papers

def generate_survey_outline(query, layer_taxonomies, development_directions, communities_summaries, previous_outline='', improvement_suggestions=''):
    taxonomy_text = ""
    for layer in [1]:
        layer_desc = {
            1: 'Foundational Layer',
            2: 'Development Layer', 
            3: 'Recent/Trending Layer'
        }
        summary = layer_taxonomies.get(f'layer_{layer}', {}).get('summary', '')
        # print(f"Layer {layer} summary: \n{summary}")
        if summary:
            taxonomy_text += f"\n### {layer_desc[layer]}:\n{summary}\n"
    
    # Build development directions text
    dev_dir_text = ""
    for seed_id, info in development_directions.items():
        dev_dir_text += f"\n**Direction: {info.get('seed_title', 'Unknown')}** (Seed ID: {seed_id})\n"
        dev_dir_text += f"{info.get('summary', '')}\n"
    
    community_text = ""
    for key, community in communities_summaries.items():
        community_text += f"Paper community ID: {key} \nSummary:\n{community.get('summary', '')}\n\n"


    base_prompt = prompt_helper.generate_prompt(prompt_helper.OUTLINE_PROMPT, 
                                           paras={
                                               "QUERY": query,
                                               "LAYER_TAXONOMIES": taxonomy_text,
                                               "DEVELOPMENT_DIRECTIONS": dev_dir_text,
                                               "PAPER_COMMUNITIES": community_text, 
                                               "MIN_MAIN_SECTIONS": Config.MIN_MAIN_SECTIONS,
                                               "MAX_MAIN_SECTIONS": Config.MAX_MAIN_SECTIONS,
                                               "MIN_SUBSECTIONS": Config.MIN_SUBSECTIONS,
                                               "MAX_SUBSECTIONS": Config.MAX_SUBSECTIONS
                                           })
    if previous_outline and improvement_suggestions!='':
        base_prompt += f"""
# ITERATION: Improving Previous Outline

PREVIOUS OUTLINE:
{json.dumps(previous_outline, indent=2)}

EVALUATION FEEDBACK:
{improvement_suggestions}

TASK: Regenerate the outline addressing all feedback while maintaining quality in other areas.
"""
    with open('test_prompts.txt', "a") as f:
        f.write(f"OUTLINE PROMPT:\n {base_prompt}")
    chat_agent = ChatAgent()
    try:
        response = chat_agent.gemini_chat(base_prompt, temperature=Config.LLM_TEMPERATURE)
        
        # Clean response
        response = response.strip()
        # Remove markdown code blocks if present
        if response.startswith('```'):
            response = re.sub(r'^```(?:json)?\n', '', response)
            response = re.sub(r'\n```$', '', response)
        
        # Parse JSON
        outline = json.loads(response)
        
        # Validate structure
        if not isinstance(outline, list):
            raise ValueError("Outline must be a JSON array")
        
        print(f"Generated outline with {len(outline)} sections")
        return outline
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Problematic response (first 500 chars): {response[:500]}")
        
        # Save for debugging
        debug_path = "outline_generation_error.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(f"Response:\n{response}")
        
        raise Exception(f"Failed to parse outline JSON. Debug info saved to {debug_path}")
    
    except Exception as e:
        print(f"Outline generation failed: {e}")
        raise

def evaluate_outline(query, outline, save_dir):
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
    • The outline structure MUST include Introduction and Conclusion, check if the given outline have these sections and suggest improvement if missing.
    • Specific suggestions for improvement (only if issues are found).
    Outline to evaluate: {outline_text}

    """
    with open('test_prompts.txt', "a") as f:
        f.write(f"EVALUATE PROMPT:\n {evaluation_prompt}")
    evaluation = llm_summary_with_retry(evaluation_prompt)

    with open(f"{save_dir}/outline_to_evaluate.txt", "a", encoding="utf-8") as f:
        f.write(evaluation)
    return evaluation

def get_layer_subgraph(G, layer):
    nodes = [n for n, attr in G.nodes(data=True) if attr.get('layer') == layer]
    return G.subgraph(nodes)

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
                layer_2_summary = summarize_development_path(query, layer1_papers + layer2_papers, path)
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
    if not layer_2_summary: layer_2_summary = summarize_development_path(query, layer1_papers + layer2_papers, path)
    if len(layer3_papers): path_taxonomy = summarize_development_path(query,layer3_papers, path, layer_2_summary)
    else: 
        path_taxonomy = summarize_development_path(query, layer1_papers + layer2_papers + layer3_papers, path)
 
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
    summary = summarize_development_path(info, path)
    return info, path, summary

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
    community_summary_output_path = f"{save_dir}/communities_summary.json"
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
    # # --- Layer 1 seed direction summaries ---
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

    # for node_id in all_paths:
    #     paper_attr = G.nodes[node_id]
    #     save_path = os.path.join(save_dir_core_paper, f"{node_id}.pdf")
    #     if os.path.exists(save_path):
    #         print(f"PDF for paper ID {node_id} already exists, skipping download.")
    #         continue
    #     download_paper(save_path, paper_attr.get('pdf_link'))

    # --- Layer method group summaries ---
    layer_method_group_txt = ""
    layer_method_group_json = {}
    for layer in [1]:
        layer_summary, papers = summarize_layer_method_groups(query, G, layer)
        layer_method_group_txt += f"Layer {layer} method group summary:\n{layer_summary}\n\n"
        layer_method_group_json[f"layer_{layer}"] = {
            "summary": layer_summary,
            "papers": papers
        }       
        all_paths.extend(papers)
        # # Download papers in this layer's method group
        # for n in papers:
        #     paper_attr = G.nodes[n]
            # os.makedirs(save_dir_core_paper, exist_ok=True)
            # save_path = os.path.join(save_dir_core_paper, f"{n}.pdf")
            # if os.path.exists(save_path):
            #     print(f"PDF for paper ID {n} already exists, skipping download.")
            #     continue
            # download_paper(save_path, paper_attr.get('pdf_link'))
    
    all_paths = list(set(all_paths))  # unique
    with open(layer_summary_output_path, "w", encoding="utf-8") as f:
        json.dump(layer_method_group_json, f, ensure_ascii=False, indent=2)
    print(f"Layer method group summaries saved to {layer_summary_output_path}")

    community_summaries = {}
    ls = Leiden_summarizer(graph_path)
    communities = ls.leiden_algorithm()
    for i, community in enumerate(communities):
        summary, papers = summarize_community(query, G, community)
        community_summaries[f"community_{i}"] = {
            "summary": summary,
            "papers": papers
        }
    with open(community_summary_output_path, "w", encoding="utf-8") as f:
        json.dump(community_summaries, f, ensure_ascii=False, indent=2)
    print(f"Layer method group summaries saved to {community_summary_output_path}")
    with open(f"{info_dir}/metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    save_papers_info_json(all_paths, G, info_dir, os.path.join(info_dir, "metadata_core_papers.json"), metadata)

    # Generate outline for the survey
    print('Generating survey outline...')
    layer_method_group_json = json.load(open(layer_summary_output_path, "r", encoding="utf-8"))
    develop_direction = json.load(open(seed_taxonomy_output_path, "r", encoding="utf-8"))
    communities_summary = json.load(open(community_summary_output_path, "r", encoding="utf8"))
    outline = generate_survey_outline(query, layer_method_group_json, develop_direction, communities_summary, previous_outline='', improvement_suggestions='')
    iteration = 0
    # use another LLM to evaluate the logicality of the outline
    while iteration < Config.MAX_OUTLINE_EVALUATION:
        iteration += 1
        print(f"Ensuring outline coherence and logicality, iteration {iteration}...")
        evaluation = evaluate_outline(query, outline, save_outline_dir)
        print(evaluation)
        previous_outline = outline

        # else:
            # print("Outline needs improvement, regenerating...")
        print("Outline improving")
        outline = generate_survey_outline(query, layer_method_group_json, develop_direction, communities_summary, previous_outline, evaluation)
    # save outline to json file
    with open(survey_outline_path, "w", encoding="utf-8") as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)
    print(f"Survey outline saved to {survey_outline_path}")
if __name__ == "__main__":
    main()