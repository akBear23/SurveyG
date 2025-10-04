import sys
import google.generativeai as genai
import PyPDF2
import docx
import os
import json
import re
from typing import Optional, Dict, List, Tuple
import pandas as pd
from datetime import datetime
import networkx as nx
import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (containing 'writing/') to sys.path
sys.path.insert(0, os.path.join(current_dir, '../'))

from writing.summarize import PaperSummarizerRAG
import sys
from dotenv import load_dotenv, find_dotenv
from pathlib import Path 
from scripts.prompt import PromptHelper

class LiteratureReviewGenerator:
    def __init__(self, query, api_key: str):
        """
        Initialize Literature Review Generator with Gemini API key
        
        Args:
            api_key (str): Gemini API key
        """
        self.prompt_helper = PromptHelper()
        genai.configure(api_key=api_key)
        self.query = query
        self.save_dir = f"paper_data/{self.query.replace(' ', '_').replace(':', '')}/literature_review_output"
        os.makedirs(self.save_dir, exist_ok=True)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.papers_data = []
        self.citations_map = {}  # Map paper names to citation keys
        rag_db_path = f"paper_data/{query.replace(' ', '_').replace(':', '')}/rag_database"
        os.makedirs(f"paper_data/{query.replace(' ', '_').replace(':', '')}/rag_database/", exist_ok=True)

        self.summarizer = PaperSummarizerRAG(query, api_key, rag_db_path)
        self.max_improvement_iterations = 10  # Maximum iterations for section improvement
        self.layer_method_group_json = json.load(open(f"paper_data/{self.query.replace(' ', '_').replace(':', '')}/paths/layer_method_group_summary.json", "r", encoding="utf-8"))
        self.develop_direction = json.load(open(f"paper_data/{self.query.replace(' ', '_').replace(':', '')}/paths/layer1_seed_taxonomy.json", "r", encoding="utf-8"))
        self.graph_path = f"paper_data/{self.query.replace(' ', '_').replace(':', '')}/info/paper_citation_graph.json"
        self.G = self.load_graph(self.graph_path)
    def load_graph(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        G = nx.DiGraph()
        for node in data['nodes']:
            G.add_node(node['id'], **node)
        for edge in data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge)
        return G

    #region evaluate section quality
    def evaluate_section_quality(self, section_title: str, section_content: str, 
                            section_focus: str, outline: str, pre_section: str) -> Dict[str, any]:
        """
        Evaluate the quality of a literature review section based on multiple criteria
        
        Args:
            section_title (str): Title of the section
            section_content (str): Content of the section
            section_focus (str): Expected focus/scope of the section
            
        Returns:
            Dict: Evaluation results with scores and improvement suggestions
        """
        evaluation_prompt = self.prompt_helper.generate_prompt(self.prompt_helper.EVALUATE_SECTION_PROMPT, 
                                                               paras={
                                                                   'PRE_SECTION': pre_section,
                                                                   'SECTION_TITLE': section_title,
                                                                   'SECTION_FOCUS': section_focus, 
                                                                   'OUTLINE': outline,
                                                                   'SECTION_CONTENT': section_content
                                                               })
        
        try:
            response = self.model.generate_content(
                evaluation_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=102400,
                    temperature=0.2,
                )
            )
            
            # Debug: Print raw response
            print(f"Raw response: {response.text}")
            
            # Clean up response text
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json\n', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```\n', '').replace('```', '').strip()
            
            # Check if response is empty
            if not response_text:
                raise ValueError("Empty response from API")
            
            # Additional cleaning for escape characters
            # Fix common escape issues
            response_text = response_text.replace('\\', '\\\\')  # Escape backslashes
            response_text = response_text.replace('\\"', '"')    # Fix escaped quotes
            response_text = response_text.replace('\\\\n', '\\n')  # Fix double-escaped newlines
            
            print(f"Cleaned response text: {response_text[:500]}...")
            
            # Parse JSON response
            evaluation_result = json.loads(response_text)
            return evaluation_result
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Error at position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
            print(f"Problematic text around error: {response_text[max(0, e.pos-50):e.pos+50] if hasattr(e, 'pos') else 'N/A'}")
            
            # Try alternative parsing approaches
            try:
                # Method 1: Try to fix common JSON issues
                fixed_text = response_text.replace('\n', '\\n').replace('\t', '\\t')
                evaluation_result = json.loads(fixed_text)
                return evaluation_result
            except:
                pass
            
            try:
                # Method 2: Use regex to extract JSON parts
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    evaluation_result = json.loads(json_str)
                    return evaluation_result
            except:
                pass
            
            # Return fallback result
            return {
                "overall_score": 3.0,
                "individual_scores": {
                    "content_coverage": 3,
                    "citation_density": 3,
                    "academic_rigor": 3,
                    "synthesis_quality": 3,
                    "critical_analysis": 3,
                    "coherence": 3,
                    "depth_of_analysis": 3,
                    "specificity": 3
                },
                "strengths": [],
                "weaknesses": ["JSON parsing failed"],
                "is_satisfactory": False,
                "improvement_needed": ["Evaluation failed - manual review needed"],
                "suggested_queries": [],
                "error": f"JSON parsing error: {str(e)}"
            }
            
        except Exception as e:
            print(f"Error evaluating section: {e}")
            return {
                "overall_score": 3.0,
                "individual_scores": {
                    "content_coverage": 3,
                    "citation_density": 3,
                    "academic_rigor": 3,
                    "synthesis_quality": 3,
                    "critical_analysis": 3,
                    "coherence": 3,
                    "depth_of_analysis": 3,
                    "specificity": 3
                },
                "strengths": [],
                "weaknesses": ["Evaluation failed"],
                "is_satisfactory": False,
                "improvement_needed": ["Evaluation failed - manual review needed"],
                "suggested_queries": [],
                "error": f"General error: {str(e)}"
            }
    #endregion
    
    #region retrieve additional papers from RAG
    def retrieve_additional_papers(self, queries: List[str], n_results: int = 3) -> List[Dict]:
        """
        Retrieve additional papers from RAG system based on queries
        
        Args:
            queries (List[str]): List of search queries
            n_results (int): Number of results per query
            
        Returns:
            List[Dict]: Combined search results
        """
        all_results = []
        seen_doc_ids = set()
        
        for query in queries:
            try:
                results = self.summarizer.search_similar_papers(query, n_results)
                
                # Filter out duplicates and add to results
                for result in results:
                    doc_id = result.get('doc_id', '')
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        all_results.append(result)
                        
            except Exception as e:
                print(f"Error retrieving papers for query '{query}': {e}")
                continue
        
        return all_results
    #endregion
    
    # region improve section with additional papers
    def improve_section_with_additional_papers(self, section_title: str, current_content: str,
                                             section_focus: str, additional_papers: List[Dict],
                                             evaluation_feedback: Dict) -> str:
        """
        Improve a section by incorporating additional papers and addressing feedback
        
        Args:
            section_title (str): Title of the section
            current_content (str): Current section content
            section_focus (str): Expected focus of the section
            additional_papers (List[Dict]): Additional papers from RAG retrieval
            evaluation_feedback (Dict): Feedback from section evaluation
            
        Returns:
            str: Improved section content
        """
        
        # Prepare additional papers information
        additional_info = ""
        for i, paper in enumerate(additional_papers, 1):
            metadata = paper.get('metadata', {})
            additional_info += f"**Additional Paper {i}**:\n"
            additional_info += f"Title: {metadata.get('title', 'N/A')}\n"
            additional_info += f"Authors: {metadata.get('authors', 'N/A')}\n"
            additional_info += f"Summary: {paper.get('full_summary', paper.get('abstract', 'N/A'))}\n"
            additional_info += f"Relevance Score: {paper.get('similarity_score', 0):.3f}\n"
            if paper.get('keyword', []) != []:
                additional_info += f"Keywords: {', '.join(paper.get('keyword', []))}"
            additional_info += f"Citation Key: {paper.get('citation_key', '')}\n\n"
        improvement_prompt = self.prompt_helper.generate_prompt(self.prompt_helper.SECTION_IMPROVE_PROMPT,
                                                                paras={
                                                                    "SECTION_TITLE": section_title,
                                                                    "SECTION_FOCUS": section_focus,
                                                                    "CURRENT_CONTENT": current_content,
                                                                    "OVERALL_SCORE": {evaluation_feedback.get('overall_score', 'N/A')},
                                                                    "STRENGTHS": {', '.join(evaluation_feedback.get('strengths', []))},
                                                                    "WEAKNESS": {', '.join(evaluation_feedback.get('weaknesses', []))},
                                                                    "IMPROVEMENT_NEEDED": {', '.join(evaluation_feedback.get('improvement_needed', []))},
                                                                    "ADDITIONAL_INFO": additional_info
                                                                })
        
        try:
            response = self.model.generate_content(
                improvement_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=102400,
                    temperature=0.4,
                )
            )
            return response.text
            
        except Exception as e:
            print(f"Error improving section: {e}")
            return current_content
    
    
    #endregion
    def get_proofs_text(self, proof_ids: list) -> str:
        """
        get proofs text (including paper title, abstract, year from the graph attributes)
        """
        proofs_text = ''
        for proof in proof_ids:
            try: 
                proof = proof.lower().strip()
                if 'layer' in proof:
                    proof.replace('layer', '').strip()
            except:
                pass
            if proof in self.develop_direction.keys():
                direction_dict = self.develop_direction[proof]
                proofs_text += 'Development direction: \n'
                proofs_text += direction_dict.get('summary', '') + '\n'
                nodes = [proof]
                nodes.extend(direction_dict.get('paths', []))
                for node in nodes:
                    if node in self.G.nodes:
                        node_data = self.G.nodes[node]
                        title = node_data.get('title', '')
                        abstract = node_data.get('abstract', '')
                        year = node_data.get('year', '')
                        summary = node_data.get('summary', '')
                        proofs_text += f"Title: {title}\n"
                        proofs_text += f"Abstract: {abstract}\n"
                        proofs_text += f"Year: {year}\n"
                        proofs_text += f"Summary: {summary}\n\n"
            if proof in [1, 2, 3]:
                proofs_text += 'Taxonomy summaries: \n'
                layer_dict = self.layer_method_group_json.get(f"{proof}", {})
                proofs_text += layer_dict.get('summary', '') + '\n'
                for paper in layer_dict.get('papers', []):
                    if paper in self.G.nodes:
                        node_data = self.G.nodes[paper]
                        title = node_data.get('title', '')
                        abstract = node_data.get('abstract', '')
                        year = node_data.get('year', '')
                        summary = node_data.get('summary', '')
                        proofs_text += f"Title: {title}\n"
                        proofs_text += f"Abstract: {abstract}\n"
                        proofs_text += f"Year: {year}\n"
                        proofs_text += f"Summary: {summary}\n\n"
        return proofs_text

# region write initial section
    def write_initial_section(self, section_title: str, processed_papers: List[Dict], 
                            section_focus: str, outline: str, proof_ids: list, pre_section:str) -> str:
        """
        Write initial version of a literature review section
        """
        # step 1: Prepare citations info
        citations_info = ""
        for paper in processed_papers:
            metadata = paper['metadata']
            citations_info += f"\\cite{{{paper['citation_key']}}}: {metadata.get('title', 'N/A')} by {metadata.get('authors', 'N/A')} ({metadata.get('published_date', 'N/A')})\n"
        
        # step 2: get papers summary
        papers_summary = ""
        for i, paper in enumerate(processed_papers, 1):
            summary = paper.get('summary', '')
            papers_summary += f"**Paper {i} ({paper['citation_key']})**: {paper['file_name']}\n{summary}\n\n"
        # step 3: get proofs text (including paper title, abstract, year from the graph attributes)
        proofs_text = self.get_proofs_text(proof_ids)
        section_prompt = self.prompt_helper.generate_prompt(self.prompt_helper.WRITE_INITIAL_SECTION_PROMPT,
                                                            paras={
                                                                'SECTION_TITLE': section_title,
                                                                'SECTION_FOCUS': section_focus,
                                                                'PROOFS_TEXT': proofs_text,
                                                                'CITATION_INFO': citations_info,
                                                                'OUTLINE': outline,
                                                                'PAPERS_SUMMARY': papers_summary[:50000],
                                                                'PRE_SECTION': pre_section
                                                            })
        
        # step 3: Generate initial section content
        try:
            response = self.model.generate_content(
                section_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=102400,
                    temperature=0.4,
                )
            )
            return response.text
        except Exception as e:
            print(f"Error writing initial section: {e}")
            return ""
    #endregion

    #region write literature review section with reflection
    def write_literature_review_section_with_reflection(self, section_title: str, 
                                                      processed_papers: List[Dict],
                                                      section_focus: str, outline: str, proof_ids: list, pre_section: str) -> str:
        """
        Write a literature review section with self-reflection and iterative improvement
        
        Args:
            section_title (str): Title of the section
            processed_papers (List[Dict]): List of processed papers
            section_focus (str): Specific focus for this section
            
        Returns:
            str: Final improved section content
        """
        
        print(f"   Writing: {section_title}")
        
        # Step 1: Generate initial section
        initial_section = self.write_initial_section(section_title, processed_papers, section_focus, outline, proof_ids, pre_section)

        current_content = initial_section
        iteration = 0
        
        while iteration < self.max_improvement_iterations:
            iteration += 1
            print(f"      Iteration {iteration}: Evaluating section quality...")
            
            # Step 2: Evaluate section quality
            evaluation = self.evaluate_section_quality(section_title, current_content, section_focus, outline, pre_section)
            
            print(f"      Overall Score: {evaluation.get('overall_score', 'N/A')}/5")
            
            # Step 3: Check if section is satisfactory
            if evaluation.get('is_satisfactory', False):
                print(f"      Section meets quality standards!")
                break
                
            if iteration >= self.max_improvement_iterations:
                print(f"      Maximum iterations reached. Using current version.")
                break
            
            # Step 4: Retrieve additional papers if needed
            suggested_queries = evaluation.get('suggested_queries', [])

            with open(f"{self.save_dir}/suggested_queries.txt", "a") as f:
                f.write(f"Suggested queries for iteration {iteration}:\n, {suggested_queries}")
            if suggested_queries:
                print(f"      Retrieving additional papers for: {', '.join(suggested_queries[:2])}")
                additional_papers = self.retrieve_additional_papers(suggested_queries[:2])
                print(f"      Found {len(additional_papers)} additional papers")
            else:
                additional_papers = []
            
            # Step 5: Improve section
            print(f"      Improving section based on feedback...")
            current_content = self.improve_section_with_additional_papers(
                section_title, current_content, section_focus, additional_papers, evaluation
            )
        
        return current_content
    
    #endregion
    def parse_outline_text(file_path: str) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
        """
        Parses a Markdown-formatted outline text file to extract section titles and content.
        It now also extracts and returns the 'Section Focus' for each section.

        Args:
            file_path (str): The path to the outline text file.

        Returns:
            Tuple[List[str], Dict[str, str], Dict[str, str]]: A tuple containing:
                - A list of section titles in order.
                - A dictionary mapping each section title to its full content.
                - A dictionary mapping each section title to its 'Section Focus'.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                outline_content = f.read()
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return [], {}, {}
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return [], {}, {}

        # Regex to find all section titles (e.g., "### 1. Introduction")
        section_titles_raw = re.findall(r'###\s*(.*)', outline_content)

        # Split the content by section titles to get the body of each section
        sections_raw = re.split(r'###\s*.*', outline_content)
        
        # Remove the first element (pre-title text) and clean up any empty strings
        sections_content = [s.strip() for s in sections_raw if s.strip()]

        # Create the final dictionaries and list
        section_title_list = []
        sections_dict = {}
        section_definitions = {}

        for title, content in zip(section_titles_raw, sections_content):
            # Extract the clean title without numbering
            clean_title = re.sub(r'^\d+\.\s*', '', title).strip()
            section_title_list.append(clean_title)
            
            # Add the full content to the main sections dictionary
            sections_dict[clean_title] = content
            
            # Regex to find the Section Focus part within the section's content
            focus_match = re.search(r'\s*Section Focus:\s*(.*)', content)
            if focus_match:
                section_definitions[clean_title] = focus_match.group(1).strip()
            else:
                section_definitions[clean_title] = "N/A"

        return section_title_list, sections_dict, section_definitions
    #region generate complete literature review
    def generate_complete_literature_review(self, paper_paths: List[str], 
                                          review_title: str = "Literature Review") -> Dict:
        """
        Generate a complete literature review from multiple papers with self-reflection
        """
        print("Starting literature review generation...")
        
        # Step 1: Process all papers
        print("\n1. Processing papers...")

        # process core papers first for drafting (seprate folder)
        core_papers_path = os.path.join(paper_paths, "core_papers")
        core_papers = self.summarizer.process_folder(
            folder_path=core_papers_path,
            skip_existing=True,  # Skip if already processed
            delay_seconds=0.0, 
            metadata_file = os.path.join(f"paper_data/{self.query}/info", "metadata.json")
        )
        
        # # Process all papers
        all_papers = self.summarizer.process_folder(
            folder_path=paper_paths,
            skip_existing=True,  # Skip already processed files
            delay_seconds=0.0,
            metadata_file = os.path.join(f"paper_data/{self.query}/info", "metadata_all_papers.json")
        )
        def check_variable_type(variable):
            if isinstance(variable, dict):
                return "dict"
            elif isinstance(variable, list):
                return "list"
            else:
                return type(variable).__name__
        print(core_papers)
        # print(check_variable_type(core_papers))
        # print(check_variable_type(all_papers))
        # Fix: ensure both are lists before concatenation
        all_processed_papers = core_papers + all_papers
        # print(all_processed_papers[0]['metadata'].keys())
        
        # Use core papers for section writing, but all papers for citations
        processed_papers = core_papers
        # print(processed_papers)
        if not processed_papers:
            return {"error": "No papers could be processed"}
        
        # Step 2: Write individual sections with self-reflection
        print("\n2. Writing literature review sections with self_reflection...")
        sections = {}
        print("\n3. Generating literature review outline...")
        
        outline_path = f"{self.save_dir}/survey_outline.json"
        # read outline file
        with open(outline_path, 'r', encoding='utf-8') as f:
            outline = json.load(f)
        section_title_list = [section['section_outline'].split("\n")[0].lstrip('# ').strip() for section in outline]
        sections = {section['section_outline'].split("\n")[0].lstrip('# ').strip(): section['section_outline'] for section in outline}
        section_definitions = {section['section_outline'].split("\n")[0].lstrip('# ').strip(): section['section_focus'] for section in outline}
        proof_ids = {section['section_outline'].split("\n")[0].lstrip('# ').strip(): section['proof_ids'] for section in outline}
        print(section_title_list)
        print(sections)

        ii = 0
        for section_title, section_focus in section_definitions.items():
            # using previous section if any
            if ii > 0:
                pre_section = sections[section_title_list[ii-1]]
            else:
                pre_section = ""
            checkpoint_dir = f"{self.save_dir}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            # make sure the section title is valid for filename
            checkpoint_path = os.path.join(checkpoint_dir, f"section_{section_title.replace(' ', '_').replace('/', '_').replace(':', '').lower()}_checkpoint.txt")
            # Load from checkpoint if exists
            if os.path.exists(checkpoint_path):
                print(f"  ⏭️  Loading section '{section_title}' from checkpoint.")
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    section_content = f.read()
            else:
                print(f"  ✍️  Writing section '{section_title}'...")
                print(f"Section outline: {sections[section_title]}")
                section_content = self.write_literature_review_section_with_reflection(
                    section_title, processed_papers, section_focus, sections[section_title], proof_ids[section_title], pre_section
                )
                # Save checkpoint for this section
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    f.write(section_content)
            sections[section_title] = section_content
            ii += 1
        
        
        # Step 3: Generate LaTeX document
        print("\n3. Generating LaTeX document...")
        latex_document = self.generate_latex_document_with_sections(sections, all_processed_papers, review_title)
        
        # Compile final review
        final_review = {
            "title": review_title,
            "papers_processed": len(processed_papers),
            "paper_list": [p['file_name'] for p in processed_papers],
            "citations_map": self.citations_map,
            "sections": sections,
            "latex_document": latex_document,
            "generation_date": datetime.now().isoformat(),
            "processed_papers_data": processed_papers
        }
        
        return final_review
    #endregion

    # region generate LaTeX bibliography
    def generate_latex_bibliography(self, processed_papers: List[Dict]) -> str:
        """Generate LaTeX bibliography entries"""
        bibliography = "\\begin{thebibliography}{" + str(len(processed_papers)) + "}\n\n"
        
        for paper in processed_papers:
            metadata = paper['metadata']
            citation_key = paper['citation_key']
            
            title = metadata.get('title', 'Unknown Title')
            authors = metadata.get('authors', 'Unknown Authors')
            year = metadata.get('published_date', 'Unknown Year')
            journal = metadata.get('venue', 'Unknown Venue')
            
            # Format bibliography entry
            bibliography += f"\\bibitem{{{citation_key}}}\n"
            bibliography += f"{authors}.\n"
            bibliography += f"\\textit{{{title}}}.\n"
            bibliography += f"{journal}, {year}.\n\n"
        
        bibliography += "\\end{thebibliography}\n"
        return bibliography
    #endregion
    
    #region generate LaTeX document
    def generate_latex_document(self, review_data: Dict, processed_papers: List[Dict], all_papers: List[Dict]) -> str:
        """Generate complete LaTeX document"""
        
        latex_content = f"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath,amsfonts,amssymb}}
\\usepackage{{graphicx}}
\\usepackage[margin=2.5cm]{{geometry}}
\\usepackage{{setspace}}
\\usepackage{{natbib}}
\\usepackage{{url}}
\\usepackage{{hyperref}}

\\title{{{review_data['title']}}}
\\author{{Literature Review}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\doublespacing

{review_data['content']}

\\singlespacing

{self.generate_latex_bibliography(all_papers)}

\\end{{document}}
"""
        return latex_content
    #endregion
    
    #region generate LaTeX document with sections
    def generate_latex_document_with_sections(self, sections: Dict, all_papers: List[Dict], 
                                        review_title: str = "Literature Review") -> str:
        """
        Generate complete LaTeX document with individual sections
        """
        
        # LaTeX document header (giữ nguyên)
        latex_header = r"""\documentclass[12pt,a4paper]{article}
    \usepackage[utf8]{inputenc}
    \usepackage[T1]{fontenc}
    \usepackage{amsmath,amsfonts,amssymb}
    \usepackage{graphicx}
    \usepackage[margin=2.5cm]{geometry}
    \usepackage{setspace}
    \usepackage{natbib}
    \usepackage{url}
    \usepackage{hyperref}
    \usepackage{booktabs}
    \usepackage{longtable}
    \usepackage{array}
    \usepackage{multirow}
    \usepackage{wrapfig}
    \usepackage{float}
    \usepackage{colortbl}
    \usepackage{pdflscape}
    \usepackage{tabu}
    \usepackage{threeparttable}
    \usepackage{threeparttablex}
    \usepackage[normalem]{ulem}
    \usepackage{makecell}
    \usepackage{xcolor}

    % Set line spacing
    \doublespacing

    % Configure hyperref
    \hypersetup{
        colorlinks=true,
        linkcolor=blue,
        filecolor=magenta,      
        urlcolor=cyan,
        citecolor=red,
    }

    % Title and author information
    \title{""" + review_title + r"""}
    \author{Literature Review}
    \date{\today}

    \begin{document}

    \maketitle

    % Abstract (optional)
    \begin{abstract}
    This literature review provides a comprehensive analysis of recent research in the field. The review synthesizes findings from """ + str(len(all_papers)) + r""" research papers, identifying key themes, methodological approaches, and future research directions.
    \end{abstract}

    \newpage
    \tableofcontents
    \newpage

    """

        # Generate main content sections (FIXED: Remove duplication)
        latex_content = ""
        
        # Define section order
        # section_order = [
        #     "Introduction",
        #     "Background and Related Work", 
        #     "Problem Classification and Taxonomy",
        #     "Methodological Approaches",
        #     "Current Challenges and Limitations",
        #     "Future Research Directions",
        #     "Conclusion"
        # ]
        section_order = list(sections.keys())
        print(section_order)
        # Track processed sections to avoid duplication
        processed_sections = set()
        
        # Add each section to the LaTeX document
        for section_title in section_order:
            if section_title in sections and section_title not in processed_sections:
                # Format section with proper LaTeX sectioning
                # latex_content += f"\\section{{{section_title}}}\n"
                latex_content += f"\\label{{sec:{section_title.lower().replace(' ', '_').replace('and', 'and')}}}\n\n"
                
                # Add section content
                section_content = sections[section_title]
                
                # Clean up and format the content
                formatted_content = self._format_latex_content(section_content)
                latex_content += formatted_content + "\n\n"
                
                # Mark as processed
                processed_sections.add(section_title)
        
        # Generate bibliography with ALL papers
        bibliography = self._generate_advanced_latex_bibliography(all_papers)
        
        # Combine all parts
        complete_latex = latex_header + latex_content + bibliography + "\n\\end{document}"
        
        return complete_latex
    #endregion

    #region format LaTeX content
    def _format_latex_content(self, content: str) -> str:
        """Format and clean content for LaTeX output"""
        # Remove any existing LaTeX document structure commands that might interfere
        content = re.sub(r'\\documentclass.*?\n', '', content)
        content = re.sub(r'\\begin\{document\}.*?\n', '', content)
        content = re.sub(r'\\end\{document\}.*?\n', '', content)
        content = re.sub(r'\\maketitle.*?\n', '', content)
        
        # Fix common formatting issues
        content = content.replace('&', '\\&')  # Escape ampersands
        content = content.replace('%', '\\%')  # Escape percent signs
        content = content.replace('_', '\\_')  # Escape underscores
        content = re.sub(r'(?<!\\)#', '\\#', content)  # Escape hash symbols
        
        # Improve paragraph formatting
        content = re.sub(r'\n\n+', '\n\n', content)  # Remove excessive line breaks
        
        # Format lists properly
        content = re.sub(r'^\s*[-•]\s+(.+)$', r'\\item \1', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(\d+)\.\s+(.+)$', r'\\item \2', content, flags=re.MULTILINE)
        
        # Handle emphasis and bold text
        content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)  # Bold
        content = re.sub(r'\*(.*?)\*', r'\\textit{\1}', content)      # Italic
        content = re.sub(r'`([^`]+)`', r'\\texttt{\1}', content)      # Code
        
        return content
    #endregion

    #region generate advanced LaTeX bibliography
    def _generate_advanced_latex_bibliography(self, processed_papers: List[Dict]) -> str:
        """Generate an advanced LaTeX bibliography with proper formatting"""
        bibliography = "\\newpage\n\\section*{References}\n\\addcontentsline{toc}{section}{References}\n\n"
        bibliography += "\\begin{thebibliography}{" + str(len(processed_papers)) + "}\n\n"
        
        for paper in processed_papers:
            metadata = paper['metadata']
            citation_key = paper['citation_key']
            # print("metadata: ", metadata)
            # Extract metadata with fallbacks
            title = metadata.get('title', 'Unknown Title').strip()
            # authors = metadata.get('authors', 'Unknown Authors').strip()
            authors = metadata.get('authors', 'Unknown Authors')
            if isinstance(authors, list):
                # If it's a list, join the authors
                authors = ', '.join([str(author).strip() for author in authors if author])
            elif isinstance(authors, str):
                authors = authors.strip()
            else:
                authors = str(authors).strip()
            year = metadata.get('published_date', 'Unknown Year').strip()
            journal = metadata.get('venue', 'Unknown Venue').strip()
            
            # Clean up the data
            if title == "Not available" or not title:
                title = f"Research Paper ({paper['file_name']})"
            if authors == "Not available" or not authors:
                authors = "Unknown Authors"
            if year == "Not available" or not year:
                year = "n.d."
            if journal == "Not available" or not journal:
                journal = "Unpublished manuscript"
            
            # Format bibliography entry
            bibliography += f"\\bibitem{{{citation_key}}}\n"
            
            # Format authors
            if ',' in authors and len(authors.split(',')) > 1:
                author_list = [author.strip() for author in authors.split(',')]
                if len(author_list) > 3:
                    formatted_authors = ', '.join(author_list[:3]) + ', et al.'
                else:
                    formatted_authors = ', '.join(author_list[:-1]) + ', and ' + author_list[-1]
            else:
                formatted_authors = authors
            
            bibliography += f"{formatted_authors} ({year}). "
            bibliography += f"\\textit{{{title}}}. "
            bibliography += f"{journal}.\n\n"
        
        bibliography += "\\end{thebibliography}\n"
        return bibliography

    #endregion
    
    # region save literature review
    def save_literature_review(self, review_data: Dict):
        """Save the literature review to files"""
        output_dir = self.save_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LaTeX file
        latex_file_path = os.path.join(output_dir, "literature_review.tex")
        with open(latex_file_path, 'w', encoding='utf-8') as f:
            f.write(review_data['latex_document'])
        
        # Save Markdown version
        markdown_content = f"# {review_data['title']}\n\n"
        markdown_content += f"**Generated on:** {review_data['generation_date']}\n"
        markdown_content += f"**Papers analyzed:** {review_data['papers_processed']}\n\n"
        
        markdown_content += "## Papers Included:\n"
        for i, paper in enumerate(review_data['paper_list'], 1):
            citation_key = review_data['citations_map'].get(paper, f"paper{i}")
            markdown_content += f"{i}. {paper} [{citation_key}]\n"
        
        markdown_content += "\n## Literature Review\n\n"
        
        # Add sections content
        for section_title, section_content in review_data['sections'].items():
            markdown_content += f"### {section_title}\n\n"
            # Convert LaTeX citations to markdown format
            content = re.sub(r'\\cite\{([^}]+)\}', r'[\1]', section_content)
            markdown_content += content + "\n\n"
        
        with open(os.path.join(output_dir, "literature_review.md"), 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Save as JSON for programmatic access
        with open(os.path.join(output_dir, "literature_review_data.json"), 'w', encoding='utf-8') as f:
            # Remove full_text to reduce file size
            clean_data = review_data.copy()
            for paper in clean_data.get('processed_papers_data', []):
                paper.pop('full_text', None)
            json.dump(clean_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nLiterature review saved to: {output_dir}/")
        print("Files created:")
        print("- literature_review.tex (LaTeX source)")
        print("- literature_review.md (Markdown version)")
        print("- literature_review_data.json (structured data)")
        print("\nTo compile LaTeX: pdflatex literature_review.tex")
    #endregion

# Example usage
def process_papers_from_directory():
    if len(sys.argv) != 2:
        print("Usage: python writing/writing_survey.py \"your research query\"")
        print("Example: python writing/writing_survey.py \"federated learning privacy\"")
        return
    query = sys.argv[1]
    load_dotenv(Path(".env"))
    API_KEY = os.getenv("API_KEY") 
    lit_review_gen = LiteratureReviewGenerator(query, API_KEY)
    
    # Get all PDF files from a directory
    papers_directory = f"paper_data/{query.replace(' ', '_').replace(':', '')}"  
 
    # Generate review with self-reflection
    review_data = lit_review_gen.generate_complete_literature_review(
        papers_directory,
        "A Comprehensive Literature Review with Self-Reflection"
    )
    
    # Save results
    if "error" not in review_data:
        lit_review_gen.save_literature_review(review_data)
        print("\nLiterature review with self-reflection completed!")
    else:
        print(f"Error: {review_data['error']}")

if __name__ == "__main__":
    process_papers_from_directory()