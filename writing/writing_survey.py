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
        self.advanced_model = genai.GenerativeModel('gemini-2.5-pro')
        self.papers_data = []
        self.citations_map = {}  # Map paper names to citation keys
        rag_db_path = f"paper_data/{query.replace(' ', '_').replace(':', '')}/rag_database"
        os.makedirs(f"paper_data/{query.replace(' ', '_').replace(':', '')}/rag_database/", exist_ok=True)
        self.keyword_dir = f"paper_data/{query.replace(' ', '_').replace(':', '')}/keywords"
        self.summarizer = PaperSummarizerRAG(query, api_key, rag_db_path)
        self.max_improvement_iterations = 3  # Maximum iterations for section improvement
        self.layer_method_group_json = json.load(open(f"paper_data/{self.query.replace(' ', '_').replace(':', '')}/paths/layer_method_group_summary.json", "r", encoding="utf-8"))
        self.develop_direction = json.load(open(f"paper_data/{self.query.replace(' ', '_').replace(':', '')}/paths/layer1_seed_taxonomy.json", "r", encoding="utf-8"))
        self.community_summary = json.load(open(f"paper_data/{self.query.replace(' ', '_').replace(':', '')}/paths/communities_summary.json", "r", encoding="utf-8"))
        self.graph_path = f"paper_data/{self.query.replace(' ', '_').replace(':', '')}/info/paper_citation_graph.json"
        self.node_info_path = f"{self.keyword_dir}/processed_checkpoint.json"
        self.G = self.load_graph(self.graph_path, self.node_info_path)
    def get_paper_text(self, paper_infos):
        papers_text = ''
        for info in paper_infos:
            try:
                if info['summary'] != info['abstract'] and info['summary'] != '':
                    papers_text += f"[{info['citation_key']}] {info['title']} ({info['year']})\nSummary: {info['summary']}\n\n"
            except: papers_text += f"[{info['citation_key']}] {info['title']} ({info['year']})\nSummary: {info['abstract']}\n\n"
        return papers_text
    def load_graph(self, json_path, node_info_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(node_info_path, 'r', encoding='utf-8') as f:
            node_info = json.load(f)
        id2node_info = {}
        for node in node_info:
            node_id = node['file_name'].replace('.pdf', '')
            info = node["metadata"]
            info['citation_key'] = node['citation_key']
            id2node_info[node_id] = info
        G = nx.DiGraph()
        for node in data['nodes']:
            node_id = node['id']
            info = {**node, **id2node_info[node_id]}
            G.add_node(node_id, **info)
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
            response = self.advanced_model.generate_content(
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
    
    def filter_additional_papers_with_llm_check(self, queries: List[str], subsection_title, subsection_focus, current_content, weaknesses):
        prompt = self.prompt_helper.generate_prompt(self.prompt_helper.CHECK_RAG_RESULT_PROMPT,
                                                    paras={
                                                        "SUBSECTION_TITLE": subsection_title,
                                                        "SUBSECTION_FOCUS": subsection_focus,
                                                        "CURRENT_CONTENT": current_content,
                                                        "WEAKNESSES": weaknesses,
                                                        "RETRIEVED_PAPERS": retrieved_papers
                                                    })
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
                                                                    "SYNTHESIS_SCORE": {evaluation_feedback.get('individual_scores', {}).get('synthesis_quality', 'N/A')},
                                                                    "CRITICAL_SCORE": {evaluation_feedback.get('individual_scores', {}).get('critical_analysis', 'N/A')},
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
    def get_proofs_text(self, proof_ids: list, get_nodes_info: bool) -> str:
        """
        get proofs text (including paper title, abstract, year from the graph attributes)
        """
        all_nodes_id = []
        all_nodes = []
        paper_summaries = ''
        development_direction = ''
        community_summary = ''
        for proof in proof_ids:
            try: 
                proof = proof.lower().strip()
                if 'layer' in proof:
                    proof.replace('layer', '').strip()
            except:
                pass
            if proof in self.develop_direction.keys():
                direction_dict = self.develop_direction[proof]
                development_direction += 'Development direction: \n'
                development_direction += direction_dict.get('summary', '') + '\n'
                nodes = [proof]
                nodes.extend(direction_dict.get('paths', []))
                all_nodes_id.extend(nodes)
                for node in nodes:
                    if node in self.G.nodes and get_nodes_info:
                        node_data = self.G.nodes[node]
                        all_nodes.extend([node_data])
                        # title = node_data.get('title', '')
                        # abstract = node_data.get('abstract', '')
                        # year = node_data.get('year', '')
                        # summary = node_data.get('summary', '')
                        # paper_summaries += self.get_paper_text(node_data)
            if proof in [1, 2, 3, '1', '2', '3', 'layer_1', 'layer_2', 'layer_3']:
                community_summary += 'Layer 1 summaries: \n'
                layer_dict = self.layer_method_group_json.get(f"{proof}", {})
                community_summary += layer_dict.get('summary', '') + '\n'
                nodes = layer_dict.get('papers', [])
                all_nodes_id.extend(nodes)
                for paper in nodes:
                    if paper in self.G.nodes and get_nodes_info:
                        node_data = self.G.nodes[paper]
                        all_nodes.extend([node_data])
                        # title = node_data.get('title', '')
                        # abstract = node_data.get('abstract', '')
                        # year = node_data.get('year', '')
                        # summary = node_data.get('summary', '')
                        # paper_summaries += f"Title: {title}\n"
                        # paper_summaries += f"Abstract: {abstract}\n"
                        # paper_summaries += f"Year: {year}\n"
                        # paper_summaries += f"Summary: {summary}\n\n"
            if proof in self.community_summary.keys():
                community_dict = self.community_summary[proof]
                community_summary += 'Community summaries: \n'
                community_summary += community_dict.get('summary', '') + '\n'
                nodes = community_dict.get('papers', [])
                all_nodes_id.extend(nodes)
                for node in nodes:
                    if node in self.G.nodes and get_nodes_info:
                        node_data = self.G.nodes[node]
                        all_nodes.extend([node_data])
                        # title = node_data.get('title', '')
                        # abstract = node_data.get('abstract', '')
                        # year = node_data.get('year', '')
                        # summary = node_data.get('summary', '')
                        # paper_summaries += f"Title: {title}\n"
                        # paper_summaries += f"Abstract: {abstract}\n"
                        # paper_summaries += f"Year: {year}\n"
                        # paper_summaries += f"Summary: {summary}\n\n"

        paper_summaries = self.get_paper_text(all_nodes)
        return community_summary, development_direction, all_nodes_id, paper_summaries
    def write_initial_subsection(self, subsection_title: str, subsection_focus: str, 
                               section_outline: str, proof_ids: list, pre_subsection: str,
                               processed_papers: List[Dict]) -> str:
        """
        Write initial version of a literature review subsection
        """
        # # Step 1: Prepare citations info
        # citations_info = ""
        # for paper in processed_papers:
        #     metadata = paper['metadata']
        #     citations_info += f"\\cite{{{paper['citation_key']}}}: {metadata.get('title', 'N/A')} by {metadata.get('authors', 'N/A')} ({metadata.get('published_date', 'N/A')})\n"
        
        # # Step 2: Get papers summary
        # paper_summaries = ""
        # for i, paper in enumerate(processed_papers, 1):
        #     summary = paper.get('summary', '')
        #     paper_summaries += f"**Paper {i} ({paper['citation_key']})**: {paper['file_name']}\n{summary}\n\n"
        
        # Step 3: Get proofs text
        community_summary, development_direction, proof_papers, paper_summaries = self.get_proofs_text(proof_ids, get_nodes_info=True)
        # Step 4: Generate initial subsection content
        subsection_prompt = self.prompt_helper.generate_prompt(
            self.prompt_helper.WRITE_INITIAL_SUBSECTION_PROMPT,
            paras={
                'SUBSECTION_TITLE': subsection_title,
                'SUBSECTION_FOCUS': subsection_focus,
                'PAPER_INFO': paper_summaries,
                'COMMUNITY_SUMMARY': community_summary,
                'DEVELOPMENT_DIRECTION': development_direction,
            }
        )
        
        try:
            response = self.model.generate_content(
                subsection_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=102400,
                    temperature=0.4,
                )
            )
            return response.text
        except Exception as e:
            print(f"Error writing initial subsection: {e}")
            return ""
    def evaluate_subsection_quality(self, subsection_title: str, subsection_content: str, 
                                 subsection_focus: str, outline: str, pre_subsection: str) -> Dict[str, any]:
        """
        Evaluate the quality of a literature review subsection
        """
        evaluation_prompt = self.prompt_helper.generate_prompt(
            self.prompt_helper.EVALUATE_SUBSECTION_PROMPT,
            paras={
                'PRE_SUBSECTION': pre_subsection,
                'SUBSECTION_TITLE': subsection_title,
                'SUBSECTION_FOCUS': subsection_focus,
                'OUTLINE': outline,
                'SUBSECTION_CONTENT': subsection_content
            }
        )
        
        try:
            response = self.advanced_model.generate_content(
                evaluation_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=102400,
                    temperature=0.2,
                )
            )
            
            # Clean up response text
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json\n', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```\n', '').replace('```', '').strip()
            
            # Parse JSON response
            evaluation_result = json.loads(response_text)
            return evaluation_result
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error in subsection evaluation: {e}")
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
            print(f"Error evaluating subsection: {e}")
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
    def improve_subsection_with_additional_papers(self, subsection_title: str, current_content: str,
                                                subsection_focus: str, additional_papers: List[Dict],
                                                outline: str,
                                                evaluation_feedback: Dict, 
                                                pre_subsection_content: str) -> str:
        """
        Improve a subsection by incorporating additional papers and addressing feedback
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
        
        improvement_prompt = self.prompt_helper.generate_prompt(
            self.prompt_helper.SUBSECTION_IMPROVE_PROMPT,
            paras={
                "SUBSECTION_TITLE": subsection_title,
                "SUBSECTION_FOCUS": subsection_focus,
                "PRE_SUBSECTION": pre_subsection_content,
                "OUTLINE": outline,
                "CURRENT_CONTENT": current_content,
                "OVERALL_SCORE": {evaluation_feedback.get('overall_score', 'N/A')},
                "SYNTHESIS_SCORE": {evaluation_feedback.get('individual_scores', {}).get('synthesis_quality', 'N/A')},
                "CRITICAL_SCORE": {evaluation_feedback.get('individual_scores', {}).get('critical_analysis', 'N/A')},
                "STRENGTH": {', '.join(evaluation_feedback.get('strengths', []))},
                "WEAKNESS": {', '.join(evaluation_feedback.get('weaknesses', []))},
                "IMPROVEMENT_NEEDED": {', '.join(evaluation_feedback.get('improvement_needed', []))},
                "ADDITIONAL_INFO": additional_info
            }
        )
        
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
            print(f"Error improving subsection: {e}")
            return current_content
    
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
    def write_initial_section_overview(self, section_title: str, section_focus: str, 
                                       full_outline_text: str, proof_ids: list, pre_section_content: str,
                                       processed_papers: List[Dict]) -> str:
        """
        Write the initial overview for a literature review section. This is a high-level summary
        before diving into subsections.
        """
        # citations_info = ""
        # for paper in processed_papers:
        #     metadata = paper['metadata']
        #     citations_info += f"\\cite{{{paper['citation_key']}}}: {metadata.get('title', 'N/A')} by {metadata.get('authors', 'N/A')} ({metadata.get('published_date', 'N/A')})\n"
        
        # papers_summary = ""
        # for i, paper in enumerate(processed_papers, 1):
        #     summary = paper.get('summary', '')
        #     papers_summary += f"**Paper {i} ({paper['citation_key']})**: {paper['file_name']}\n{summary}\n\n"
        
        proofs_text, proof_papers, paper_summaries = self.get_proofs_text(proof_ids, get_nodes_info=True)
        citations_info = ""
        for paper in proof_papers:
            metadata = paper['metadata']
            citations_info += f"\\cite{{{paper['citation_key']}}}: {metadata.get('title', 'N/A')} by {metadata.get('authors', 'N/A')} ({metadata.get('published_date', 'N/A')})\n"
        
        section_overview_prompt = self.prompt_helper.generate_prompt(
            self.prompt_helper.WRITE_INITIAL_SECTION_OVERVIEW_PROMPT,
            paras={
                'SECTION_TITLE': section_title,
                'SECTION_FOCUS': section_focus,
                'PROOFS_TEXT': proofs_text,
                'CITATION_INFO': citations_info,
                'OUTLINE': full_outline_text,
                'PAPERS_SUMMARY': paper_summaries,
                'PRE_SECTION': pre_section_content
            }
        )
        
        try:
            response = self.model.generate_content(
                section_overview_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=102400,
                    temperature=0.4,
                )
            )
            return response.text
        except Exception as e:
            print(f"Error writing initial section overview: {e}")
            return f"Error: Could not generate overview for section '{section_title}' due to {e}"
    #region write literature review section with reflection
    def write_literature_review_section_with_reflection(self, section_data: Dict, 
                                                      processed_papers: List[Dict],
                                                      full_outline_text: str, 
                                                      pre_section_content: str) -> Tuple[str, Dict[str, str]]:
        """
        Write a literature review section with self-reflection and iterative improvement,
        including all its subsections.
        
        Args:
            section_data (Dict): Contains section_number, section_title, section_focus, subsections.
            processed_papers (List[Dict]): List of processed papers.
            full_outline_text (str): The complete JSON outline as a string for context.
            pre_section_content (str): Content of the previous main section, if any.
            
        Returns:
            Tuple[str, Dict[str, str]]: Final improved section content (including all subsections)
                                       and a dictionary of subsection contents.
        """
        
        section_number = section_data['section_number']
        section_title = section_data['section_title']
        section_focus = section_data['section_focus']
        section_proof_ids = section_data.get('proof_ids', []) # Top-level proof IDs for the section
        subsections_data = section_data['subsections']
        
        print(f"   Writing Section {section_number}: {section_title}")
        
        # # --- 1. Write and refine section overview ---
        # section_overview_checkpoint_path = os.path.join(
        #     self.save_dir, 
        #     f"section_{section_number.replace('.', '_')}_{section_title.replace(' ', '_').replace('/', '_').replace(':', '').lower()}_overview_checkpoint.tex"
        # )

        # section_overview_content = ""
        # if os.path.exists(section_overview_checkpoint_path):
        #     print(f"     ⏭️  Loading section overview for '{section_title}' from checkpoint.")
        #     with open(section_overview_checkpoint_path, 'r', encoding='utf-8') as f:
        #         section_overview_content = f.read()
        # else:
        #     print(f"     ✍️  Writing initial overview for section '{section_title}'...")
        #     section_overview_content = self.write_initial_section_overview(
        #         section_title, section_focus, full_outline_text, section_proof_ids, pre_section_content, processed_papers
        #     )
        #     with open(section_overview_checkpoint_path, 'w', encoding='utf-8') as f:
        #         f.write(section_overview_content)
        
        # --- 2. Write and refine each subsection ---
        section_overview_content = ''
        all_subsections_content_for_section = {}
        if not section_overview_content.strip().startswith("\\section"):
            full_section_latex_content = f"\\section{{{section_title}}}\n"
        else:
            full_section_latex_content = ''
        full_section_latex_content += f"\\label{{sec:{section_title.lower().replace(' ', '_').replace('and', '_and_')}}}\n\n"
        full_section_latex_content += section_overview_content + "\n\n" # Add the refined overview
        
        pre_subsection_content = "" # To store content of the previous subsection

        for subsection_data in subsections_data:
            subsection_number = subsection_data['number']
            subsection_title = subsection_data['title']
            subsection_focus = subsection_data['subsection_focus']
            subsection_proof_ids = subsection_data.get('proof_ids', [])
            
            print(f"      Writing Subsection {subsection_number}: {subsection_title}")

            checkpoint_path = os.path.join(
                self.save_dir, 
                f"subsection_{subsection_number.replace('.', '_')}_{subsection_title.replace(' ', '_').replace('/', '_').replace(':', '').lower()}_checkpoint.tex"
            )
            
            current_subsection_content = ""
            if os.path.exists(checkpoint_path):
                print(f"        ⏭️  Loading subsection '{subsection_title}' from checkpoint.")
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    current_subsection_content = f.read()
            else:
                print(f"        ✍️  Writing initial content for subsection '{subsection_title}'...")
                current_subsection_content = self.write_initial_subsection(
                    subsection_title, subsection_focus, full_outline_text, subsection_proof_ids, 
                    pre_subsection_content, processed_papers
                )
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    f.write(current_subsection_content)

            iteration = 0
            while iteration < self.max_improvement_iterations:
                iteration += 1
                print(f"         Iteration {iteration}: Evaluating subsection quality...")
                
                evaluation = self.evaluate_subsection_quality(
                    subsection_title, current_subsection_content, subsection_focus, 
                    full_outline_text, pre_subsection_content
                )
                
                print(f"         Overall Score: {evaluation.get('overall_score', 'N/A')}/5")
                
                if evaluation.get('is_satisfactory', False):
                    print(f"         Subsection meets quality standards!")
                    break
                
                suggested_queries = evaluation.get('suggested_queries', [])
                if suggested_queries:
                    print(f"         Retrieving additional papers for subsection: {', '.join(suggested_queries[:2])}")
                    additional_papers = self.retrieve_additional_papers(suggested_queries[:2])
                    print(f"         Found {len(additional_papers)} additional papers for subsection")
                else:
                    additional_papers = []
                
                print(f"         Improving subsection based on feedback...")
                current_subsection_content = self.improve_subsection_with_additional_papers(
                    subsection_title, current_subsection_content, subsection_focus, 
                    additional_papers, full_outline_text, evaluation, pre_subsection_content
                )
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    f.write(current_subsection_content)

            # After refining, add the subsection content to the section's full content
            all_subsections_content_for_section[subsection_title] = current_subsection_content
            if current_subsection_content.strip().startswith("\\section"):
                current_subsection_content = current_subsection_content.replace("\\section", "\\subsection")
            
            subsection_prefix = f"\\subsection"
            if current_subsection_content.strip().startswith(subsection_prefix):
                # Find the end of the subsection line and label to remove it
                # This assumes the label immediately follows the subsection title
                # and we want to remove both if they are at the very beginning.
                
                # A more robust way would be to parse it, but for simple string manipulation:
                temp_content = current_subsection_content.strip()
                
                # Find the first newline after the potential subsection line
                first_newline_idx = temp_content.find('\n')
                
                if first_newline_idx != -1:
                    # Check if the text before this newline contains the label as well
                    # This is a bit simplistic and might need refinement depending on exact latex structure
                    if "\\label" in temp_content[:first_newline_idx]:
                        # If the label is on the same line as subsection or immediately after
                        # find the end of the label line
                        second_newline_idx = temp_content.find('\n', first_newline_idx + 1)
                        if second_newline_idx != -1:
                            current_subsection_content = temp_content[second_newline_idx + 1:]
                        else:
                            current_subsection_content = "" # Only the subsection and label were present
                    else: # Only the subsection line
                        current_subsection_content = temp_content[first_newline_idx + 1:]
                else: # No newline, meaning only the subsection was in the content
                    current_subsection_content = ""

            # Now, add the correct subsection line at the beginning
            full_section_latex_content += f"\\subsection{{{subsection_title}}}\n\\label{{sec:{subsection_number.replace('.', '_')}_{subsection_title.lower().replace(' ', '_').replace('and', '_and_')}}}\n\n"
            full_section_latex_content += current_subsection_content + '\n'

            # full_section_latex_content += current_subsection_content + "\n\n"
            # pre_subsection_content = current_subsection_content # Update for the next iteration

        return full_section_latex_content, all_subsections_content_for_section
    
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
    def generate_complete_literature_review(self, paper_paths: str, 
                                          review_title: str = "Literature Review") -> Dict:
        """
        Generate a complete literature review from multiple papers with self-reflection
        """
        print("Starting literature review generation...")
        
        # Step 1: Process all papers
        print("\n1. Processing papers...")

        # process core papers first for drafting (separate folder)
        core_papers_path = os.path.join(paper_paths)
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

        print(f"Processed {len(core_papers)} core papers.")
        print(f"Processed {len(all_papers)} all papers (including core papers).")
        
        all_processed_papers = all_papers # This will be used for bibliography
        processed_papers_for_content = core_papers # Use core papers for detailed content generation

        if not processed_papers_for_content:
            return {"error": "No core papers could be processed for content generation."}
        
        # Step 2: Load the JSON outline
        print("\n2. Loading literature review outline...")
        outline_path = f"{self.save_dir}/survey_outline.json"
        
        if not os.path.exists(outline_path):
            return {"error": f"Outline file not found at {outline_path}. Please generate the outline first."}

        with open(outline_path, 'r', encoding='utf-8') as f:
            outline_json = json.load(f)
        
        # Convert the JSON outline to a string for passing as context to prompts
        full_outline_text = json.dumps(outline_json, indent=2)

        # Step 3: Write individual sections (including subsections) with self-reflection
        print("\n3. Writing literature review sections and subsections with self-reflection...")
        
        sections_content = {} # Stores the final content of each main section
        all_subsection_contents = {} # Stores content of all subsections, keyed by full title (e.g., "1.1 Background: ...")

        # Iterate through main sections
        pre_section_content = "" # Content of the previous main section

        for section_data in outline_json:
            section_number = section_data['section_number']
            section_title = section_data['section_title']

            # Call the updated function to write/refine the section and its subsections
            full_section_latex_content, subsections_in_this_section = self.write_literature_review_section_with_reflection(
                section_data, processed_papers_for_content, full_outline_text, pre_section_content
            )
            
            sections_content[section_title] = full_section_latex_content
            all_subsection_contents.update(subsections_in_this_section) # Add all subsections from this section
            pre_section_content = full_section_latex_content # Update for the next main section

        # Step 4: Generate LaTeX document
        print("\n4. Generating LaTeX document...")
        latex_document = self.generate_latex_document_with_sections(
            sections_content, all_processed_papers, review_title
        )
        
        # Compile final review
        final_review = {
            "title": review_title,
            "papers_processed": len(all_processed_papers),
            "paper_list": [p['file_name'] for p in all_processed_papers],
            "citations_map": {p['file_name']: p['citation_key'] for p in all_processed_papers}, # Ensure this map is populated
            "sections": sections_content, # This now contains main section content including subsections
            "subsections": all_subsection_contents, # Explicitly store subsection content
            "latex_document": latex_document,
            "generation_date": datetime.now().isoformat(),
            "processed_papers_data": all_processed_papers
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
#     def generate_latex_document_with_sections(self, sections: Dict[str, str], all_papers: List[Dict], 
#                                         review_title: str = "Literature Review") -> str:
#         """
#         Generate complete LaTeX document with individual sections and subsections.
        
#         Args:
#             sections (Dict[str, str]): A dictionary where keys are section titles 
#                                         and values are the full LaTeX content of each section
#                                         (including its subsections).
#             all_papers (List[Dict]): List of all processed papers for bibliography generation.
#             review_title (str): The title of the entire literature review document.
            
#         Returns:
#             str: The complete LaTeX document as a string.
#         """
        
#         # LaTeX document header
#         latex_header = r"""\documentclass[12pt,a4paper]{article}
# \usepackage[utf8]{inputenc}
# \usepackage[T1]{fontenc}
# \usepackage{amsmath,amsfonts,amssymb}
# \usepackage{graphicx}
# \usepackage[margin=2.5cm]{geometry}
# \usepackage{setspace}
# \usepackage{natbib}
# \usepackage{url}
# \usepackage{hyperref}
# \usepackage{booktabs}
# \usepackage{longtable}
# \usepackage{array}
# \usepackage{multirow}
# \usepackage{wrapfig}
# \usepackage{float}
# \usepackage{colortbl}
# \usepackage{pdflscape}
# \usepackage{tabu}
# \usepackage{threeparttable}
# \usepackage{threeparttablex}
# \usepackage[normalem]{ulem}
# \usepackage{makecell}
# \usepackage{xcolor}

# % Set line spacing
# \doublespacing

# % Configure hyperref
# \hypersetup{
#     colorlinks=true,
#     linkcolor=blue,
#     filecolor=magenta,      
#     urlcolor=cyan,
#     citecolor=red,
# }

# % Title and author information
# \title{""" + review_title + r"""}
# \author{Literature Review}
# \date{\today}

# \begin{document}

# \maketitle

# % Abstract (optional)
# \begin{abstract}
# This literature review provides a comprehensive analysis of recent research in the field of """ + self.query + r""". The review synthesizes findings from """ + str(len(all_papers)) + r""" research papers, identifying key themes, methodological approaches, and future research directions.
# \end{abstract}

# \newpage
# \tableofcontents
# \newpage

# """
#         # Generate main content sections
#         latex_content = ""
        
#         # The 'sections' dictionary already contains the full LaTeX content for each section,
#         # which includes its overview and all subsections. We just need to concatenate them.
#         for section_title, section_body_latex in sections.items():
#             # The section_body_latex already contains \section{} and \subsection{} commands
#             # generated during the writing process.
#             latex_content += section_body_latex + "\n\n"
        
#         # Generate bibliography with ALL papers
#         bibliography = self._generate_advanced_latex_bibliography(all_papers)
        
#         # Combine all parts
#         complete_latex = latex_header + latex_content + bibliography + "\n\\end{document}"
        
#         return complete_latex
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