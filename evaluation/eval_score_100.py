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
from src.models.LLM.ChatAgent import ChatAgent


prompt_for_content = {
    "coverage": """Here is an academic survey about the topic "{topic}": --- {content} --- <instruction> Please evaluate this survey about the topic {topic} based on the criterion above provided below, and give a score from 0 to 100 according to the score description: --- Criterion Description: Coverage: Coverage assesses the extent to which the survey encapsulates all relevant aspects of the topic, ensuring comprehensive discussion on both central and peripheral topics. --- Score 0-20 Description: The survey has very limited coverage, only touching on a small portion of the topic and lacking discussion on key areas. Score 21-40 Description: The survey covers some parts of the topic but has noticeable omissions, with significant areas either underrepresented or missing. Score 41-60 Description: The survey is generally comprehensive in coverage but still misses a few key points that are not fully discussed. Score 61-80 Description: The survey covers most key areas of the topic comprehensively, with only very minor topics left out. Score 81-100 Description: The survey comprehensively covers all key and peripheral topics, providing detailed discussions and extensive information. --- Return the score without any other information: """,
    
    "structure": """Here is an academic survey about the topic "{topic}": --- {content} --- <instruction> Please evaluate this survey about the topic {topic} based on the criterion above provided below, and give a score from 0 to 100 according to the score description: --- Criterion Description: Structure: Structure evaluates the logical organization and coherence of sections and subsections, ensuring that they are logically connected. --- Score 0-20 Description: The survey lacks logic, with no clear connections between sections, making it difficult to understand the overall framework. Score 21-40 Description: The survey has weak logical flow with some content arranged in a disordered or unreasonable manner. Score 41-60 Description: The survey has a generally reasonable logical structure, with most content arranged orderly, though some links and transitions could be improved such as repeated subsections. Score 61-80 Description: The survey has good logical consistency, with content well arranged and natural transitions, only slightly rigid in a few parts. Score 81-100 Description: The survey is tightly structured and logically clear, with all sections and content arranged most reasonably, and transitions between adjacent sections smooth without redundancy. --- Return the score without any other information: """,
    
    "relevance": """Here is an academic survey about the topic "{topic}": --- {content} --- <instruction> Please evaluate this survey about the topic {topic} based on the criterion above provided below, and give a score from 0 to 100 according to the score description: --- Criterion Description: Relevance: Relevance measures how well the content of the survey aligns with the research topic and maintain a clear focus. --- Score 0-20 Description: The content is outdated or unrelated to the field it purports to review, offering no alignment with the topic Score 21-40 Description: The survey is somewhat on topic but with several digressions; the core subject is evident but not consistently adhered to. Score 41-60 Description: The survey is generally on topic, despite a few unrelated details. Score 61-80 Description: The survey is mostly on topic and focused; the narrative has a consistent relevance to the core subject with infrequent digressions. Score 81-100 Description: The survey is exceptionally focused and entirely on topic; the article is tightly centered on the subject, with every piece of information contributing to a comprehensive understanding of the topic. --- Return the score without any other information:"""
}

new_content_prompt = {
    "synthesis": """Here is an academic survey about the topic "{topic}": --- {content} --- <instruction> Please evaluate this survey about the topic {topic} based on the criterion above provided below, and give a score from 0 to 100 according to the score description: --- Criterion Description: Synthesis: Synthesis evaluates the ability to interconnect disparate studies, identify overarching patterns or contradictions, and construct a cohesive intellectual framework beyond individual summaries. --- Score 0-20 Description: The survey is purely a collection of isolated study summaries with no attempt to connect ideas or identify broader trends. Score 21-40 Description: The survey occasionally links studies but fails to synthesize them into meaningful patterns; connections are superficial. Score 41-60 Description: The survey identifies some thematic relationships between studies but lacks a unified framework to explain their significance. Score 61-80 Description: The survey synthesizes most studies into coherent themes or debates, though some connections remain underdeveloped. Score 81-100 Description: The survey masterfully integrates studies into a novel framework, revealing latent trends, resolving contradictions, and proposing paradigm-shifting perspectives. --- Return the score without any other information:""",
    
    "critical_analysis": """Here is an academic survey about the topic "{topic}": --- {content} --- <instruction> Please evaluate this survey about the topic {topic} based on the criterion above provided below, and give a score from 0 to 100 according to the score description: --- Criterion Description: Critical Analysis: Critical Analysis examines the depth of critique applied to existing studies, including identification of methodological limitations, theoretical inconsistencies, and research gaps. --- Score 0-20 Description: The survey merely lists existing studies without any analytical commentary or critique. Score 21-40 Description: The survey occasionally mentions limitations of studies but lacks systematic analysis or synthesis of gaps. Score 41-60 Description: The survey provides sporadic critical evaluations of some studies, but the critique is shallow or inconsistent. Score 61-80 Description: The survey systematically critiques most key studies and identifies research gaps, though some areas lack depth. Score 81-100 Description: The survey demonstrates rigorous critical analysis of methodologies and theories, clearly maps research frontiers, and proposes novel directions based on synthesized gaps. --- Return the score without any other information:"""
}

TOPICS = ["A survey on Visual Transformer", "Hallucination in Large Language Models", "Graph Neural Networks", "Retrieval-Augmented Generation for Large Language Models", "knowledge graph embedding"] #, "Deep Meta-Learning", "Out-of-Distribution Detection", "reinforcement learning for language processing", "Exploration Methods in Reinforcement Learning", "Stabilizing Generative Adversarial Networks"]
TOPICS = ["Graph Neural Networks", "Retrieval-Augmented Generation for Large Language Models"]
class EvaluateSurvey: 
    def __init__(self, ablation_study = ''):
        self.ablation_study = ablation_study
        self.chat_agent = ChatAgent()
        self.GPT_MODEL = "gpt-4o-mini"
        self.GEMINI_MODEL = "gemini-2.5-flash"
        self.all_prompts = {**prompt_for_content, **new_content_prompt}
        self.TOPICS = TOPICS
        self.eval_result_dir= f"eval_results/{self.ablation_study}"
        os.makedirs(self.eval_result_dir, exist_ok=True)
    def eval(self, query: str, model: str, provider: str) -> Dict[str, int]:
        """
        Reads the survey content, evaluates it against all criteria using the LLM, 
        and returns a dictionary of scores.

        Args:
            query: The topic of the survey (e.g., "Hallucination in Large Language Models").
            model: The LLM model to use for evaluation (e.g., "gemini-2.5-flash").
            provider: The provider of the LLM (e.g., "google" or "openai").
        
        Returns:
            A dictionary where keys are criterion names and values are the scores (int).
        """
        topic = query
        # 1. Determine file path and read survey content
        # Clean the query for use in a directory name
        safe_query = query.replace(' ', '_').replace(':', '').replace('/', '_')
        save_dir = Path(f"paper_data/{safe_query}/literature_review_output{self.ablation_study}")
        survey_file = save_dir / "literature_review.md"

        try:
            with open(survey_file, 'r', encoding='utf-8') as f:
                survey_content = f.read()
            if not survey_content.strip():
                 raise ValueError("Survey file is empty.")
        except FileNotFoundError:
            print(f"Error: Survey file not found at {survey_file}")
            return {}
        except Exception as e:
            print(f"Error reading survey file: {e}")
            return {}

        evaluation_results: Dict[str, int] = {}

        # 2. & 3. Iterate through all evaluation criteria and format the prompt
        for criterion, prompt_template in self.all_prompts.items():
            print(f"Evaluating criterion: {criterion}...")
            
            # Format the specific prompt for the current criterion
            prompt = prompt_template.format(topic=topic, content=survey_content)
            
            # 4. Call the chat_agent to get the score
            try:
                # The chat method should return a string, which is expected to be just the score.
                llm_output = self.chat_agent.chat(prompt, temperature=0.2, model=model, provider=provider)
            except Exception as e:
                print(f"Error during LLM call for {criterion}: {e}")
                llm_output = "" # Set to empty string to trigger parsing failure

            # 5. Parse the output to get the score
            score = self._parse_score(llm_output)
            evaluation_results[criterion] = score
            print(f"  -> Score for {criterion}: {score}")

        # 6. Optional: Save results to a file
        results_file = f"{self.eval_result_dir}/evaluation_scores_{safe_query}_{provider}.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=4)
            print(f"\nEvaluation complete. Results saved to {results_file}")
            json_save_success = True
        except Exception as e:
            print(f"Error saving results: {e}")
        results_file_txt = f"{self.eval_result_dir}/evaluation_scores_{model.replace('-', '_')}_{provider}.txt"
        try:
            with open(results_file_txt, 'w', encoding='utf-8') as f:
                f.write(f"--- Evaluation Results for Topic: {topic} ---\n")
                f.write(f"Model: {model} | Provider: {provider}\n")
                f.write("-" * 40 + "\n")
                
                # Write results in a clean, human-readable format
                for criterion, score in evaluation_results.items():
                    f.write(f"{criterion.capitalize().replace('_', ' '):<20}: {score}\n")
                    
                if not json_save_success:
                     f.write("\n\nNote: JSON save failed, these TXT results serve as the primary output.\n")
                     
            print(f"Evaluation scores saved to TXT: {results_file_txt}")
        except Exception as e:
            print(f"Critical Error: Failed to save results to TXT file: {e}")

        return evaluation_results

    def _parse_score(self, llm_output: str) -> int:
        """
        Attempts to extract a single integer score from the LLM's text output.
        The prompts instruct the LLM to return *only* the score.
        
        Args:
            llm_output: The text returned by the LLM.
            
        Returns:
            The extracted score as an integer, or -1 if parsing fails.
        """
        # Remove any non-digit characters and whitespace, then try to convert to int.
        # This handles cases where the LLM might return 'Score: 85' or '85.'
        cleaned_output = re.sub(r'\D', '', llm_output).strip()
        
        if not cleaned_output:
            print(f"Warning: Could not extract a score from LLM output: '{llm_output}'")
            return -1 # Use -1 to indicate an error/failure
        
        try:
            # Take the first sequence of digits found
            return int(cleaned_output)
        except ValueError:
            print(f"Warning: Failed to convert cleaned output '{cleaned_output}' to integer.")
            return -1
    def evaluate_all_topics(self):
        for topic in self.TOPICS:
            print("evaluating topic: ", topic)
            print("using: gemini")
            print("ablation: ", self.ablation_study)
            evaluation_results_gemini = self.eval(topic, self.GEMINI_MODEL, 'gemini')
        # for topic in self.TOPICS:
        #     print("evaluating topic: ", topic)
        #     print("using: gpt")
        #     print("ablation: ", self.ablation_study)
        #     evaluation_results_gpt = self.eval(topic, self.GPT_MODEL, 'gpt')

def main():
    evaluate = EvaluateSurvey(ablation_study='')
    evaluate.evaluate_all_topics()

if __name__ == "__main__":
    main()