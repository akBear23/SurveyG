#!/usr/bin/env python3
"""
Survey-Optimized Paper Crawler

A specialized crawler designed specifically for LLM-generated surveys that ensures:
1. Comprehensive coverage: foundational + recent + trending papers
2. Quality filtering: citation-aware with temporal adjustments  
3. Diverse sources: Semantic Scholar (primary source for comprehensive coverage)
4. Survey-specific metrics: importance scoring, tier classification
5. Reference following: builds complete research lineage

Key improvements:
- Multi-tier collection strategy 
- Smart deduplication and quality scoring
- Enhanced metadata for survey writing
- Comprehensive source integration
- Citation velocity analysis for trending papers
"""

import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import time
import requests
import json
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
import re
import math
import hashlib
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

load_dotenv(Path(".env"))
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY") 
from dataclasses import dataclass
from src.models.LLM.ChatAgent import ChatAgent

@dataclass
class SearchKeywords:
    primary_keywords: List[str]
    secondary_keywords: List[str]
    boolean_queries: List[str]
    synonyms: Dict[str, List[str]]
    technical_terms: List[str]
    domain_specific: List[str]
class SurveyOptimizedCrawler:
    """
    Survey-optimized paper crawler that ensures comprehensive coverage
    for LLM-generated research surveys
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the survey-optimized crawler
        
        Args:
            verbose: Enable detailed output
        """
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; SurveyBot/1.0; +https://research.bot)'
        })
        
        # Survey-specific configuration
        self.config = {
            # Collection ratios
            'foundational_ratio': 0.4,    # 40% highly cited foundational papers
            'recent_ratio': 0.3,          # 30% recent papers with traction
            'trending_ratio': 0.3,        # 30% trending papers with growth

            # Citation thresholds (adjusted by paper age)
            'foundational_min_citations': 10,  
            'recent_min_citations': 5,        
            'trending_min_citations': 3,       
            
            # Time ranges
            'foundational_age_range': (3, 15),     # 3-15 years old
            'recent_age_range': (1, 5),            # 1-5 years old
            'trending_age_range': (0, 1),          # 0-2 years old

            # Quality thresholds
            'min_abstract_length': 100,
            'min_title_words': 3,
            'quality_score_threshold': 0.6,
            
            # Search parameters
            'max_results_per_query': 50
        }
        
        # Paper storage and deduplication
        self.papers = {}
        self.title_signatures = set()
        self.author_paper_map = defaultdict(list)
        
        self._log("Survey-Optimized Crawler initialized")
        self.chat_agent = ChatAgent()
    def _create_keyword_prompt(self, topic,user_keywords, max_keywords):
        """Create prompt for keyword generation"""
        # field_context = f" in the field of {field}" if field else ""
        
        return f"""
        You are an expert research librarian. Generate comprehensive search keywords for academic papers in Computer Science.

        USER INPUT:
        - Topic: {topic}
        - Initial Keywords: {', '.join(user_keywords)}
        
        Generate response in JSON format:
        {{
            "primary_keywords": [list of {max_keywords//2} most important keywords],
            "secondary_keywords": [list of {max_keywords//4} supporting keywords],
            "boolean_queries": [list of 5-7 boolean search strings using AND, OR],
            "synonyms": {{"keyword1": ["synonym1", "synonym2"]}},
            "technical_terms": [domain-specific technical terms],
            "domain_specific": [field-specific terminology and acronyms]
        }}
        
        Focus on terms used in academic paper titles, abstracts, and keywords.
        """
    
    def _parse_gemini_response(self, response_text):
        """Parse Gemini response to extract keywords"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            # print(json_match)
            if json_match:
                data = json.loads(json_match.group())
                # print(data)
                return SearchKeywords(
                    primary_keywords=data.get('primary_keywords', []),
                    secondary_keywords=data.get('secondary_keywords', []),
                    boolean_queries=data.get('boolean_queries', []),
                    synonyms=data.get('synonyms', {}),
                    technical_terms=data.get('technical_terms', []),
                    domain_specific=data.get('domain_specific', [])
                )
        except:
            pass
        
        return self._fallback_keywords([], "")
    
    def _fallback_keywords(self, user_keywords, topic):
        """Fallback keywords if AI generation fails"""
        return SearchKeywords(
            primary_keywords=user_keywords + [topic] if topic else user_keywords,
            secondary_keywords=[],
            boolean_queries=[f'"{topic}"' if topic else ''],
            synonyms={},
            technical_terms=[],
            domain_specific=[]
        )
    def _prepare_search_queries(self, keywords):
        """Prepare search queries from keywords"""
        queries = []
        
        # Add boolean queries first
        queries.extend(keywords.boolean_queries)
        
        # Add individual primary keywords
        # for kw in keywords.primary_keywords[:5]:
        #     queries.append(f'"{kw}"')
        
        # # Add combinations
        if len(keywords.primary_keywords) >= 2:
            queries.append(f'"{keywords.primary_keywords[0]}" AND "{keywords.primary_keywords[1]}"')
        
        return queries
    
    def collect_survey_papers(self, topic: str, user_kws: str, target_papers: int = 100) -> List[Dict[str, Any]]:
        """
        Collect papers optimized for survey generation
        
        Args:
            query: Research topic or question
            target_papers: Target number of papers to collect
            
        Returns:
            List of papers with survey-specific metadata
        """
        self._log("=" * 60)
        self._log(f" SURVEY-OPTIMIZED PAPER COLLECTION")
        self._log(f" Topic: '{topic}'")
        self._log(f" Target: {target_papers} papers")
        self._log("=" * 60)
        
        # Expand query for better coverage
        # expanded_queries = self._expand_query(query)
        # self._log(f" Generated {len(expanded_queries)} query variations")
        prompt_generate_kws = self._create_keyword_prompt(topic, user_kws, 10)
        generated_text = self.chat_agent.gemini_chat(prompt_generate_kws, temperature=0.1)
        # print(generated_text)
        keywords = self._parse_gemini_response(generated_text)
        print(f"Generated {len(keywords.primary_keywords)} primary keywords:")
        expanded_queries = [topic.lower()]
        for kw in keywords.primary_keywords:
            print(f"  • {kw}")
            expanded_queries.append(f'"{kw}"')
        for kw in keywords.secondary_keywords:
            expanded_queries.append(f'"{kw}"')

        # Calculate tier targets
        foundational_target = int(target_papers * self.config['foundational_ratio'])
        recent_target = int(target_papers * self.config['recent_ratio'])
        trending_target = target_papers - foundational_target - recent_target
        
        self._log(f"\n Multi-Tier Collection Strategy:")
        self._log(f"     Foundational: {foundational_target} papers (highly cited, established)")
        self._log(f"     Recent: {recent_target} papers (last 3 years, some citations)")
        self._log(f"     Trending: {trending_target} papers (citation growth, emerging)")
        
        # Collect papers by tier
        # Track total collected papers to avoid unnecessary API calls
        total_collected = 0
        all_papers = []

        # Tier 1: Foundational Papers
        self._log(f"\n  TIER 1: Foundational Papers")
        foundational_papers = self._collect_foundational_papers(expanded_queries, foundational_target)
        all_papers.extend(foundational_papers)
        total_collected += len(foundational_papers)

        # Tier 2: Recent Papers
        self._log(f"\n  TIER 2: Recent Papers")
        recent_papers = self._collect_recent_papers(expanded_queries, min(recent_target, target_papers - total_collected))
        all_papers.extend(recent_papers)
        total_collected += len(recent_papers)

        # Tier 3: Trending Papers
        self._log(f"\n  TIER 3: Trending Papers")
        trending_papers = self._collect_trending_papers(expanded_queries, min(trending_target, target_papers - total_collected))
        all_papers.extend(trending_papers)
        total_collected += len(trending_papers)
        
        # deduplicate papers
        all_papers = self._deduplicate_papers(all_papers)
        
        # Generate and display statistics
        self._generate_collection_statistics(all_papers, target_papers)
        
        return all_papers
    
    def _expand_query(self, query: str) -> List[str]:
        """Generate query variations for better coverage"""
        expanded = [query]
        
        # Academic variations
        academic_terms = ["survey", "review", "analysis", "approach", "method"]
        for term in academic_terms[:3]:
            expanded.append(f"{query} {term}")
        
        # Research focus variations
        research_focuses = [
            f"advances in {query}",
            f"recent developments {query}", 
            f"state of the art {query}",
            f"{query} techniques",
            f"{query} algorithms"
        ]
        expanded.extend(research_focuses)
        
        return expanded
    def get_pdf_link(self, paper):
        pdf_url = None
        # Try arXiv if available
        arxiv_id = None
        external_ids = paper.get('externalIds', {})
        if isinstance(external_ids, dict):
            arxiv_id = external_ids.get('ArXiv')
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        # Try DOI (not always direct PDF)
        doi = external_ids.get('DOI') if isinstance(external_ids, dict) else None
        if doi and not pdf_url:
            pdf_url = f"https://doi.org/{doi}"
        return pdf_url
    def _collect_foundational_papers(self, queries: List[str], target: int) -> List[Dict]:
        """Collect foundational papers (highly cited, established work)"""
        papers = []
        current_year = datetime.now().year
        
        for i, query in enumerate(queries):
            if len(papers) >= target:
                break
                
            self._log(f"   Query {i+1}/{len(queries)}: '{query}'")
            
            # Define foundational year range (2-15 years old)
            end_year = current_year - self.config['foundational_age_range'][0]
            start_year = current_year - self.config['foundational_age_range'][1]
            
            # Search across available sources
            query_papers = self._search_all_sources(
                query, 
                year_range=(start_year, end_year),
                min_citations=self.config['foundational_min_citations'],
                max_results=min(100, target//len(query) * 5)
            )
            papers.extend(query_papers)
        papers = self._deduplicate_papers(papers)
        # calculate score = citation_count * (1 / max(1, (2025 - year))) and take top 20% base on calculated score
        for paper in papers:
            # Calculate score
            year = int(paper.get('year', 2025))
            citations = paper.get('citationCount', 0)
            score = citations * (1 / max(1, (2025 - year)))
            paper['score'] = score
        
        papers.sort(key=lambda x: x.get('score', 0), reverse=True)
        papers = papers[:max(1, int(len(papers) * 0.2))]

        self._log(f" Collected {len(papers)} foundational papers")
        return papers
    
    def _collect_recent_papers(self, queries: List[str], target: int) -> List[Dict]:
        """Collect recent papers with some citation traction"""
        papers = []
        current_year = datetime.now().year
        
        for i, query in enumerate(queries):
            if len(papers) >= target:
                break
                
            self._log(f" Query {i+1}/{len(queries)}: '{query}'")
            
            # Define recent year range 
            start_year = current_year - self.config['recent_age_range'][1]
            end_year = current_year
            
            # Search with lower citation threshold for recent work
            query_papers = self._search_all_sources(
                query,
                year_range=(start_year, end_year),
                min_citations=self.config['recent_min_citations'],
                max_results=min(100, target//len(query) * 5)
            )
            papers.extend(query_papers)
        papers = self._deduplicate_papers(papers)
        # calculate score = citation_count * (1 / max(1, (2025 - year))) and take top 20% base on calculated score
        for paper in papers:
            # Calculate score
            year = paper.get('year', 2025)
            citations = paper.get('citationCount', 0)
            score = citations * (1 / max(1, (2025 - year)))
            paper['score'] = score
        
        papers.sort(key=lambda x: x.get('score', 0), reverse=True)
        papers = papers[:max(1, int(len(papers) * 0.2))]
        
        self._log(f" Collected {len(papers)} recent papers")
        return papers
    
    def _collect_trending_papers(self, queries: List[str], target: int) -> List[Dict]:
        """Collect trending papers (showing citation growth)"""
        papers = []
        current_year = datetime.now().year
        
        for i, query in enumerate(queries):
            if len(papers) >= target:
                break
                
            self._log(f" Query {i+1}/{len(queries)}: '{query}'")
            
            # Define trending year range 
            start_year = current_year - self.config['trending_age_range'][1]
            end_year = current_year - self.config['trending_age_range'][0]
            
            # Search for papers
            query_papers = self._search_all_sources(
                query,
                year_range=(start_year, end_year),
                min_citations=self.config['trending_min_citations'],
                max_results=min(100, target//len(query) * 5)
            )
            papers.extend(query_papers)
        papers = self._deduplicate_papers(papers)
        # calculate score = citation_count * (1 / max(1, (2025 - year))) and take top 20% base on calculated score
        for paper in papers:
            # Calculate score
            year = paper.get('year', 2025)
            citations = paper.get('citationCount', 0)
            score = citations * (1 / max(1, (2025 - year)))
            paper['score'] = score
        
        papers.sort(key=lambda x: x.get('score', 0), reverse=True)
        papers = papers[:max(1, int(len(papers) * 0.2))]
        
        self._log(f" Collected {len(papers)} trending papers")
        return papers
    
    def _search_all_sources(self, query: str, year_range: Tuple[int, int] = None, 
                           min_citations: int = 0, max_results: int = 100) -> List[Dict]:
        """Search Semantic Scholar for comprehensive paper collection"""
        all_results = []
        
        # Primary Source: Semantic Scholar (comprehensive academic database)
        try:
            semantic_results = self._search_semantic_scholar(query, year_range, min_citations, max_results)
            all_results.extend(semantic_results)
            self._log(f" Semantic Scholar: {len(semantic_results)} papers")
        except Exception as e:
            self._log(f" Semantic Scholar error: {e}")

        return all_results
        # # Deduplicate and return top results
        # deduplicated = self._deduplicate_papers(all_results)
        # return deduplicated[:max_results]
    
    def _search_semantic_scholar(self, query: str, year_range: Tuple[int, int] = None, 
                                min_citations: int = 0, limit=100) -> List[Dict]:
        """Search Semantic Scholar with enhanced metadata collection"""
        results = []
        
        # Semantic Scholar API endpoint
        api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
          
        # Build query parameters
        params = {
            'query': query,
            'limit': limit,
            'fields': 'paperId,title,authors,year,citationCount,abstract,url,venue,publicationDate,externalIds,references,citations'
        }             
        headers = {"x-api-key": 'l8YcOwyvxm7IWxaXJxAh87XhMqQQrQVg3XkPdKiF'}
        if year_range:
            params['year'] = f"{year_range[0]}-{year_range[1]}"
        
        try:
            response = self.session.get(api_url, params=params, timeout=60,headers=headers)
            while response.status_code == 429:
                self._log("Rate limit exceeded")
                # retry 
                time.sleep(60)
                response = self.session.get(api_url, params=params, timeout=60)
                self._log(response.status_code)
            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                
                # Handle case where papers might be None
                if not papers:
                    return results
                
                for paper in papers:
                    if not paper:  # Skip None papers
                        continue
                        
                    citations = paper.get('citationCount', 0)
                    if citations >= min_citations:
                        # Handle potential None values safely
                        title = paper.get('title', '') or ''
                        abstract = paper.get('abstract', '') or ''
                        authors = paper.get('authors', [])
                        if not isinstance(authors, list):
                            authors = []
                        pdf_url = self.get_pdf_link(paper)
                        standardized = {
                            'id': paper.get('paperId', '') or '',
                            'title': title,
                            'authors': [author.get('name', '') for author in authors if isinstance(author, dict)],
                            'year': paper.get('year'),
                            'citationCount': citations,
                            'abstract': abstract,
                            'url': paper.get('url', '') or '',
                            'pdf_url': pdf_url,
                            'venue': paper.get('venue', '') or '',
                            'publicationDate': paper.get('publicationDate', '') or '',
                            'externalIds': paper.get('externalIds', {}),
                            'references': paper.get('references', []) or [],
                            'cited_by': paper.get('citations', []) or []
                        }
                        results.append(standardized)
                        print('Collected: ', len(results), end='\r')
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            self._log(f"Semantic Scholar API error: {e}")
        
        return results
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers using multiple similarity metrics"""
        deduplicated = []
        seen_signatures = set()
        
        for paper in papers:
            # Create signature for deduplication
            title = paper.get('title', '').lower().strip()
            title_signature = self._create_title_signature(title)
            
            if title_signature not in seen_signatures and len(title) > 10:
                seen_signatures.add(title_signature)
                deduplicated.append(paper)
        
        return deduplicated
    
    def _create_title_signature(self, title: str) -> str:
        """Create a signature for title-based deduplication"""
        # Remove common words and normalize
        title = re.sub(r'[^\w\s]', '', title.lower())
        words = title.split()
        
        # Remove very common words
        stop_words = {'a', 'an', 'the', 'of', 'for', 'in', 'on', 'with', 'by', 'to', 'and', 'or'}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Create signature from first few meaningful words
        signature_words = meaningful_words[:10]
        return '_'.join(sorted(signature_words))
    
    def _generate_collection_statistics(self, papers: List[Dict], target: int):
        """Generate and display comprehensive collection statistics"""
        self._log("\n" + "=" * 60)
        self._log(" COLLECTION STATISTICS")
        self._log("=" * 60)
        
        total_collected = len(papers)
        self._log(f" Total Papers Collected: {total_collected} (target: {target})")
        
        # Citation statistics
        citations = [paper.get('citationCount', 0) for paper in papers]
        if citations:
            self._log(f"\n Citation Statistics:")
            self._log(f"   Average: {sum(citations) / len(citations):.1f}")
            self._log(f"   Median: {sorted(citations)[len(citations)//2]}")
            self._log(f"   Max: {max(citations)}")
            self._log(f"   Papers with 50+ citations: {sum(1 for c in citations if c >= 50)}")
        
        # Year distribution
        years = [paper.get('year', 0) for paper in papers if paper.get('year')]
        if years:
            year_counts = Counter(years)
            self._log(f"\n Year Distribution (top 5):")
            for year, count in year_counts.most_common(5):
                self._log(f"   {year}: {count} papers")
        
        # Quality metrics
        quality_scores = [paper.get('score', 0) for paper in papers]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            self._log(f"\n Quality Metrics:")
            self._log(f"   Average score: {avg_quality:.2f}")
        
        self._log("=" * 60)
    
    def _log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(message)


def main():
    """Main function for standalone usage"""
    
    # 1. Check for the two required arguments (query and count)
    # if len(sys.argv) != 3:
    #     print("Usage: python scripts/survey_crawler.py \"your research query\" target_paper_count")
    #     print("Example: python scripts/survey_crawler.py \"federated learning privacy\" 500")
    #     return
    
    # Argument 1 (index 1) is the entire research query (since it was quoted)
    # query = sys.argv[1]
    topic = 'A SURVEY ON ADVERSARIAL RECOMMENDER SYSTEMS'
    user_kws = 'adversarial attacks, recommender systems, adversarial ML'

    # Argument 2 (index 2) is the paper count
    # target_papers_arg = sys.argv[2]
    #
    # Initialize the default target papers count
    target_papers = 300 
    
    # 2. Convert the paper count argument to an integer
    # try:
    #     target_papers = int(target_papers_arg)
        
    #     # Optional check for non-positive numbers
    #     if target_papers <= 0:
    #         print(f"Warning: Target paper count must be positive. Resetting to 300.")
    #         target_papers = 300
            
    # except ValueError:
    #     print(f"Error: Invalid count '{target_papers_arg}'. Target paper count must be an integer. Using default (300).")
    
    # Initialize crawler
    crawler = SurveyOptimizedCrawler(verbose=True)
    
    # Collect papers
    papers = crawler.collect_survey_papers(topic, user_kws, target_papers)
    
    # Save results
    save_dir = f"paper_data/{topic.replace(' ', '_')}/info"
    os.makedirs(save_dir, exist_ok=True)

    output_file = f"{save_dir}/crawl_papers.json"
    with open(output_file, 'w') as f:
        json.dump(papers, f, indent=2)
    
    print(f"\n Results saved to: {output_file}")
    print(f" Collected {len(papers)} papers ready for survey generation")


if __name__ == "__main__":
    main()