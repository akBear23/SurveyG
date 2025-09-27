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
- Multi-tier collection strategy (40% foundational, 35% recent, 25% trending)
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
            'foundational_ratio': 0.3,    # 30% highly cited foundational papers
            'recent_ratio': 0.35,          # 35% recent papers with traction
            'trending_ratio': 0.35,        # 35% trending papers with growth

            # Citation thresholds (adjusted by paper age)
            'foundational_min_citations': 20,  # Reduced from 50
            'recent_min_citations': 3,         # Reduced from 5
            'trending_min_citations': 5,       # Reduced from 10
            
            # Time ranges
            'foundational_age_range': (2, 15),     # 2-15 years old
            'recent_age_range': (0, 3),            # 0-3 years old
            'trending_age_range': (0, 6),          # 0-6 years old

            # Quality thresholds
            'min_abstract_length': 100,
            'min_title_words': 3,
            'quality_score_threshold': 0.6,
            
            # Search parameters
            'max_results_per_query': 150,
            'max_queries_per_tier': 5,
            'reference_follow_depth': 2,
        }
        
        # Paper storage and deduplication
        self.papers = {}
        self.title_signatures = set()
        self.author_paper_map = defaultdict(list)
        
        self._log("Survey-Optimized Crawler initialized")
    
    def collect_survey_papers(self, query: str, target_papers: int = 300) -> List[Dict[str, Any]]:
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
        self._log(f" Query: '{query}'")
        self._log(f" Target: {target_papers} papers")
        self._log("=" * 60)
        
        # Expand query for better coverage
        expanded_queries = self._expand_query(query)
        self._log(f" Generated {len(expanded_queries)} query variations")
        
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
        if total_collected >= target_papers:
            self._log(f"Reached target after foundational tier.")
            enhanced_papers = self._enhance_papers_for_survey(all_papers)
            final_papers = self._final_quality_filter(enhanced_papers, target_papers)
            survey_ready_papers = self._add_survey_metadata(final_papers)
            self._generate_collection_statistics(survey_ready_papers, target_papers)
            return survey_ready_papers

        # Tier 2: Recent Papers
        self._log(f"\n  TIER 2: Recent Papers")
        recent_papers = self._collect_recent_papers(expanded_queries, min(recent_target, target_papers - total_collected))
        all_papers.extend(recent_papers)
        total_collected += len(recent_papers)
        if total_collected >= target_papers:
            self._log(f"Reached target after recent tier.")
            enhanced_papers = self._enhance_papers_for_survey(all_papers)
            final_papers = self._final_quality_filter(enhanced_papers, target_papers)
            survey_ready_papers = self._add_survey_metadata(final_papers)
            self._generate_collection_statistics(survey_ready_papers, target_papers)
            return survey_ready_papers

        # Tier 3: Trending Papers
        self._log(f"\n  TIER 3: Trending Papers")
        trending_papers = self._collect_trending_papers(expanded_queries, min(trending_target, target_papers - total_collected))
        all_papers.extend(trending_papers)
        total_collected += len(trending_papers)
        
        # Post-processing and enhancement
        self._log(f"\n POST-PROCESSING")
        enhanced_papers = self._enhance_papers_for_survey(all_papers)
        
        # Final quality filtering and ranking
        final_papers = self._final_quality_filter(enhanced_papers, target_papers)
        
        # Add survey-specific metadata
        survey_ready_papers = self._add_survey_metadata(final_papers)
        
        # Generate and display statistics
        self._generate_collection_statistics(survey_ready_papers, target_papers)
        
        return survey_ready_papers
    
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
        expanded.extend(research_focuses[:2])
        
        # Domain-specific additions
        if any(term in query.lower() for term in ['learning', 'neural', 'ai', 'machine']):
            expanded.extend([
                f"{query} deep learning",
                f"{query} machine learning"
            ])
        
        return expanded[:8]  # Limit to avoid too many requests
    
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
                max_results=self.config['max_results_per_query']
            )
            
            # Filter for foundational quality
            for paper in query_papers:
                if len(papers) >= target:
                    break
                    
                if self._is_foundational_quality(paper):
                    paper['survey_tier'] = 'foundational'
                    paper['collection_query'] = query
                    papers.append(paper)
        
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
            
            # Define recent year range (0-3 years old)
            start_year = current_year - self.config['recent_age_range'][1]
            end_year = current_year
            
            # Search with lower citation threshold for recent work
            query_papers = self._search_all_sources(
                query,
                year_range=(start_year, end_year),
                min_citations=self.config['recent_min_citations'],
                max_results=self.config['max_results_per_query']
            )
            
            # Filter for recent quality
            for paper in query_papers:
                if len(papers) >= target:
                    break
                    
                if self._is_recent_quality(paper):
                    paper['survey_tier'] = 'recent'
                    paper['collection_query'] = query
                    papers.append(paper)
        
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
            
            # Define trending year range (1-6 years old)
            start_year = current_year - self.config['trending_age_range'][1]
            end_year = current_year - self.config['trending_age_range'][0]
            
            # Search for papers with potential citation growth
            query_papers = self._search_all_sources(
                query,
                year_range=(start_year, end_year),
                min_citations=self.config['trending_min_citations'],
                max_results=self.config['max_results_per_query']
            )
            
            # Filter for trending quality (citation velocity)
            for paper in query_papers:
                if len(papers) >= target:
                    break
                    
                if self._is_trending_quality(paper):
                    paper['survey_tier'] = 'trending'
                    paper['collection_query'] = query
                    papers.append(paper)
        
        self._log(f" Collected {len(papers)} trending papers")
        return papers
    
    def _search_all_sources(self, query: str, year_range: Tuple[int, int] = None, 
                           min_citations: int = 0, max_results: int = 100) -> List[Dict]:
        """Search Semantic Scholar for comprehensive paper collection"""
        all_results = []
        
        # Primary Source: Semantic Scholar (comprehensive academic database)
        try:
            semantic_results = self._search_semantic_scholar(query, year_range, min_citations)
            all_results.extend(semantic_results)
            self._log(f" Semantic Scholar: {len(semantic_results)} papers")
        except Exception as e:
            self._log(f" Semantic Scholar error: {e}")

        
        # Deduplicate and return top results
        deduplicated = self._deduplicate_papers(all_results)
        return deduplicated[:max_results]
    
    def _search_semantic_scholar(self, query: str, year_range: Tuple[int, int] = None, 
                                min_citations: int = 0) -> List[Dict]:
        """Search Semantic Scholar with enhanced metadata collection"""
        results = []
        
        # Semantic Scholar API endpoint
        api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # Build query parameters
        params = {
            'query': query,
            'limit': 100,
            'fields': 'paperId,title,authors,year,citationCount,abstract,url,venue,publicationDate,fieldsOfStudy,influentialCitationCount,references,citations'
        }
        
        if year_range:
            params['year'] = f"{year_range[0]}-{year_range[1]}"
        
        try:
            response = self.session.get(api_url, params=params, timeout=60)
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
                        
                        standardized = {
                            'id': paper.get('paperId', '') or '',
                            'title': title,
                            'authors': [author.get('name', '') for author in authors if isinstance(author, dict)],
                            'year': paper.get('year'),
                            'citationCount': citations,
                            'influentialCitations': paper.get('influentialCitationCount', 0) or 0,
                            'abstract': abstract,
                            'url': paper.get('url', '') or '',
                            'venue': paper.get('venue', '') or '',
                            'fieldsOfStudy': paper.get('fieldsOfStudy', []) or [],
                            'source': 'semantic_scholar',
                            'publicationDate': paper.get('publicationDate', '') or '',
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
        signature_words = meaningful_words[:5]
        return '_'.join(sorted(signature_words))
    
    def _is_foundational_quality(self, paper: Dict) -> bool:
        """Check if paper meets foundational tier quality standards"""
        citations = paper.get('citationCount', 0)
        year = paper.get('year', datetime.now().year)
        title = paper.get('title', '') or ''
        abstract = paper.get('abstract', '') or ''
        
        # Basic quality checks
        if len(title) < 10 or len(abstract) < 100:
            return False
        
        # Citation requirement (adjusted for age)
        current_year = datetime.now().year
        age = current_year - year
        min_citations = max(20, 30 - age * 2)  # More gradual reduction based on age
        
        if citations < min_citations:
            return False
        
        # Quality indicators
        quality_score = self._calculate_quality_score(paper)
        return quality_score >= 0.3  # Data-driven threshold

    def _is_recent_quality(self, paper: Dict) -> bool:
        """Check if paper meets recent tier quality standards"""
        citations = paper.get('citationCount', 0)
        year = paper.get('year', datetime.now().year)
        title = paper.get('title', '') or ''
        abstract = paper.get('abstract', '') or ''
        
        # Basic quality checks
        if len(title) < 10 or len(abstract) < 50:
            return False
        
        # Lower citation threshold for recent papers
        current_year = datetime.now().year
        age = current_year - year
        
        if age <= 1:
            min_citations = 1  # Very recent papers
        elif age <= 2:
            min_citations = 3  # Recent papers
        else:
            min_citations = 5  # Older recent papers
        
        if citations < min_citations:
            return False
        
        quality_score = self._calculate_quality_score(paper)
        return quality_score >= 0.3  # Data-driven threshold

    def _is_trending_quality(self, paper: Dict) -> bool:
        """Check if paper shows trending characteristics (citation growth)"""
        citations = paper.get('citationCount', 0)
        year = paper.get('year', datetime.now().year)
        current_year = datetime.now().year
        age = current_year - year
        
        if age <= 0:
            return False  # Too new to be trending
        
        # Calculate citation velocity (citations per year)
        citation_velocity = citations / max(age, 1)
        
        # Trending criteria: good citation velocity
        min_velocity = 4  # Reduced from 8 to 4 citations per year
        
        if citation_velocity >= min_velocity:
            quality_score = self._calculate_quality_score(paper)
            return quality_score >= 0.3  # Data-driven threshold
        
        return False
    
    def _calculate_quality_score(self, paper: Dict) -> float:
        """Calculate a quality score for survey suitability based on h-index, venue, and citations"""
        score = 0.0
        # Venue quality (0.2)
        venue = paper.get('venue', '') or ''
        venue = venue.lower()
        prestigious_venues = ['nature', 'science', 'cell', 'neurips', 'icml', 'iclr', 'aaai', 'ijcai', 'acl', 'emnlp']
        if any(v in venue for v in prestigious_venues):
            score += 0.2
        # h-index quality (0.6)
        h_index = paper.get('h_index', 0)
        if h_index >= 100:
            score += 0.6
        elif h_index >= 50:
            score += 0.4
        elif h_index >= 20:
            score += 0.2
        # Citation quality (0.2)
        citations = paper.get('citationCount', 0)
        if citations >= 100:
            score += 0.2
        elif citations >= 50:
            score += 0.15
        elif citations >= 10:
            score += 0.1
        elif citations >= 5:
            score += 0.05
        return min(score, 1.0)
    
    def _enhance_papers_for_survey(self, papers: List[Dict]) -> List[Dict]:
        """Add survey-specific enhancements to papers, including LLM summary of full content"""
        from src.models.LLM.ChatAgent import ChatAgent
        import requests
        import tempfile
        from PyPDF2 import PdfReader
        enhanced = []
        chat_agent = ChatAgent()
        for paper in papers:
            # Add survey-specific metrics
            paper['survey_score'] = self._calculate_quality_score(paper)
            paper['survey_importance'] = self._determine_importance_level(paper)
            paper['survey_keywords'] = self._extract_survey_keywords(paper)
            paper['citation_context'] = self._analyze_citation_context(paper)
            # Download and summarize full paper content
            url = paper.get('url', '')
            full_text = None
            if url and url.endswith('.pdf'):
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as tmp_pdf:
                        tmp_pdf.write(response.content)
                        tmp_pdf.flush()
                        reader = PdfReader(tmp_pdf.name)
                        full_text = "\n".join(page.extract_text() or '' for page in reader.pages)
                except Exception as e:
                    full_text = None
                    paper['summary_error'] = f"PDF download or parse error: {e}"
            # Fallback to abstract if full text not available
            if not full_text or len(full_text) < 500:
                full_text = paper.get('abstract', '')
            title = paper.get('title', '')
            prompt = (
                f"Summarize the main contribution of the following paper for a research survey, include what problem the paper addresses, the method used to solve it, the key findings, and key math equations (if applicable). "
                f"Additionally, analyze whether this paper proposes a new research direction or method path, "
                f"and explain your reasoning. Respond with a clear statement: 'New Direction: Yes' or 'New Direction: No'. "
                f"Also, classify the paper as one of the following types: 'Survey', 'Empirical', 'Theoretical', or 'Methodology'. Respond with a clear statement: 'Type: <type>'.\n"
                f"Title: {title}\nContent: {full_text[:12000]}"
            )
            try:
                summary = chat_agent.gemini_chat(prompt, temperature=0.3)
                # Parse LLM response for 'New Direction' statement
                new_direction = 1 if 'new direction: yes' in summary.lower() else 0
                # Parse LLM response for type classification
                paper_type = None
                for t in ['survey', 'empirical', 'theoretical', 'methodology']:
                    if f'type: {t}' in summary.lower():
                        paper_type = t.capitalize()
                        break
                if not paper_type:
                    paper_type = 'Unknown'
            except Exception as e:
                summary = f"LLM summary error: {e}"
                new_direction = 0
                paper_type = 'Unknown'
            paper['summary'] = summary
            paper['new_direction'] = new_direction
            paper['paper_type'] = paper_type
            enhanced.append(paper)
        return enhanced
    
    def _determine_importance_level(self, paper: Dict) -> str:
        """Determine the importance level for survey inclusion"""
        score = paper.get('survey_score', 0)
        citations = paper.get('citationCount', 0)
        tier = paper.get('survey_tier', '')
        
        if score >= 0.8 and citations >= 100:
            return 'high'
        elif score >= 0.7 and citations >= 50:
            return 'high'
        elif score >= 0.6 and citations >= 20:
            return 'medium'
        elif tier == 'recent' and score >= 0.6:
            return 'medium'  # Recent papers get importance boost
        else:
            return 'low'
    
    def _extract_survey_keywords(self, paper: Dict) -> List[str]:
        """Extract keywords relevant for survey organization"""
        title = paper.get('title', '') or ''
        title = title.lower()
        abstract = paper.get('abstract', '') or ''
        abstract = abstract.lower()
        
        # Common research keywords
        keywords = []
        
        # Method keywords
        method_terms = ['algorithm', 'method', 'approach', 'technique', 'framework', 'model', 'system']
        for term in method_terms:
            if term in title or term in abstract:
                keywords.append(term)
        
        # Domain keywords from fields of study
        fields = paper.get('fieldsOfStudy', []) or []
        if fields:
            keywords.extend([field.lower() for field in fields[:3] if field])
        
        return list(set(keywords))[:5]  # Limit to 5 keywords
    
    def _analyze_citation_context(self, paper: Dict) -> Dict:
        """Analyze citation context for survey writing"""
        citations = paper.get('citationCount', 0)
        year = paper.get('year', datetime.now().year)
        current_year = datetime.now().year
        age = current_year - year
        
        # Citation velocity
        velocity = citations / max(age, 1) if age > 0 else citations
        
        # Citation category
        if citations >= 1000:
            category = 'highly_influential'
        elif citations >= 100:
            category = 'influential'
        elif citations >= 50:
            category = 'well_cited'
        elif citations >= 10:
            category = 'moderately_cited'
        else:
            category = 'emerging'
        
        return {
            'velocity': round(velocity, 2),
            'category': category,
            'age_group': 'recent' if age <= 3 else 'established' if age <= 10 else 'foundational'
        }
    
    def _final_quality_filter(self, papers: List[Dict], target: int) -> List[Dict]:
        """Apply final quality filtering and ranking"""
        # Sort by survey score and citations
        scored_papers = [
            (paper, paper.get('survey_score', 0) * 0.6 + 
             min(paper.get('citationCount', 0) / 100, 1.0) * 0.4)
            for paper in papers
        ]
        
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Take top papers, ensuring tier balance
        final_papers = []
        tier_counts = defaultdict(int)
        max_per_tier = target // 3
        
        for paper, score in scored_papers:
            tier = paper.get('survey_tier', 'unknown')
            
            if (len(final_papers) < target and 
                tier_counts[tier] < max_per_tier * 1.5):  # Allow some flexibility
                final_papers.append(paper)
                tier_counts[tier] += 1
        
        return final_papers
    
    def _add_survey_metadata(self, papers: List[Dict]) -> List[Dict]:
        """Add final survey-specific metadata"""
        for i, paper in enumerate(papers):
            paper['survey_id'] = i + 1
            paper['collection_timestamp'] = datetime.now().isoformat()
            paper['recommended_section'] = self._recommend_survey_section(paper)
            
        return papers
    
    def _recommend_survey_section(self, paper: Dict) -> str:
        """Recommend which section of the survey this paper belongs to"""
        tier = paper.get('survey_tier', '')
        importance = paper.get('survey_importance', '')
        
        if tier == 'foundational' and importance == 'high':
            return 'background'
        elif tier == 'foundational':
            return 'related_work'
        elif tier == 'recent' and importance == 'high':
            return 'recent_advances'
        elif tier == 'recent':
            return 'current_work'
        elif tier == 'trending':
            return 'future_directions'
        else:
            return 'supplementary'
    
    def _generate_collection_statistics(self, papers: List[Dict], target: int):
        """Generate and display comprehensive collection statistics"""
        self._log("\n" + "=" * 60)
        self._log(" COLLECTION STATISTICS")
        self._log("=" * 60)
        
        total_collected = len(papers)
        self._log(f" Total Papers Collected: {total_collected} (target: {target})")
        
        # Tier distribution
        tier_counts = Counter(paper.get('survey_tier', 'unknown') for paper in papers)
        self._log(f"\n  Tier Distribution:")
        for tier, count in tier_counts.items():
            percentage = (count / total_collected) * 100 if total_collected > 0 else 0
            self._log(f"   {tier.title()}: {count} papers ({percentage:.1f}%)")
        
        # Source distribution
        source_counts = Counter(paper.get('source', 'unknown') for paper in papers)
        self._log(f"\n Source Distribution:")
        for source, count in source_counts.items():
            percentage = (count / total_collected) * 100 if total_collected > 0 else 0
            self._log(f"   {source.title()}: {count} papers ({percentage:.1f}%)")
        
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
        quality_scores = [paper.get('survey_score', 0) for paper in papers]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            high_quality = sum(1 for s in quality_scores if s >= 0.7)
            self._log(f"\n Quality Metrics:")
            self._log(f"   Average quality score: {avg_quality:.2f}")
            self._log(f"   High quality papers (≥0.7): {high_quality}")
        
        # Survey recommendations
        self._log(f"\n Survey Writing Tips:")
        self._log(f"   - Start with 'foundational' papers for background")
        self._log(f"   - Use 'recent' papers for current state-of-the-art")
        self._log(f"   - Include 'trending' papers for future directions")
        self._log(f"   - Focus on 'high' importance papers for main content")
        
        self._log("=" * 60)
    
    def _log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(message)


def main():
    """Main function for standalone usage"""
    
    # 1. Check for the two required arguments (query and count)
    if len(sys.argv) != 3:
        print("Usage: python scripts/survey_crawler.py \"your research query\" target_paper_count")
        print("Example: python scripts/survey_crawler.py \"federated learning privacy\" 500")
        return
    
    # Argument 1 (index 1) is the entire research query (since it was quoted)
    query = sys.argv[1]

    # Argument 2 (index 2) is the paper count
    target_papers_arg = sys.argv[2]
    
    # Initialize the default target papers count
    target_papers = 300 
    
    # 2. Convert the paper count argument to an integer
    try:
        target_papers = int(target_papers_arg)
        
        # Optional check for non-positive numbers
        if target_papers <= 0:
            print(f"Warning: Target paper count must be positive. Resetting to 300.")
            target_papers = 300
            
    except ValueError:
        print(f"Error: Invalid count '{target_papers_arg}'. Target paper count must be an integer. Using default (300).")
    
    # Initialize crawler
    crawler = SurveyOptimizedCrawler(verbose=True)
    
    # Collect papers
    papers = crawler.collect_survey_papers(query, target_papers)
    
    # Save results
    output_file = f"survey_papers_{query.replace(' ', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(papers, f, indent=2)
    
    print(f"\n Results saved to: {output_file}")
    print(f" Collected {len(papers)} papers ready for survey generation")


if __name__ == "__main__":
    main()
