"""
Graph structure and node/edge management for hierarchical survey knowledge graph.
"""
import networkx as nx
import json
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
from datetime import datetime

# Load spaCy model for NLP tasks
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class PaperNode:
    paper_id: str
    tier: str
    title: str
    authors: List[str]
    abstract: str
    year: int
    venue: str
    keywords: List[str]
    concepts: List[str]
    methods: List[str]
    domains: List[str]
    citation_count: int
    citation_velocity: float
    influential_citations: int
    survey_importance: str
    recommended_section: str
    contribution_type: str
    centrality_score: float = 0.0
    bridge_score: float = 0.0
    novelty_score: float = 0.0
    influence_score: float = 0.0

class HierarchicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.papers = {}
        self.tier_nodes = {
            'foundational': set(),
            'recent': set(),
            'trending': set()
        }
        self.relationship_types = {
            'SIMILAR_APPROACH': 'Uses similar methodology or approach',
            'COMPETING_METHOD': 'Proposes alternative/competing solution',
            'COMPLEMENTARY': 'Enhances or complements the other work',
            'CITATION_CLUSTER': 'Frequently cited together',
            'SAME_DOMAIN': 'Works in the same research domain',
            'BUILDS_UPON': 'Recent/Trending work building on Foundational',
            'EXTENDS': 'Extends concepts from earlier work',
            'CHALLENGES': 'Challenges assumptions of earlier work',
            'IMPLEMENTS': 'Practical implementation of theoretical work',
            'VALIDATES': 'Provides empirical validation',
            'SUPERSEDES': 'Newer method replacing older approach',
            'INTRODUCES_CONCEPT': 'First to introduce key concept',
            'POPULARIZES': 'Made concept mainstream',
            'CITES': 'Direct citation relationship',
            'APPLIES': 'Applies method to new domain'
        }
        self.concept_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 3))
        self.concept_matrix = None
    def add_paper(self, paper_data: Dict) -> str:
        text_content = f"{paper_data.get('title', '')} {paper_data.get('abstract', '')}"
        concepts = self._extract_concepts(text_content)
        methods = self._extract_methods(text_content)
        domains = self._extract_domains(paper_data.get('fieldsOfStudy', []))
        paper_node = PaperNode(
            paper_id=paper_data['id'],
            tier=paper_data.get('survey_tier', 'unknown'),
            title=paper_data.get('title', ''),
            authors=paper_data.get('authors', []),
            abstract=paper_data.get('abstract', ''),
            year=paper_data.get('year', datetime.now().year),
            venue=paper_data.get('venue', ''),
            keywords=paper_data.get('survey_keywords', []),
            concepts=concepts,
            methods=methods,
            domains=domains,
            citation_count=paper_data.get('citationCount', 0),
            citation_velocity=paper_data.get('citation_context', {}).get('velocity', 0),
            influential_citations=paper_data.get('influentialCitations', 0),
            survey_importance=paper_data.get('survey_importance', 'medium'),
            recommended_section=paper_data.get('recommended_section', 'supplementary'),
            contribution_type=self._classify_contribution_type(paper_data)
        )
        self.graph.add_node(paper_node.paper_id, **asdict(paper_node))
        self.papers[paper_node.paper_id] = paper_node
        self.tier_nodes[paper_node.tier].add(paper_node.paper_id)
        return paper_node.paper_id
    def build_relationships(self):
        self._compute_concept_similarity()
        self._build_intra_tier_relationships()
        self._build_inter_tier_relationships()
        self._compute_graph_metrics()
    def _extract_concepts(self, text: str) -> List[str]:
        if not nlp or not text:
            return []
        doc = nlp(text)
        concepts = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                concepts.append(chunk.text.lower())
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
                concepts.append(ent.text.lower())
        return list(set(concepts))[:10]
    def _extract_methods(self, text: str) -> List[str]:
        method_patterns = [
            r'\b(?:algorithm|method|approach|technique|framework|model|system)\b',
            r'\b(?:learning|training|optimization|classification|clustering)\b',
            r'\b(?:neural|deep|machine|artificial|statistical)\b',
            r'\b(?:supervised|unsupervised|reinforcement|semi-supervised)\b'
        ]
        methods = []
        text_lower = text.lower()
        for pattern in method_patterns:
            matches = re.findall(pattern, text_lower)
            methods.extend(matches)
        return list(set(methods))
    def _extract_domains(self, fields_of_study: List[str]) -> List[str]:
        if not fields_of_study:
            return []
        domain_mapping = {
            'computer science': ['cs', 'computer science', 'artificial intelligence'],
            'machine learning': ['machine learning', 'deep learning', 'neural networks'],
            'natural language processing': ['nlp', 'natural language', 'linguistics'],
            'computer vision': ['computer vision', 'image processing', 'vision'],
            'robotics': ['robotics', 'autonomous systems'],
            'mathematics': ['mathematics', 'statistics', 'probability'],
            'physics': ['physics', 'quantum'],
            'biology': ['biology', 'bioinformatics', 'computational biology']
        }
        domains = []
        for field in fields_of_study:
            field_lower = field.lower()
            for domain, keywords in domain_mapping.items():
                if any(keyword in field_lower for keyword in keywords):
                    domains.append(domain)
                    break
        return list(set(domains))
    def _classify_contribution_type(self, paper_data: Dict) -> str:
        title = paper_data.get('title', '').lower()
        abstract = paper_data.get('abstract', '').lower()
        text = f"{title} {abstract}"
        if any(word in text for word in ['survey', 'review', 'overview', 'taxonomy']):
            return 'survey'
        elif any(word in text for word in ['experiment', 'empirical', 'evaluation', 'benchmark']):
            return 'empirical'
        elif any(word in text for word in ['theory', 'theoretical', 'analysis', 'proof']):
            return 'theoretical'
        elif any(word in text for word in ['method', 'algorithm', 'approach', 'framework']):
            return 'methodology'
        else:
            return 'other'
    def _compute_concept_similarity(self):
        if not self.papers:
            return
        documents = []
        paper_ids = []
        for paper_id, paper in self.papers.items():
            doc_text = f"{paper.title} {paper.abstract} {' '.join(paper.concepts)}"
            documents.append(doc_text)
            paper_ids.append(paper_id)
        try:
            self.concept_matrix = self.concept_vectorizer.fit_transform(documents)
            self.paper_id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}
        except:
            print("Warning: Could not compute concept similarity matrix")
            self.concept_matrix = None
    def _build_intra_tier_relationships(self):
        for tier, paper_ids in self.tier_nodes.items():
            paper_list = list(paper_ids)
            for i, paper1_id in enumerate(paper_list):
                for paper2_id in paper_list[i+1:]:
                    paper1 = self.papers[paper1_id]
                    paper2 = self.papers[paper2_id]
                    concept_sim = self._get_concept_similarity(paper1_id, paper2_id)
                    domain_overlap = len(set(paper1.domains) & set(paper2.domains))
                    max_domains = max(len(paper1.domains), len(paper2.domains), 1)
                    method_overlap = len(set(paper1.methods) & set(paper2.methods))
                    max_methods = max(len(paper1.methods), len(paper2.methods), 1)
                    if concept_sim > 0.7:
                        weight = 0.8 + 0.2 * (concept_sim - 0.7) / 0.3
                        self._add_relationship(paper1_id, paper2_id, 'SIMILAR_APPROACH', weight=round(weight, 3))
                    elif concept_sim > 0.5:
                        weight = 0.5 + 0.3 * (concept_sim - 0.5) / 0.2
                        self._add_relationship(paper1_id, paper2_id, 'COMPLEMENTARY', weight=round(weight, 3))
                    if domain_overlap > 0:
                        weight = 0.3 + 0.4 * (domain_overlap / max_domains)
                        self._add_relationship(paper1_id, paper2_id, 'SAME_DOMAIN', weight=round(weight, 3))
                    if method_overlap > 0:
                        weight = 0.2 + 0.3 * (method_overlap / max_methods)
                        self._add_relationship(paper1_id, paper2_id, 'METHOD_OVERLAP', weight=round(weight, 3))
    def _build_inter_tier_relationships(self):
        for source_tier in ['recent', 'trending']:
            for source_id in self.tier_nodes[source_tier]:
                for target_id in self.tier_nodes['foundational']:
                    source_paper = self.papers[source_id]
                    target_paper = self.papers[target_id]
                    concept_sim = self._get_concept_similarity(source_id, target_id)
                    if concept_sim > 0.6:
                        if source_paper.year > target_paper.year + 2:
                            self._add_relationship(source_id, target_id, 'BUILDS_UPON', weight=concept_sim)
                        elif 'extends' in source_paper.abstract.lower():
                            self._add_relationship(source_id, target_id, 'EXTENDS', weight=concept_sim)
                        elif 'challenge' in source_paper.abstract.lower():
                            self._add_relationship(source_id, target_id, 'CHALLENGES', weight=concept_sim)
        for recent_id in self.tier_nodes['recent']:
            for trending_id in self.tier_nodes['trending']:
                recent_paper = self.papers[recent_id]
                trending_paper = self.papers[trending_id]
                concept_sim = self._get_concept_similarity(recent_id, trending_id)
                if concept_sim > 0.5 and trending_paper.citation_velocity > recent_paper.citation_velocity:
                    self._add_relationship(recent_id, trending_id, 'EVOLVES_TO', weight=concept_sim)
    def _get_concept_similarity(self, paper1_id: str, paper2_id: str) -> float:
        if self.concept_matrix is None or getattr(self.concept_matrix, 'shape', [0])[0] == 0 or paper1_id not in self.paper_id_to_idx or paper2_id not in self.paper_id_to_idx:
            return 0.0
        idx1 = self.paper_id_to_idx[paper1_id]
        idx2 = self.paper_id_to_idx[paper2_id]
        similarity = cosine_similarity(self.concept_matrix[idx1], self.concept_matrix[idx2])[0][0]
        return similarity
    def _add_relationship(self, source_id: str, target_id: str, rel_type: str, weight: float = 1.0, **properties):
        self.graph.add_edge(
            source_id, target_id,
            relationship_type=rel_type,
            weight=weight,
            description=self.relationship_types.get(rel_type, ''),
            **properties
        )
    def _compute_graph_metrics(self):
        undirected_graph = self.graph.to_undirected()
        centrality = nx.betweenness_centrality(undirected_graph)
        pagerank = nx.pagerank(self.graph)
        for paper_id, paper in self.papers.items():
            paper.centrality_score = centrality.get(paper_id, 0.0)
            paper.influence_score = pagerank.get(paper_id, 0.0)
            paper.bridge_score = self._compute_bridge_score(paper_id)
            paper.novelty_score = self._compute_novelty_score(paper)
            self.graph.nodes[paper_id].update(asdict(paper))
    def _compute_bridge_score(self, paper_id: str) -> float:
        paper_tier = self.papers[paper_id].tier
        cross_tier_edges = 0
        total_edges = 0
        for _, target, data in self.graph.edges(paper_id, data=True):
            total_edges += 1
            if self.papers[target].tier != paper_tier:
                cross_tier_edges += 1
        return cross_tier_edges / max(total_edges, 1)
    def _compute_novelty_score(self, paper: PaperNode) -> float:
        base_score = 0.0
        unique_concepts = len(set(paper.concepts))
        base_score += min(unique_concepts / 10, 0.3)
        if paper.tier == 'trending':
            base_score += min(paper.citation_velocity / 20, 0.3)
        if paper.contribution_type == 'methodology':
            base_score += 0.2
        current_year = datetime.now().year
        if paper.year >= current_year - 2 and paper.citation_count > 50:
            base_score += 0.2
        return min(base_score, 1.0)
    def export_graph_json(self, filepath: str):
        nodes = [asdict(paper) for paper in self.papers.values()]
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edge = {
                'source': source,
                'target': target,
                'relationship_type': data.get('relationship_type', ''),
                'weight': data.get('weight', 1.0),
                'description': data.get('description', ''),
                **{k: v for k, v in data.items() if k not in ['relationship_type', 'weight', 'description']}
            }
            edges.append(edge)
        graph_json = {'nodes': nodes, 'edges': edges}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_json, f, indent=2, ensure_ascii=False)
        print(f"💾 Knowledge graph exported to: {filepath}")
