"""
Query engine for hierarchical knowledge graph (survey writing, concept extraction, etc).
"""
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from .graph import HierarchicalKnowledgeGraph, PaperNode

class SurveyQueryEngine:
    def __init__(self, knowledge_graph: HierarchicalKnowledgeGraph):
        self.kg = knowledge_graph
    def get_foundational_papers(self, min_citations: int = 50, max_papers: int = 20) -> List[PaperNode]:
        foundational_papers = []
        for paper_id in self.kg.tier_nodes['foundational']:
            paper = self.kg.papers[paper_id]
            if paper.citation_count >= min_citations:
                foundational_papers.append({
                    'paper': paper,
                    'importance_score': paper.influence_score + (paper.citation_count / 1000)
                })
        foundational_papers.sort(key=lambda x: x['importance_score'], reverse=True)
        return [item['paper'] for item in foundational_papers[:max_papers]]
    def get_recent_breakthroughs(self, min_velocity: float = 10, max_papers: int = 15) -> List[PaperNode]:
        recent_papers = []
        for paper_id in self.kg.tier_nodes['recent']:
            paper = self.kg.papers[paper_id]
            if paper.citation_velocity >= min_velocity:
                recent_papers.append({
                    'paper': paper,
                    'breakthrough_score': paper.citation_velocity + paper.novelty_score * 10
                })
        recent_papers.sort(key=lambda x: x['breakthrough_score'], reverse=True)
        return [item['paper'] for item in recent_papers[:max_papers]]
    def get_trending_directions(self, max_papers: int = 10) -> List[PaperNode]:
        trending_papers = []
        for paper_id in self.kg.tier_nodes['trending']:
            paper = self.kg.papers[paper_id]
            trending_papers.append({
                'paper': paper,
                'trend_score': paper.novelty_score + paper.bridge_score + (paper.citation_velocity / 20)
            })
        trending_papers.sort(key=lambda x: x['trend_score'], reverse=True)
        return [item['paper'] for item in trending_papers[:max_papers]]
    def get_evolution_path(self, concept: str) -> List[Tuple[PaperNode, str]]:
        evolution_path = []
        concept_papers = []
        for paper_id, paper in self.kg.papers.items():
            if any(concept.lower() in c.lower() for c in paper.concepts + paper.keywords):
                concept_papers.append(paper)
        tier_order = {'foundational': 0, 'recent': 1, 'trending': 2}
        concept_papers.sort(key=lambda p: (tier_order[p.tier], p.year))
        for i, paper in enumerate(concept_papers):
            if i == 0:
                evolution_path.append((paper, "INTRODUCES_CONCEPT"))
            else:
                prev_paper = concept_papers[i-1]
                relationship = self._get_relationship_between(prev_paper.paper_id, paper.paper_id)
                evolution_path.append((paper, relationship or "CONTINUES"))
        return evolution_path
    def get_competing_approaches(self, paper_id: str, max_papers: int = 5) -> List[PaperNode]:
        competing_papers = []
        for _, target, data in self.kg.graph.edges(paper_id, data=True):
            if data.get('relationship_type') == 'COMPETING_METHOD':
                competing_papers.append(self.kg.papers[target])
        source_paper = self.kg.papers[paper_id]
        for other_id, other_paper in self.kg.papers.items():
            if (other_id != paper_id and other_paper.tier == source_paper.tier and len(set(source_paper.domains) & set(other_paper.domains)) > 0 and len(set(source_paper.methods) & set(other_paper.methods)) == 0):
                competing_papers.append(other_paper)
        return competing_papers[:max_papers]
    def get_methodology_overview(self) -> Dict[str, List[PaperNode]]:
        methodology_map = defaultdict(list)
        for paper_id, paper in self.kg.papers.items():
            if paper.contribution_type == 'methodology':
                for method in paper.methods:
                    methodology_map[method].append(paper)
        for method in methodology_map:
            methodology_map[method].sort(key=lambda p: p.influence_score + p.citation_count/1000, reverse=True)
        return dict(methodology_map)
    def get_survey_outline(self) -> Dict[str, List[PaperNode]]:
        outline = {
            'background': [],
            'related_work': [],
            'recent_advances': [],
            'current_work': [],
            'future_directions': [],
            'supplementary': []
        }
        for paper_id, paper in self.kg.papers.items():
            section = paper.recommended_section
            if section in outline:
                outline[section].append(paper)
        for section in outline:
            outline[section].sort(key=lambda p: p.influence_score + (p.citation_count / 1000), reverse=True)
        return outline
    def _get_relationship_between(self, paper1_id: str, paper2_id: str) -> Optional[str]:
        if self.kg.graph.has_edge(paper1_id, paper2_id):
            edge_data = self.kg.graph[paper1_id][paper2_id]
            if edge_data:
                return list(edge_data.values())[0].get('relationship_type')
        return None
