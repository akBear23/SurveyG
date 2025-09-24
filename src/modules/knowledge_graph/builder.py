"""
Builder for hierarchical knowledge graph from survey papers.
"""
from .graph import HierarchicalKnowledgeGraph

def create_knowledge_graph_from_survey_papers(survey_papers):
    """Create a hierarchical knowledge graph from survey papers"""
    print("🔧 Building Hierarchical Knowledge Graph...")
    kg = HierarchicalKnowledgeGraph()
    print(f"📄 Adding {len(survey_papers)} papers to knowledge graph...")
    for paper in survey_papers:
        kg.add_paper(paper)
    print("🔗 Building relationships between papers...")
    kg.build_relationships()
    print(f"✅ Knowledge graph created with:")
    print(f"   📊 {len(kg.papers)} papers")
    print(f"   🔗 {kg.graph.number_of_edges()} relationships")
    print(f"   🏛️  {len(kg.tier_nodes['foundational'])} foundational papers")
    print(f"   🆕 {len(kg.tier_nodes['recent'])} recent papers")
    print(f"   📈 {len(kg.tier_nodes['trending'])} trending papers")
    rel_counts = {}
    for _, _, data in kg.graph.edges(data=True):
        rel_type = data.get('relationship_type', 'unknown')
        rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
    print("   📑 Relationship type counts:")
    for rel_type, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"      {rel_type}: {count}")
    return kg
