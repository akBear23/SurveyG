class PromptHelper:
    def __init__(self):
        # workflow 
        # 1 - bfs (done)
        # 2 - layer 1 
        # 3 - K community (done)
        # 4 - Outline (done)
        # 5 - Evaluate 
        self.BFS_PROMPT = """You are an expert academic researcher analyzing the evolution of research in "[QUERY]".

TASK: Analyze the following citation path, where each paper builds upon previous work.

CITATION PATH [NUMBER_OF_PAPERS] papers:
[PAPER_INFO]
[PREVIOUS_CONTEXT]

ANALYSIS REQUIREMENTS:
For this development path, provide:

1. **Methodological Evolution** (2-3 sentences):
- What are the key methodological shifts or innovations?
- How do methods evolve from foundational to recent work?

2. **Knowledge Progression** (4-6 sentences):
- What problems are being addressed?
- How does each paper build on the limitations of previous work?
- What new capabilities or insights emerge?

3. **Temporal Context** (1-2 sentences):
- How does publication timing relate to technological/theoretical advances?
- Are there notable gaps or acceleration periods?

4. **Synthesis** (2-3 sentences):
- What unified narrative connects these works?
- What is the collective contribution to "[QUERY]"?

CONSTRAINTS:
- Be specific and cite paper numbers (e.g., "Paper 3 introduces...")
- Focus on connections and evolution, not just individual contributions
- Avoid generic statements; ground analysis in actual methods/results
- Highlight key transitions with phrases like "building on this..." or "in contrast..."
- Total length: 500-700 words
"""
        self.LAYER_PROMPT = """You are analyzing the [LAYER_DESCRIPTION] in "[QUERY]" research.

PAPERS TO ANALYZE ([NUMBER_OF_PAPER] papers):
[PAPER_INFO]

TASK: Create a thematic taxonomy organizing these works into coherent methodological groups.

OUTPUT STRUCTURE:

1. **Overview** (2-3 sentences):
   - What characterizes this layer's contributions?
   - What problems/challenges do these works address?

2. **Methodological Groups** (Identify 3-5 main approaches):
   For each group:
   - **Group Name**: [Descriptive name]
   - **Core Approach**: [1-2 sentences on methodology]
   - **Key Works**: [List paper numbers, e.g., Papers 1, 5, 12]
   - **Contribution**: [How this group advances the field]

3. **Cross-Group Patterns** (2-3 sentences):
   - What common trends or complementary approaches exist?
   - How do groups relate or differ?

4. **Layer Significance** (1-2 sentences):
   - What is this layer's overall impact on "[QUERY]"?

CONSTRAINTS:
- Be specific about methods (algorithms, architectures, techniques)
- Groups should be distinct yet comprehensive (cover all papers)
- Cite paper numbers explicitly
- Length: 500-700 words

Provide a structured, analytical taxonomy."""
        
        self.COMMUNITY_PROMPT = """
*Instruction:* You are a research analyst synthesizing a body of literature. Your task is to analyze the provided papers on the survey topic "[QUERY]".

TASK: Cluster the provided papers into 2-3 distinct subgroups based on their methodologies, contributions, or thematic focus. Then provide a critical analysis of each cluster and the field overall.

PAPER INFORMATION: 
[PAPER_INFO]
*Output your analysis in the following exact structure:*

1.  Explain your reasoning for how you clustered papers into subgroups based on their methodologies, contributions, and thematic scope.

2.  *For each subgroup:*
    *   *Subgroup name*: [Clear descriptive name]
    *   *Papers*: [List paper titles and years]
    *   *Analysis*: Provide 4-6 sentences covering:
        *   *Core methodologies and approaches*: Describe the common technical or methodological toolkit shared by these papers.
        *   *Thematic focus and key contributions*: Explain the specific problem or aspect of "[QUERY]" this group addresses and their main intellectual contributions to the field.
        *   *Critique and comparison*: Act as a critic. Compare the papers within the subgroup. How do they relate? Which one introduced a key innovation? What are the shared or individual limitations (e.g., scalability, assumptions, evaluative rigor)?

3.  *Overall Perspective* (3-4 sentences):
    Synthesize the intellectual trajectory of the research area. How have the subgroups evolved? How do they relate to or build upon one another? Identify the key transitions, paradigm shifts, or unresolved tensions between the methodological approaches.
GUIDELINES:
- Reference papers specifically: "Paper X introduces..." or "Unlike Paper Y..."
- Be critical but fair: identify limitations without dismissing contributions
- Focus on connections, not just descriptions
- Total target length: 500-700 words
"""
        self.OUTLINE_PROMPT = """You are creating a comprehensive literature review outline for: "[QUERY]"

# AVAILABLE RESEARCH SYNTHESIS

## Layer Taxonomies (Thematic Organization)
[LAYER_TAXONOMIES]

## Paper communities 
[PAPER_COMMUNITIES]

## Development Directions (Evolutionary Paths)
[DEVELOPMENT_DIRECTIONS]

# TASK: Create Literature Review Outline

REQUIREMENTS:
1. **Structure Philosophy**:
   - Follow a PEDAGOGICAL PROGRESSION: Foundations → Core Methods → Advanced Topics → Applications → Future
   - Must include: Introduction, Foundational Concepts, Conclusion sections
   - [MIN_MAIN_SECTIONS]-[MAX_MAIN_SECTIONS] main body sections
   - Each main section: [MIN_SUBSECTIONS]-[MAX_SUBSECTIONS] subsections
   - Use only TWO levels of hierarchy (e.g., 2.1, 2.2). Do NOT create deeper nesting (e.g., 2.1.1, 2.1.2)
   - Balance CHRONOLOGICAL DEVELOPMENT with THEMATIC DEPTH

2. **Content Principles**:
   - **Narrative Arc**: Tell a coherent story from foundational concepts through cutting-edge developments
   - **Contextual Grounding**: Start with prerequisite knowledge before diving into meta-learning specifics
   - **Methodological Depth**: Group related approaches together 
   - **Practical Relevance**: Include dedicated sections for applications and real-world impact
   - **Forward-Looking**: Address emerging trends, challenges, and ethical considerations
   - Show connections and evolution between works, not just list them

3. **Section Design Guidelines**:
   
   **Early Sections (Foundation Building)**:
   - Section 1: Motivation and scope
   - Section 2: Essential background/foundational concepts from prerequisite fields
   - Sections 3-4: Core paradigms and early breakthroughs
   
   **Middle Sections (Methodological Depth)**:
   - Organize by major methodological families
   - Within each family: progression from foundational to advanced techniques
   - Show cross-connections between families
   
   **Later Sections (Modern Developments & Impact)**:
   - Recent trends and cutting-edge innovations
   - Dedicated applications section showing real-world impact
   - Conclusion with theoretical gaps, practical challenges, and ethical considerations

4. **Evidence Tracking**:
   - Link each section to supporting evidence from both taxonomies and development paths
   - Use layer numbers (1,2,3) for taxonomy-based sections
   - Use seed IDs for development-path-based sections
   - Key papers should be distributed across relevant sections

# OUTPUT FORMAT (JSON ONLY):
For each section, add the section and its subsection titles in a hierarchical manner in the same 'section_outline'
For each section, add a paragraph to the key 'section_focus' to indicate the main focus of that section
For each section, add an id (taken from the information layer number or the development seed ids) to the key 'proof_ids' to indicate the proof for each section, if proof is from the taxonomy of layer 1 put "layer_1", if the proof is from the paper community, put id of that community (example: community_0, community_1,...) if the proof is from the development direction, put the seed(s) paper id.
    
Return a JSON array where each element represents ONE section:

```json
  [
        "section_outline": "### 1. Introduction
    *   1.1. Background: Knowledge Graphs and Their Significance
    *   1.2. The Role of Knowledge Graph Embedding
    *   1.3. Scope and Organization of the Review",
        "section_focus": "This section introduces Knowledge Graphs (KGs) and the fundamental problem of Knowledge Graph Embedding (KGE). It highlights the importance of KGE for various AI tasks and outlines the scope and structure of the review.",
        "proof_ids": [layer_1, community_2]
      ], 
      ...
```

CRITICAL JSON REQUIREMENTS:
- Return ONLY valid JSON (no markdown, no code blocks, no ```json```)
- Escape special characters properly
- Section titles should be numbered (1., 2., 3., etc.)
- Subsections numbered relative to parent (2.1., 2.2., etc.)
- Each section_focus should be approximately 100-150 words

"""
        
        self.EVALUATE_OUTLINE_PROMPT = """
Evaluate the literature review outline for: '{query}'

# EVALUATION CRITERIA

## 1. Structural Requirements (CRITICAL)
- [ ] Has a clear Introduction section establishing scope and purpose
- [ ] Has a substantive Conclusion section with synthesis and implications  
- [ ] Contains 6-9 main body sections
- [ ] Each main section has 2-4 subsections
- [ ] Uses ONLY 2-level nesting (e.g., 2.1, 2.2) - NO deeper nesting
- [ ] Proper numbering sequence and logical hierarchy

## 2. Conceptual Organization  
- [ ] Follows logical progression appropriate for the domain
- [ ] Early sections establish context, definitions, and foundations
- [ ] Middle sections develop core themes/methodologies systematically
- [ ] Later sections address applications, controversies, or future directions
- [ ] Clear narrative arc that builds understanding progressively
- [ ] Transitions between sections feel natural

## 3. Content Quality & Coverage
- [ ] Shows thematic organization rather than just chronological listing
- [ ] Demonstrates awareness of connections between research streams
- [ ] Coverage appears balanced across relevant subfields
- [ ] Each section has a clear, focused purpose (check section_focus)
- [ ] Avoids redundancy and shows clear section boundaries
- [ ] Includes synthesis and critical analysis, not just summary

## 4. Evidence Integration
- [ ] Each substantive section has appropriate proof_ids
- [ ] proof_ids correctly reference available evidence sources
- [ ] Key words distributed logically across relevant sections
- [ ] Evidence supports the claimed thematic organization

## 5. Technical Requirements  
- [ ] Valid JSON structure with proper escaping
- [ ] All required fields present and properly formatted
- [ ] Section titles are clear and descriptive

# OUTPUT FORMAT

**PASS/FAIL**: [Overall assessment based on CRITICAL criteria]

**Critical Issues** (must fix):
- [List any violations of structural requirements or technical validity]

**Strengths**:
- [What works well in organization and conceptual structure]

**Weaknesses**:
- [Conceptual gaps, organizational issues, or coverage problems]

**Specific Recommendations**:
1. [Most critical improvement needed]
2. [Secondary improvements]
3. [Optional enhancements]

**Revised Section Suggestions** (if structural changes needed):
[Provide specific rewrites for problematic sections with explanations]

---
Outline to evaluate:
{outline_text}
"""
    def generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f'[{k}]', str(paras[k]))
        return prompt

if __name__ == "__main__":
    prompt = PromptHelper()
    test_prompt = prompt.generate_prompt(prompt.EVALUATE_OUTLINE_PROMPT, paras={'{query}': 'knowledge graph embedding', 
                                                                    '{outline_text}': 'Simple outline text'})
    print(test_prompt)
