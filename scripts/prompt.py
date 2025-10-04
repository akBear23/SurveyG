class PromptHelper:
    def __init__(self):
        # workflow 
        # 1 - bfs (done)
        # 2 - layer 1 
        # 3 - K community (done)
        # 4 - Outline (done)
        # 5 - Evaluate 
        self.BFS_PROMPT = """You are an expert academic researcher analyzing the evolution of research in "[QUERY]".

TASK: Analyze the following citation path where each paper builds upon previous work.

CITATION PATH [NUMBER_OF_PAPERS] papers:
[PAPER_INFO]
[PREVIOUS_CONTEXT]

ANALYSIS REQUIREMENTS:
For this development path, provide:

1. **Methodological Evolution** (2-3 sentences):
- What are the key methodological shifts or innovations?
- How do methods evolve from foundational to recent work?

2. **Knowledge Progression** (3-4 sentences):
- What problems are being addressed?
- How does each paper build on limitations of previous work?
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
- Total length: 400-600 words

Provide a scholarly yet concise analysis.
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

First, cluster the papers into distinct subgroups based on a critical analysis of their contributions, methodologies, and thematic scope. Then, provide a structured summary that not only describes these groups but also critiques and compares the papers within and across them.

PAPER INFORMATION: 
[PAPER_INFO]
*Output your analysis in the following exact structure:*

1.  <think>
    Explain your reasoning for how you clustered papers into subgroups based on their methodologies, contributions, and thematic scope.
    </think>

2.  *For each subgroup:*
    *   *Subgroup name*: [Clear descriptive name]
    *   *Papers*: [List paper titles and years]
    *   *Analysis*: Provide 2-4 paragraphs covering:
        *   *Core methodologies and approaches*: Describe the common technical or methodological toolkit shared by these papers.
        *   *Thematic focus and key contributions*: Explain the specific problem or aspect of "[QUERY]" this group addresses and their main intellectual contributions to the field.
        *   *Critique and comparison*: Act as a critic. Compare the papers within the subgroup. How do they relate? Which one introduced a key innovation? What are the shared or individual limitations (e.g., scalability, assumptions, evaluative rigor)?

3.  *Overall Perspective* (3-6 sentences):
    Synthesize the intellectual trajectory of the research area. How have the subgroups evolved over time? How do they relate to or build upon one another? Identify the key transitions, paradigm shifts, or unresolved tensions between the methodological approaches.
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
   - Conclusion with theoretical gaps, practical challenges, ethical considerations

4. **Evidence Tracking**:
   - Link each section to supporting evidence from both taxonomies and development paths
   - Use layer numbers (1,2,3) for taxonomy-based sections
   - Use seed IDs for development-path-based sections
   - Key papers should be distributed across relevant sections

# OUTPUT FORMAT (JSON ONLY):
For each section, add the section and its subsections titles in a hierarchical manner in the same 'section_outline'
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
- Each section_focus should be 50-100 words

"""
        
        self.EVALUATE_OUTLINE_PROMPT = """
    Evaluate the quality and structure of the following literature review outline for the topic '[QUERY]'. Assess whether the outline demonstrates meaningful organization of works rather than a simple concatenation of summaries. Your feedback should include:
    • Strengths of the outline
    • Weaknesses or issues (if any)
    • The outline structure MUST include Introduction and Conclusion, check if the given outline have these sections and suggest improvement if missing.
    • Specific suggestions for improvement (only if issues are found).
    Outline to evaluate: [OUTLINE_TEXT]
    """
    
        self.WRITE_INITIAL_SECTION_PROMPT = """
        Write a comprehensive literature review section titled "[SECTION_TITLE]" in LaTeX format.
        
        **SECTION SPECIFIC FOCUS:** [SECTION_FOCUS]

        **Section taxonomies summaries and development directions:** [PROOFS_TEXT]

        **CRITICAL REQUIREMENTS:**
        1. The generated text have to be in LaTeX, use proper LaTeX citations (\\cite{{citation_key}}) throughout the text
        2. Focus ONLY on the specific aspect assigned to this section
        3. Academic writing style with critical analysis
        4. Synthesize information across papers, don't just list them
        5. At least 500 words for this section
        6. The sub sections and sub sub sections have to follow the given section outline, about 200 words for each sub section and each sub sub section. Before creating sub sections, ensure that the main section has provide a comprehensive overview of the content in this section, at least 100 words.
        7. Include specific examples and evidence with proper citations
        8. Provide critical evaluation and comparative analysis
        9. Ensure coherent organization and logical flow, make sure to include the taxonomies summaries and development directions in the section.
        10. Only add the content do not need to add the numbering inside the section or subsection titles.

        **Available Citations:**
        [CITATION_INFO]
        **SECTION OUTLINE:** 
        [OUTLINE]
        **Papers to reference:**
        [PAPERS_SUMMARY]
        
        ***Previous section if any:** [PRE_SECTION]
        Write ONLY the content for "[SECTION_TITLE]" section. 
        Focus specifically on: [SECTION_FOCUS]
        
        Ensure the section demonstrates:
        - Comprehensive coverage of the focus area
        - High citation density (aim for 8-10 citations minimum)
        - Academic rigor and analytical depth
        - Synthesis across multiple papers
        - Critical evaluation of approaches/findings
        """
        self.EVALUATE_SECTION_PROMPT = """
        Evaluate the quality of this literature review section based on the following criteria:
        
        **Previous section if any:** [PRE_SECTION]
        **Section Title**: [SECTION_TITLE]
        **Expected Focus**: [SECTION_FOCUS]
        **Overall Review Context: Outline**: [OUTLINE]
        
        **Section Content**:
        [SECTION_CONTENT]
        
        **Evaluation Criteria** (Rate each from 1-5, where 5 is excellent):
        
        1. **Content Coverage** (1-5): Does the section comprehensively cover the expected focus area?
        2. **Citation Density** (1-5): Are there sufficient and appropriate citations throughout the text?
        3. **Academic Rigor** (1-5): Is the writing style academic and analytical rather than descriptive?
        4. **Synthesis Quality** (1-5): Does it synthesize information across papers rather than just listing findings?
        5. **Critical Analysis** (1-5): Does it provide critical evaluation and comparative analysis?
        6. **Coherence** (1-5): Is the content well-organized and logically structured?
        7. **Depth of Analysis** (1-5): Does it provide sufficient depth rather than surface-level discussion?
        8. **Specificity** (1-5): Does it focus specifically on the assigned scope without overlap with other sections?
        
        **IMPORTANT**: 
        - Return ONLY valid JSON without any markdown formatting or code blocks
        - Escape all backslashes and quotes properly in JSON strings
        - Do not include any special characters that might break JSON parsing
        
        **Response Format** (JSON only):
        {{
            "overall_score": <average_score>,
            "individual_scores": {{
                "content_coverage": <score>,
                "citation_density": <score>,
                "academic_rigor": <score>,
                "synthesis_quality": <score>,
                "critical_analysis": <score>,
                "coherence": <score>,
                "depth_of_analysis": <score>,
                "specificity": <score>
            }},
            "strengths": ["list of strengths"],
            "weaknesses": ["list of weaknesses"],
            "is_satisfactory": <true/false>,
            "improvement_needed": ["specific areas needing improvement"],
            "suggested_queries": ["suggested search queries to find additional relevant papers"]
        }}
        
        Consider a section satisfactory if overall_score >= 3.5 and no individual score is below 3.0.
        """
        self.SECTION_IMPROVE_PROMPT = """
        Improve the following literature review section based on evaluation feedback and additional papers.
        
        **Section Title**: [SECTION_TITLE]
        **Section Focus**: [SECTION_FOCUS]
        
        **Current Section Content**:
        [CURRENT_CONTENT]
        
        **Evaluation Feedback**:
        - Overall Score: {evaluation_feedback.get('overall_score', 'N/A')}/5
        - Strengths: {', '.join(evaluation_feedback.get('strengths', []))}
        - Weaknesses: {', '.join(evaluation_feedback.get('weaknesses', []))}
        - Areas for Improvement: {', '.join(evaluation_feedback.get('improvement_needed', []))}
        
        **Additional Papers Retrieved**:
        {additional_info}
        
        **Improvement Instructions**:
        1. Address the specific weaknesses identified in the evaluation
        2. Incorporate relevant information from the additional papers
        3. Improve citation density and academic rigor
        4. Enhance synthesis and critical analysis
        5. Ensure the content stays focused on: {section_focus}
        6. Maintain academic writing style
        7. Use proper LaTeX citations (\\cite{{citation_key}})
        
        **Requirements**:        
        1. The generated text have to be in LaTeX, use proper LaTeX citations (\\cite{{citation_key}}) throughout the text
        2. Focus ONLY on the specific aspect assigned to this section
        3. Academic writing style with critical analysis
        4. Synthesize information across papers, don't just list them
        5. At least 800 words for this section
        6. The sub sections and sub sub sections have to follow the given section outline, about 300-800 words for each sub section and each sub sub section. Before creating sub sections, ensure that the main section has provide a comprehensive overview of the content in this section, at least 300 words.
        7. Include specific examples and evidence with proper citations
        8. Provide critical evaluation and comparative analysis
        9. Ensure coherent organization and logical flow

        Write the improved section content only:
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
