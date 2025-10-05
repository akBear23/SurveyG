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

1.  <think>
Explain your reasoning for how you clustered papers into subgroups based on their methodologies, contributions, and thematic scope.
</think>
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

3. **Writing Quality Standards**:
   - Avoid repetitive transitional phrases; use varied language
   - Create clear section boundaries with no content overlap

4. **Section Design Guidelines**:
   
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

5. **Evidence Tracking**:
   - Each subsection MUST be grounded with its own proof_ids showing supporting evidence
   - proof_ids guidelines: Use 1-3 identifiers per subsection when evidence is available
   - Each section's section_focus should synthesize the broader theme, while subsection_focus provides specific detail
   - Use layer numbers (1,2,3) for taxonomy-based sections
   - Use seed IDs for development-path-based sections
   - Key papers should be distributed across relevant sections

6. **Subsection Design Principles**:
    - Each subsection should be self-contained enough to understand its contribution
    - subsection_focus should clearly explain WHAT concepts/methods are covered, not just WHY they matter
    - Subsections within a section should follow a logical progression (e.g., chronological, simple-to-complex, general-to-specific)
    - Avoid redundancy between subsection_focus descriptions within the same section

# OUTPUT FORMAT (JSON ONLY):
For each section, add the section and its subsection titles in a hierarchical manner in the same 'section_outline'
For each section, add a paragraph to the key 'section_focus' to indicate the main focus of that section
For each section, add an id (taken from the information layer number or the development seed ids) to the key 'proof_ids' to indicate the proof for each section, if proof is from the taxonomy of layer 1 put "layer_1", if the proof is from the paper community, put id of that community (example: community_0, community_1,...) if the proof is from the development direction, put the seed(s) paper id.
    
Return a JSON array where each element represents ONE main section with its subsections:

```json
[
  {
    "section_number": "1",
    "section_title": "Introduction",
    "section_focus": "This section establishes the foundational context for knowledge graph embeddings. It begins by explaining the evolution of knowledge representation, introduces the core challenges that knowledge graph embeddings address, and delineates the scope and organization of this review. The section sets the stage for understanding why embedding methods have become central to modern knowledge graph research and applications.",
    "subsections": [
      {
        "number": "1.1",
        "title": "Background: Knowledge Graphs",
        "subsection_focus": "Introduces the fundamental concepts of knowledge graphs, their structure as networks of entities and relations, and their historical development from semantic networks to modern large-scale knowledge bases. Discusses key examples like Freebase, DBpedia, and Wikidata, highlighting their role in organizing world knowledge and enabling intelligent systems.",
        "proof_ids": ["layer_1", "community_2", "68f34ed64fdf07bb1325097c93576658e061231e"]
      },
      {
        "number": "1.2",
        "title": "Role of KG Embedding",
        "subsection_focus": "Explains the motivation for embedding knowledge graphs into continuous vector spaces. Covers the limitations of symbolic representations, the advantages of distributed representations for reasoning and prediction, and how embeddings enable scalability and integration with modern machine learning pipelines. Establishes embedding as a bridge between symbolic and neural approaches.",
        "proof_ids": ["community_1"]
      }
    ]
  },
  ...
]
```

CRITICAL JSON REQUIREMENTS:
- Return ONLY valid JSON (no markdown, no code blocks, no json wrapper)
- Each section MUST have: section_number, section_title, section_focus, and subsections array
- Each subsection MUST have: number, title, subsection_focus, and proof_ids array
- section_focus: 100-150 words describing the main section's overall purpose
- subsection_focus: 100-150 words describing what that specific subsection covers
- All strings must properly escape special characters (quotes, newlines, etc.)
- Subsection numbers must match parent section (e.g., section 2 has subsections 2.1, 2.2, 2.3)
"""
        
        self.EVALUATE_OUTLINE_PROMPT = """
You are a grumpy expert academic researcher carefully evaluating this literature review outline.
Evaluate the literature review outline for: '[QUERY]'

# EVALUATION CRITERIA

## 1. Structural Philosophy Compliance (CRITICAL)
- [ ] Follows PEDAGOGICAL PROGRESSION: Foundations → Core Methods → Advanced Topics → Applications → Future
- [ ] Includes mandatory sections: Introduction, Foundational Concepts, Conclusion
- [ ] Has [MIN_MAIN_SECTIONS]-[MAX_MAIN_SECTIONS] main body sections
- [ ] Each main section has [MIN_SUBSECTIONS]-[MAX_SUBSECTIONS] subsections
- [ ] Uses ONLY TWO levels of hierarchy (no deeper nesting beyond 2.1, 2.2)
- [ ] Balances chronological development with thematic depth

## 2. Section Design & Progression
- [ ] **Early Sections**: Section 1 establishes motivation/scope, Section 2 covers essential background
- [ ] **Middle Sections**: Organized by major methodological families with progression from foundational to advanced
- [ ] **Later Sections**: Address modern developments, applications, and forward-looking content
- [ ] Clear narrative arc from foundational concepts through cutting-edge developments
- [ ] Contextual grounding with prerequisite knowledge before meta-learning specifics

## 3. Content Organization Principles
- [ ] Thematic organization using available research synthesis (layer taxonomies, paper communities, development directions)
- [ ] Methodological depth with related approaches grouped together
- [ ] Practical relevance with dedicated applications sections
- [ ] Forward-looking content addressing emerging trends and ethical considerations
- [ ] Shows connections and evolution between works, not just listing

## 4. Evidence Integration & Tracking
- [ ] Each subsection has its own proof_ids showing supporting evidence
- [ ] proof_ids use appropriate identifiers (layer numbers, community IDs, seed IDs)
- [ ] Key papers distributed logically across relevant sections
- [ ] Evidence properly supports the claimed thematic organization

## 5. Writing Quality Standards
- [ ] Avoids repetitive transitional phrases with varied language
- [ ] Clear section boundaries with no content overlap
- [ ] Section_focus synthesizes broader themes (100-150 words)
- [ ] Subsection_focus clearly explains WHAT concepts/methods are covered (100-150 words)
- [ ] Subsections within each section follow logical progression (chronological, simple-to-complex, etc.)
- [ ] No redundancy between subsection_focus descriptions within same section

## 6. Technical Requirements
- [ ] Valid JSON structure with proper escaping
- [ ] All required fields present: section_number, section_title, section_focus, subsections array
- [ ] Each subsection has: number, title, subsection_focus, proof_ids array
- [ ] Proper numbering sequence (section 2 has subsections 2.1, 2.2, 2.3)

# OUTPUT FORMAT

PASS/FAIL: [Overall assessment based on CRITICAL criteria]

Critical Issues (must fix):
- [List any violations of structural requirements, evidence tracking, or technical validity]

Strengths:
- [What works well in pedagogical progression, evidence integration, and narrative structure]

Weaknesses:
- [Issues with section design, content organization, or writing quality]

Specific Recommendations:
1. [Most critical improvement needed for structural or evidence compliance]
2. [Secondary improvements for content organization]
3. [Writing quality enhancements]

Revised Section Suggestions (if structural changes needed):
[Provide specific rewrites for problematic sections with explanations of how they better follow the pedagogical progression]

---
Outline to evaluate:
[OUTLINE_TEXT]
"""
        self.WRITE_INITIAL_SUBSECTION_PROMPT = """
Write a comprehensive literature review subsection titled "[SUBSECTION_TITLE]" in LaTeX format.

**SUBSECTION SPECIFIC FOCUS:** [SUBSECTION_FOCUS]
**Subsection taxonomies summaries and development directions:** [PROOFS_TEXT]

**CRITICAL REQUIREMENTS:**

1. **Content & Format:**
  - Generate text in LaTeX format with proper citations (\\cite{{citation_key}})
  - Focus ONLY on the specific aspect assigned to this subsection
  - At least 400 words for this subsection
  - No numbering in subsection titles

2. **Critical Analysis and Synthesis (HIGH PRIORITY):**
  - Go beyond description - evaluate methodological strengths/weaknesses.
  - Compare approaches critically: "While [X] achieves Y, it fails to address Z unlike [W]" - use the taxonomies to inform these comparisons.
  - Identify WHY gaps/limitations exist (theoretical barriers, practical constraints) as suggested by the development directions.
  - Analyze trade-offs explicitly (performance vs. privacy, accuracy vs. robustness) and relate to the taxonomies.
  - Synthesize information ACROSS papers - identify patterns, contradictions, and evolutionary trends as outlined in both the taxonomies and the development directions.

3. **Evidence & Citation:**
  - High citation density (6-8 citations minimum)
  - Use specific examples with citations
  - Include comparative statements

4. **Organization:**
  - Coherent flow with smooth transitions
  - Integrate taxonomies summaries and development directions naturally

**ANALYTICAL FRAMEWORK - Apply these questions:**
- What are the methodological limitations not discussed by the authors?
- How do experimental setups affect generalizability?
- What assumptions are made and are they realistic?
- Where do findings contradict each other and why?
- What theoretical gaps prevent solving identified problems?

**RED FLAGS to avoid:**
- Sequential listing: "Paper A proposes X. Paper B proposes Y. Paper C proposes Z."
- Vague critique: "This approach has limitations" (without specifying what/why)
- Unsupported claims: "This is significant" or "This is important" (without explaining why/how)
- Missing comparisons: Discussing methods in isolation without relating them
- Pure description: Describing what a paper does without analyzing its contribution or limitations
- Ignoring contradictions: Not addressing when papers reach conflicting conclusions
- Cherry-picking: Only citing results that support a narrative, ignoring contradictory evidence

**Available Citations:**
[CITATION_INFO]

**SECTION OUTLINE:**
[OUTLINE]

**Papers to reference:**
[PAPERS_SUMMARY]

**Previous subsection if any:**
[PRE_SUBSECTION]

**OUTPUT INSTRUCTIONS:**
Write ONLY the content for "[SUBSECTION_TITLE]" subsection focusing on: [SUBSECTION_FOCUS]
"""

        self.EVALUATE_SUBSECTION_PROMPT = """
        You are a grumpy expert academic researcher carefully evaluating this literature review subsection.
        Evaluate the quality of this literature review subsection based on the following criteria:
        
        **Previous subsection if any:** [PRE_SUBSECTION]
        **Subsection Title**: [SUBSECTION_TITLE]
        **Expected Focus**: [SUBSECTION_FOCUS]
        **Overall Review Context: Outline**: [OUTLINE]
        
        **Subsection Content**:
        [SUBSECTION_CONTENT]
        
        **Evaluation Criteria** (Rate each from 1-5, where 5 is excellent):
        
        1. **Content Coverage** (1-5): Does the subsection comprehensively cover the expected focus area?
        2. **Citation Density** (1-5): Are there sufficient and appropriate citations throughout the text?
        3. **Academic Rigor** (1-5): Is the writing style academic and analytical rather than descriptive?
        4. **Synthesis Quality** (1-5): Does it synthesize information across papers rather than just listing findings?
        5. **Critical Analysis** (1-5): Does it provide critical evaluation and comparative analysis?
        6. **Coherence** (1-5): Is the content well-organized and logically structured?
        7. **Depth of Analysis** (1-5): Does it provide sufficient depth rather than surface-level discussion?
        8. **Specificity** (1-5): Does it focus specifically on the assigned scope without overlap with other subsections?
        
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
        
        Consider a subsection satisfactory if overall_score >= 3.5 and no individual score is below 3.0.
        """

        self.SUBSECTION_IMPROVE_PROMPT = """
Improve the following literature review subsection based on evaluation feedback and additional papers.

**Previous subsection if any:** [PRE_SUBSECTION]
**Subsection Title**: [SUBSECTION_TITLE]
**Subsection Focus**: [SUBSECTION_FOCUS]
**Overall Review Context: Outline**: [OUTLINE]

**Current Subsection Content**:
[CURRENT_CONTENT]

**Evaluation Feedback**:
- Overall Score: [OVERALL_SCORE]
- Synthesis Quality: [SYNTHESIS_SCORE] (Target: 4.5+)
- Critical Analysis: [CRITICAL_SCORE] (Target: 4.5+)
- Strengths: [STRENGTH]
- Weaknesses: [WEAKNESS]
- Specific Improvements Needed: [IMPROVEMENT_NEEDED]

**Additional Papers Retrieved**:
[ADDITIONAL_INFO]

**MANDATORY IMPROVEMENT ACTIONS:**

 **Improvement Instructions**:
  1. Address the specific weaknesses identified in the evaluation
  2. Incorporate relevant information from the additional papers
  3. Improve citation density and academic rigor
  4. Enhance synthesis and critical analysis
  5. Ensure the content stays focused on: [SUBSECTION_FOCUS]
  6. Maintain academic writing style
  7. Use proper LaTeX citations (\\cite{{citation_key}})

**Requirements**:
  1. The generated text have to be in LaTeX, use proper LaTeX citations (\\cite{{citation_key}}) throughout the text
  2. Focus ONLY on the specific aspect assigned to this subsection
  3. Academic writing style with critical analysis
  4. Synthesize information across papers, don't just list them
  5. At least 400 words for this subsection
  6. Include specific examples and evidence with proper citations
  7. Provide critical evaluation and comparative analysis
  8. Ensure coherent organization and logical flow

  Ensure the subsection demonstrates:
  - Comprehensive coverage with deep critical evaluation
  - Explicit synthesis showing relationships between studies
  - Analytical depth beyond mere description
  - Critical comparison of approaches with justified assessments
  - Discussion of WHY limitations exist, not just WHAT they are

Write the improved subsection content only (no meta-commentary):
"""
        self.WRITE_INITIAL_SECTION_OVERVIEW_PROMPT = """
Write an introductory overview for the literature review section titled "[SECTION_TITLE]" in LaTeX format. This overview should set the stage for the detailed discussions within its subsections.

**SECTION SPECIFIC FOCUS:** [SECTION_FOCUS]

**CRITICAL REQUIREMENTS:**

1.  **Content & Format:**
    *   Generate text in LaTeX format.
    *   The overview should be between 200-300 words.
    *   It should clearly introduce the main themes and sub-topics that will be covered in the subsections of this section.
    *   Briefly explain the significance of this section within the broader context of the literature review on "[QUERY]".
    *   Highlight the key challenges or advancements that this section will explore, drawing on the provided proofs and paper summaries at a high level.
    *   Avoid diving into specific details of individual papers; save that for the subsections.
    *   No numbering or bullet points.

2.  **Synthesis & Narrative:**
    *   Synthesize the core ideas from the proof IDs (layer taxonomies, community summaries, development directions) and paper summaries relevant to this section's focus.
    *   Create a coherent narrative that logically connects the overarching theme of the section to the general contributions of the papers.
    *   Explain how this section builds upon or diverges from the content of the `PRE_SECTION`.

3.  **Evidence & Citation (High-level):**
    *   Reference the `PROOFS_TEXT` to inform the overall themes and trends discussed in the overview.
    *   You may use a few high-level citations (e.g., "\\cite{citation_key}") to establish the primary works relevant to the section's focus, but avoid excessive detailed citation here. The focus is on the narrative flow.

**SECTION TAXONOMIES SUMMARIES AND DEVELOPMENT DIRECTIONS:**
[PROOFS_TEXT]

**Available Citations:**
[CITATION_INFO]

**SECTION OUTLINE (for full context):**
[OUTLINE]

**Papers to reference (high-level themes):**
[PAPERS_SUMMARY]

**Previous section if any (for contextual flow):**
[PRE_SECTION]

**OUTPUT INSTRUCTIONS:**
Write ONLY the content for the introductory overview paragraph(s) of the "[SECTION_TITLE]" section, focusing on: [SECTION_FOCUS].
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
