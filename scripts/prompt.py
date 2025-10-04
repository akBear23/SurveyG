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
        "section_focus": "A 100-150 words paragraph",
        "proof_ids": ["layer_1", "community_2"]
      ], 
      ...
```

CRITICAL JSON REQUIREMENTS:
- Return ONLY valid JSON (no markdown, no code blocks, no ```json```)
- Escape special characters properly
- Section titles should be numbered (1., 2., 3., etc.)
- Subsections numbered relative to parent (2.1., 2.2., etc.)
- section_focus will have 100-150 words.
"""
        
        self.EVALUATE_OUTLINE_PROMPT = """
Evaluate the literature review outline for: '[QUERY]'

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

## 6. Writing Quality Indicators
- [ ] No redundant content across sections
- [ ] Varied language (avoids formulaic phrases)
- [ ] Clear section boundaries without overlap
- [ ] Smooth transitions between major sections

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
[OUTLINE_TEXT]
"""
        self.WRITE_INITIAL_SECTION_PROMPT = """

Write a comprehensive literature review section titled "[SECTION_TITLE]" in LaTeX format.

**SECTION SPECIFIC FOCUS:** [SECTION_FOCUS]

**Section taxonomies summaries and development directions:** [PROOFS_TEXT]

**CRITICAL REQUIREMENTS:**

1. **Content & Format:**

  - Generate text in LaTeX format with proper citations (\\cite{{citation_key}})

  - Focus ONLY on the specific aspect assigned to this section

  - At least 500 words for this section

  - Sub-sections follow the given outline (~200 words each)

  - Main section overview (100+ words) BEFORE creating sub-sections

  - No numbering in section/subsection titles

2. **Academic Rigor & Synthesis (HIGH PRIORITY):**

  - Synthesize information ACROSS papers - identify patterns, contradictions, and evolutionary trends

  - Connect findings to broader themes (e.g., "arms race" dynamics, trade-offs)

  - Make implicit connections EXPLICIT (e.g., "This approach addresses the limitation identified in [X] by...")

  - Develop novel frameworks or taxonomies when synthesizing disparate studies

  - Show how different concepts relate (e.g., data poisoning → evasion attacks)

3. **Critical Analysis (HIGH PRIORITY):**

  - Go beyond description - evaluate methodological strengths/weaknesses

  - Compare approaches critically: "While [X] achieves Y, it fails to address Z unlike [W]"

  - Identify WHY gaps/limitations exist (theoretical barriers, practical constraints)

  - Analyze trade-offs explicitly (performance vs. privacy, accuracy vs. robustness)

  - Question assumptions: "Despite claims of robustness, [X] assumes Y which may not hold in Z scenarios"

  - Highlight contradictory findings and discuss possible reasons

  - For each major approach, address: What works? What doesn't? Why? Under what conditions?

4. **Evidence & Citation:**

  - High citation density (8-10 citations minimum)

  - Use specific examples with citations

  - Include comparative statements: "[Paper A] shows X while [Paper B] demonstrates Y under different assumptions"

5. **Organization:**

  - Coherent flow with smooth transitions

  - Integrate taxonomies summaries and development directions naturally

  - Balance depth across sub-sections

**ANALYTICAL FRAMEWORK - Apply these questions:**

- What are the methodological limitations not discussed by the authors?

- How do experimental setups affect generalizability?

- What assumptions are made and are they realistic?

- Where do findings contradict each other and why?

- What theoretical gaps prevent solving identified problems?

**Available Citations:**

[CITATION_INFO]

**SECTION OUTLINE:**

[OUTLINE]

**Papers to reference:**

[PAPERS_SUMMARY]

**Previous section if any:**

[PRE_SECTION]

**OUTPUT INSTRUCTIONS:**

Write ONLY the content for "[SECTION_TITLE]" section focusing on: [SECTION_FOCUS]

Ensure the section demonstrates:

- Comprehensive coverage with deep critical evaluation

- Explicit synthesis showing relationships between studies

- Analytical depth beyond mere description

- Critical comparison of approaches with justified assessments

- Discussion of WHY limitations exist, not just WHAT they are

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
- Overall Score: [OVERALL_SCORE]
- Synthesis Quality: [SYNTHESIS_SCORE] (Target: 4.5+)
- Critical Analysis: [CRITICAL_SCORE] (Target: 4.5+)
- Strengths: [STRENGTH]
- Weaknesses: [WEAKNESS]
- Specific Improvements Needed: [IMPROVEMENT_NEEDED]

**Additional Papers Retrieved**:
[ADDITIONAL_INFO]

**MANDATORY IMPROVEMENT ACTIONS:**

1. **Enhance Synthesis (Priority 1):**
   - ADD 3+ explicit comparative statements per subsection using phrases like:
     * "Unlike [A], [B] addresses X through mechanism Y"
     * "In contrast to [A]'s approach, [B, C] adopt strategy X because Y"
     * "[Method A] and [Method B] converge on principle X but diverge on dimension Y"
   - IDENTIFY patterns: "[Papers A, B, C] all exhibit trade-off between X and Y, suggesting 
     fundamental constraint Z in this problem domain"
   - MAP evolution: "The field progressed from [early assumption/method X] (Papers A, B) to 
     [later recognition Y] (Papers C, D) to [current approach Z] (Papers E, F)"
   - HIGHLIGHT contradictions with analysis: "[A] reports improvement under condition X, 
     while [B] observes degradation under condition Y, indicating sensitivity to Z"
   - BUILD frameworks: "Approaches can be categorized along dimensions [D1, D2], with [methods X] 
     optimizing for [D1] and [methods Y] for [D2]"
   
2. **Deepen Critical Analysis (Priority 2):**
   - For EACH major approach, explain WHY limitations exist:
     * "Cannot handle scenario X because mechanism Y relies on assumption Z, which breaks when W"
     * Not just "has limitations" but the theoretical/practical constraint causing it
   - QUESTION assumptions systematically:
     * "While [Paper X] claims generalizability, it assumes [Y], limiting applicability to domains where [Y] holds"
     * "The evaluation uses metric X, which may overestimate performance because it doesn't account for [realistic constraint Y]"
   - ANALYZE trade-offs with specifics:
     * "Improves accuracy by X% but increases computational cost by Y-fold and requires Z times more training data"
     * Explain when/why trade-off is acceptable or problematic
   - CRITIQUE methodology:
     * "The comparison omits baseline X, making it unclear whether improvement stems from [claimed innovation] or [confounding factor]"
     * "Sample size of N on dataset D limits generalizability to [broader population/scenario]"
   
3. **Add Systematic Comparisons:**
   - Create comparison frameworks: "Three paradigms emerge—[A, B, C]—differing in [dimensions X, Y, Z]"
   - Tabular or structured comparison: "On sparse data, Method A outperforms B (X% vs Y%) 
     due to mechanism Z, but this reverses on dense data where B's assumption W holds"
   - Multi-dimensional evaluation: Compare approaches on accuracy, efficiency, interpretability, 
     scalability, assumptions, etc.
   
4. **Ground in Theory/Mechanisms:**
   - Replace descriptive statements with explanatory ones:
     * WEAK: "Method X performs well on task Y"
     * STRONG: "Method X's use of mechanism A enables handling of task Y's characteristic B, 
       as evidenced by C% improvement on dataset D"
   - Explain empirical observations through underlying mechanisms or constraints
   
5. **Increase Specificity:**
   - Add quantitative evidence: "achieves 87.3% accuracy", "reduces training time from X to Y hours"
   - Specify conditions: "on datasets with density > X", "when feature Y is available"
   - Compare numerically: "[A] reaches X% while [B] achieves Y% but with Z parameters vs W parameters"

**REVISION FRAMEWORK - Apply to Each Major Approach/Finding:**

[TOPIC/APPROACH NAME] (typically 1-2 paragraphs per major approach)

**Structure:**
1. **Context & Innovation**: What prior work motivated this? What's the key innovation? 
   "Building on [Prior Work]'s observation that X, [This Work] introduces [innovation Y] to address 
   [limitation Z]"

2. **Evidence**: What empirical support exists? Under what conditions?
   "[Paper A] demonstrates effectiveness on [dataset/scenario X], achieving [metric Y] of [value Z]"

3. **Mechanism**: How does it work? (Brief technical explanation of key idea)
   "The core mechanism involves [brief description], which enables [capability X]"

4. **Comparison**: How does it differ from alternatives?
   "Unlike [Alternative A], which relies on [mechanism X], this approach uses [mechanism Y], 
   trading [benefit of A] for [benefit of this work]"

5. **Critical Evaluation**: What are theoretical and practical limitations? WHY?
   "Cannot handle [scenario X] because [theoretical constraint Y: e.g., assumption Z must hold / 
   complexity grows as O(N²) / requires labeled data]"
   "While effective on [condition A], [Paper B] shows degradation under [condition C] due to [reason D]"

6. **Trade-offs**: What are the costs/benefits?
   "Achieves [improvement X] but requires [cost Y: e.g., more data / computational resources / 
   restrictive assumptions]"


**Requirements for Improved Version:**
1. LaTeX format with proper \\cite{{citation_key}} citations throughout
2. At least 1000 words total
3. Sub-sections: at least 500 words each with clear focus
4. Main section overview: 300+ words providing roadmap and key themes
5. **Minimum per subsection:**
   - 3 explicit comparative/contrastive statements
   - 2 limitation explanations with underlying reasons (WHY)
   - 1 pattern identification across multiple papers
   - 1 trade-off analysis with conditions/quantification
   - 1 assumption questioning or methodological critique
6. Explicitly address evaluation feedback and example revisions
7. Incorporate relevant information from additional papers
8. Reference and build on previous sections' insights
Write the improved section content only (no meta-commentary):
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
