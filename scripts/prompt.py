class PromptHelper:
    def __init__(self):
        # workflow 
        # 1 - bfs modify
        # 2 - K community (done)
        # 3 - subsecton modify
        # 4 - Evaluate 

# Remember to add information of paper like the template in PAPER_INFO
        self.BFS_PROMPT_1_touch = """
*Instruction:* You are a research analyst tracing the evolution of scientific ideas. Your task is to analyze how research in "[QUERY]" has developed through a chain of connected papers, where each work builds upon previous contributions.

TASK: Analyze the following citation path to reveal how methodologies, problems, and insights evolve across [NUMBER_OF_PAPERS] interconnected papers.

**Papers to reference (sorted chronologically):**

[PAPER_INFO]

Each paper follows this format:
[citation_key] Title (Year)
Summary: [Description of the paper's content]

*Output your analysis in the following exact structure:*

1. <think>
Examine the citation path chronologically. For each progression:
- How many distinct methodological or conceptual shifts occur?
- What specific problems does each new paper address that previous papers left unsolved or unexplored?
- What innovations or capabilities does each subsequent paper introduce?
- Are there temporal gaps or clusters that suggest external influences (e.g., new datasets, computational advances)?

*Format your analysis in this section as a clear, chronological list or table.*
</think>

2. *Evolution Analysis:*
Write a cohesive narrative (target length: ~500-700 words) organized around 1-2 major trends or transitions you identified. *(A "trend" is a sustained directional shift, such as a move to a new methodology, a change in problem focus, or a significant increase in scale/complexity.)* For each trend:

*Trend [Number]: [Descriptive name]*
- *Methodological progression*: Describe how technical approaches evolve. Reference specific papers.
- *Problem evolution*: Specifically explain what limitations or gaps each paper addresses.
- *Key innovations*: Highlight breakthrough contributions that enable new capabilities or insights.

After describing all trends, provide:

3. *Synthesis* (2-3 sentences):
What unified intellectual trajectory connects these works? What is their collective contribution to advancing "[QUERY]"?

REMEMBER:
- Always cite papers specifically by title, year, and citation info: "[citation_key] Title (Year)"
"""

        self.BFS_PROMPT_2_touch = """
*Instruction:* You are a research analyst tracing the evolution of scientific ideas through hierarchical knowledge synthesis. Your task is to analyze how research in "[QUERY]" has developed by building upon existing understanding and incorporating new papers into an evolving narrative.

CONTEXT: You have access to a previously synthesized understanding of this research area, along with new papers to integrate. Your goal is to update and refine the evolutionary analysis by incorporating these new contributions.

**Previous Synthesis (if available):**
[PREVIOUS_SYNTHESIS]

**New Papers to Integrate (sorted chronologically):**
[PAPER_INFO]

Each paper follows this format:
[citation_key] Title (Year)
Summary: [Description of the paper's content]

TASK: Integrate the new papers into the existing knowledge structure, identifying how they extend, refine, or redirect the previously understood evolutionary trajectory across [TOTAL_NUMBER_OF_PAPERS] papers.

*Output your analysis in the following exact structure:*

1. <think>
**Integration Analysis:**
- How do the new papers relate to the previously identified trends? (Do they extend, challenge, or branch from existing directions?)
- What new methodological or conceptual shifts appear with these additions?
- Do the new papers fill gaps identified in the previous synthesis, or do they open entirely new directions?
- Are there connections between new papers and earlier works not previously synthesized?
- Does the addition of new papers change the overall narrative or strengthen existing interpretations?

**Temporal Positioning:**
- Where do the new papers fit chronologically relative to the previous synthesis?
- Do they represent the latest developments, or do they fill historical gaps?

*Format your analysis in this section as a clear, structured assessment.*
</think>

2. *Updated Evolution Analysis:*
Write a cohesive narrative (target length: ~500-700 words) that integrates new findings with previous understanding. Organize around major trends or transitions:

For each trend (continuing or new):
*Trend [Number]: [Descriptive name]*
- *Methodological progression*: Describe how technical approaches evolve, explicitly noting how new papers advance or diverge from the established trajectory.
- *Problem evolution*: Explain what limitations each paper addresses, highlighting connections between new and previously analyzed works.
- *Key innovations*: Emphasize breakthrough contributions from new papers and how they relate to earlier innovations.
- *Integration points*: Explicitly state how new papers connect to or build upon previously synthesized work.

3. *Refined Synthesis* (3-4 sentences):
What unified intellectual trajectory connects all works (previous + new)? How has your understanding of the field's evolution been updated or refined? What is the collective contribution to advancing "[QUERY]" with this expanded view?

REMEMBER:
- Always cite papers specifically: "[citation_key] Title (Year)"
- Explicitly mark which insights come from newly added papers vs. previous synthesis
- Maintain coherence between previous understanding and new additions
- If no previous synthesis exists, treat this as the initial analysis
"""

        self.COMMUNITY_PROMPT = """
*Instruction:* You are a research analyst synthesizing a body of literature. Your task is to analyze the provided papers on the survey topic "[QUERY]".

TASK: Cluster the provided papers into 2-3 distinct subgroups based on their methodologies, contributions, or thematic focus. Then provide a critical analysis of each cluster and the field overall.

**Papers to reference (sorted chronologically):**

[PAPER_INFO]

Each paper follows this format:
[citation_key] Title (Year)
Summary: [Description of the paper's content]

*Output your analysis in the following exact structure:*

1.  <think>
Explain your reasoning for how you clustered papers into subgroups based on their methodologies, contributions, and thematic scope.
</think>
2.  *For each subgroup:*
    *   *Subgroup name*: [Clear descriptive name]
    *   *Papers*: [List paper with format [citation_key] Title (Year)]
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

# Remember to add layer 1 community to PAPER_COMMUNITIES
        self.OUTLINE_PROMPT = """You are creating a comprehensive literature review outline for: "[QUERY]"

# AVAILABLE RESEARCH SYNTHESIS

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

#Remember to drop <think> tags after generating subsections
        self.WRITE_INITIAL_SUBSECTION_PROMPT = """
Write a comprehensive literature review subsection titled "[SUBSECTION_TITLE]" in LaTeX format.

**SUBSECTION SPECIFIC FOCUS:** [SUBSECTION_FOCUS]
**Community summaries:** [COMMUNITY_SUMMARY] 
**Development directions:** [DEVELOPMENT_DIRECTION]

**Papers to reference (sorted chronologically):**

[PAPER_INFO]

Each paper follows this format:
[citation_key] Title (Year)
Summary: [Description of the paper's content]

First, generate your reasoning in <think></think> tags by analyzing:
- Community summaries: to understand the research direction, methodological patterns, and key limitations
- Development directions: to trace how papers build on each other and what problems each solves
- Papers to reference: to extract specific technical details for each method

In your <think> tags, you should:
- Identify methodological patterns from community summaries
- Map the paper progression chain from development directions
- Plan how to connect papers (which paper addresses which limitation)

Now consider the following guidelines carefully while writing the subsection:

**CRITICAL REQUIREMENTS:**

1. **Format:**
  - Generate text in LaTeX format with proper citations (\\cite{{citation_key}})
  - At least 400 words for this subsection
  - No numbering in subsection titles

2. **Content:**
  - Focus ONLY on the SUBSECTION SPECIFIC FOCUS assigned to this subsection
    
3. **Evidence & Citation:**
  - Use specific examples with citations

4. **Structure the Subsection:**
  - Opening: Introduce problem statement in this subsection by 1-2 sentences.
  - Body: Generating content follow these rules: 
          - Each paper is described in 2-3 sentences with specific technical contributions, using citation in Available Citations.
          - Papers are connected by showing how later work addresses limitations in earlier work based on your thought before.
          - Ensure the overall narrative follows the progression identified in your reasoning.
  - Closing: Conclusion about unresolved issues across all papers or future directions that emerge from the analyzed literature.

**RED FLAGS to avoid:**
- Sequential listing: "Paper A proposes X. Paper B proposes Y. Paper C proposes Z."
- Vague critique: "This approach has limitations" (without specifying what/why)
- Unsupported claims: "This is significant" or "This is important" (without explaining why/how)
- Missing comparisons: Discussing methods in isolation without relating them
- Pure description: Describing what a paper does without analyzing its contribution or limitations
- Ignoring contradictions: Not addressing when papers reach conflicting conclusions
- Cherry-picking: Only citing results that support a narrative, ignoring contradictory evidence
"""

        self.EVALUATE_SUBSECTION_PROMPT = """
You are a grumpy expert academic researcher carefully evaluating this literature review subsection.

**Previous subsection if any:** [PRE_SUBSECTION]
**Subsection Title**: [SUBSECTION_TITLE]
**Expected Focus**: [SUBSECTION_FOCUS]
**Overall Review Context: Outline**: [OUTLINE]
**Subsection Content**: [SUBSECTION_CONTENT]

Evaluate this subsection holistically as an expert academic reviewer would assess a literature review section.

**IMPORTANT**: 
- Return ONLY valid JSON without any markdown formatting or code blocks
- Escape all backslashes and quotes properly in JSON strings
- Do not include any special characters that might break JSON parsing
- All text fields can be detailed paragraphs with your expert analysis

**Response Format** (JSON only):
{{
  "overall_score": <1-5, where 5 is excellent>,
  "redundancy_check": "<paragraph analyzing overlap with previous subsection(s). Identify specific redundant content, repeated concepts, or duplicated citations. If no previous subsection exists or no redundancy found, state this clearly.>",
  "strengths": ["list of specific strengths with examples"],
  "weaknesses": ["list of specific weaknesses with examples"],
  "is_satisfactory": <true if overall_score >= 3.5, false otherwise>,
  "improvement_needed": ["specific actionable improvements with concrete suggestions"],
  "suggested_queries": ["suggested search queries to find additional relevant papers to address gaps or weaknesses"]
}}
"""
        
        self.CHECK_RAG_RESULT_PROMPT = """
You are an expert academic reviewer filtering retrieved papers for a literature review subsection.

**Subsection Title**: [SUBSECTION_TITLE]
**Subsection Focus**: [SUBSECTION_FOCUS]

**Current Subsection Content**:
[CURRENT_CONTENT]

**Current Weaknesses/Gaps** (from previous evaluation):
[WEAKNESSES]

**Retrieved Papers**:
[RETRIEVED_PAPERS]

Each paper follows this format:
[citation_key] Title (Year)
Summary: [paper content summary]

**Task**: Filter and return only the papers that are relevant to this subsection and can help address the weaknesses or enhance the content.

**IMPORTANT**: 
- Return ONLY valid JSON without any markdown formatting or code blocks
- Include only papers that are truly relevant to this specific subsection
- Maintain the exact same format as the input

**Response Format** (JSON only):
{{
  "filtered_papers": "<papers in the same format as [RETRIEVED_PAPERS], one per line:\\n[citation_key] Title (Year)\\nSummary: [paper content summary]\\n>",
  "reason": "<brief paragraph explaining the filtering criteria and which papers were selected>"
}}
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
  3. Ensure the content stays focused on: [SUBSECTION_FOCUS]
  4. Maintain academic writing style
  5. Use proper LaTeX citations (\\cite{{citation_key}})
  6. Remove redundant information with previous subsection. Try to write different aspects of similar papers in previous subsection.

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

# Remember to add on new paper after RAG if any to PAPER INFO + only extract model name of them
        self.CHECK_CITATION_KEY = """
You are verifying citation accuracy in a literature review subsection.

**Subsection Title**: [SUBSECTION_TITLE]

**Papers to reference (sorted chronologically):**
[PAPER_INFO]

Each paper follows this format:
[citation_key] Title (Year)
Model name: [name of model if this is a methodological paper]

**Subsection Content**: [SUBSECTION_CONTENT]

**Task**: Check if each citation_key in the subsection content correctly maps to its paper in the paper list above. Match by either the paper title or model name mentioned in context.

**IMPORTANT**: 
- Return ONLY valid JSON without any markdown formatting or code blocks
- Only report actual mismatches where citation_key clearly references wrong paper

**Response Format** (JSON only):
{{
  "citation_errors": [
    {{
      "citation_key_used": "<the citation_key used in content>",
      "context": "<brief quote showing how it's used>",
      "correct_citation_key": "<what it should be based on title/model name>",
      "reason": "<brief explanation of the mismatch>"
    }}
  ],
  "all_correct": <true if no errors found, false otherwise>
}}
"""

        self.SUBSECTION_MODIFY_CITATION_PROMPT = """
Correct the citation keys in this literature review subsection based on the verification report.

**Subsection Title**: [SUBSECTION_TITLE]

**Current Subsection Content**:
[CURRENT_CONTENT]

**Citation Errors Found**:
[CITATION_ERRORS]

The citation errors follow this format:
- citation_key_used: <incorrect key used>
- context: <where it appears>
- correct_citation_key: <what it should be>
- reason: <why it's wrong>

**Instructions**:
1. Replace each incorrect citation_key with the correct one
2. Ensure the corrected citations still fit naturally in context
3. Do NOT change any other content, wording, or structure
4. Maintain proper LaTeX citation format: \\cite{{citation_key}}
5. If multiple citations are together, keep them together: \\cite{{key1, key2}}

**IMPORTANT**:
- Return ONLY the corrected subsection content in LaTeX format
- Do NOT add explanations, comments, or markdown formatting
- Do NOT modify any text except the citation keys themselves

**Output the corrected subsection content below:**
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