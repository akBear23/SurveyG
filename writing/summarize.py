import google.generativeai as genai
import PyPDF2
import docx
import os
import json
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from datetime import datetime
import hashlib
import glob
from pathlib import Path
import time
import re
import sys
import os                                                                                                                                                                                                          
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

#region PaperSummarizerRAG Class Definition
class PaperSummarizerRAG:
    #region Constructor and Initialization
    def __init__(self, query: str, api_key: str, rag_db_path: str = "./rag_database"):
        """
        Khởi tạo Paper Summarizer với RAG System
        
        Args:
            api_key (str): Gemini API key
            rag_db_path (str): Đường dẫn đến database RAG
        """
        self.query = query
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Khởi tạo embedding model
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Khởi tạo ChromaDB cho RAG
        self.client = chromadb.PersistentClient(path=rag_db_path)
        self.collection = self.client.get_or_create_collection(
            name="paper_summaries",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Thống kê xử lý
        self.processing_stats = {
            'total_files': 0,
            'processed_successfully': 0,
            'failed_files': [],
            'skipped_files': [],
            'start_time': None,
            'end_time': None
        }
        self.paper_types = {
            'survey': 'Survey/Review Paper',
            'technical': 'Technical/Research Paper', 
            'theoretical': 'Theoretical Paper',
            'empirical': 'Empirical Study',
            'case_study': 'Case Study',
            'position': 'Position Paper',
            'short': 'Short Paper/Workshop Paper'
        }
        self.metadata_file_path = f"paper_data/{query.replace(' ', '_')}/info/metadata.json"
        self.metadata_cache = None
        self._load_metadata()
    #endregion
    
    #region Metadata Management
    def _load_metadata(self):
        """Load metadata from JSON file"""
        try:
            if os.path.exists(self.metadata_file_path):
                with open(self.metadata_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Kiểm tra cấu trúc JSON có "metadata" key không
                    if "metadata" in data:
                        self.metadata_cache = data["metadata"]
                    else:
                        self.metadata_cache = data
                print(f"✅ Loaded metadata for {len(self.metadata_cache)} papers")
            else:
                print(f"⚠️  Metadata file not found: {self.metadata_file_path}")
                self.metadata_cache = {}
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            self.metadata_cache = {}
        
    def _find_metadata_by_file_path(self, file_path: str) -> Optional[Dict]:
        """
        Tìm metadata theo file path
        
        Args:
            file_path (str): Đường dẫn file
            
        Returns:
            Dict: Metadata nếu tìm thấy, None nếu không
        """
        if not self.metadata_cache:
            return None
        
        # Chuẩn hóa file path
        normalized_path = os.path.normpath(file_path)
        file_name = os.path.basename(normalized_path)
        
        # Tìm theo đường dẫn chính xác
        if normalized_path in self.metadata_cache:
            return self.metadata_cache[normalized_path]
        
        # Tìm theo tên file
        if file_name in self.metadata_cache:
            return self.metadata_cache[file_name]
        
        # Tìm theo pattern matching
        for key, value in self.metadata_cache.items():
            if file_name in key or key in normalized_path:
                return value
        
        return None
    
    def create_citation_key(self, metadata: Dict = None, file_path: str = None) -> str:
        """
        Create a citation key for LaTeX from metadata
        
        Args:
            metadata (Dict): Metadata dict (nếu không có sẽ tìm theo file_path)
            index (int): Index number for fallback
            file_path (str): File path để tìm metadata
            
        Returns:
            str: Citation key
        """
        try:
            # Nếu không có metadata, tìm theo file_path
            if not metadata and file_path:
                metadata = self._find_metadata_by_file_path(file_path)
            # print
        
            
            # Lấy thông tin authors và year
            authors_data = metadata.get('authors', [])
            year_data = metadata.get('published_date')
            # print(published_date)
            
            # Xử lý authors (có thể là list hoặc string)
            if isinstance(authors_data, list) and authors_data:
                first_author = authors_data[0]
            elif isinstance(authors_data, str) and authors_data != "Not available":
                first_author = authors_data.split(',')[0].strip()
            else:
                first_author = ""
            
            if first_author:
                # Xử lý tên tác giả (lấy họ cuối)
                name_parts = first_author.strip().split()
                if name_parts:
                    # Lấy phần cuối (họ) và loại bỏ ký tự đặc biệt
                    last_name = name_parts[-1]
                    last_name = re.sub(r'[^A-Za-z]', '', last_name)
                    
                    if len(last_name) > 0:
                        # Tạo citation key
                        year_str = re.sub(r'[^0-9]', '', str(year_data))
                        citation_key = f"{last_name.lower()}{year_str}"
                    
            
            return citation_key
        except Exception as e:
            print(f"❌ Error creating citation key: {e}")
            
    #endregion
    
    #region File Reading Methods    
    def read_pdf(self, file_path: str) -> str:
        """Đọc nội dung từ file PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Lỗi khi đọc PDF {file_path}: {e}")
            return ""
        return text
    
    def read_docx(self, file_path: str) -> str:
        """Đọc nội dung từ file Word"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Lỗi khi đọc Word {file_path}: {e}")
            return ""
    
    def read_txt(self, file_path: str) -> str:
        """Đọc nội dung từ file text"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Lỗi khi đọc file text {file_path}: {e}")
            return ""
    
    def read_paper(self, file_path: str) -> str:
        """Đọc paper từ file (hỗ trợ PDF, DOCX, TXT)"""
        if not os.path.exists(file_path):
            print(f"File không tồn tại: {file_path}")
            return ""
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.read_pdf(file_path)
        elif file_extension == '.docx':
            return self.read_docx(file_path)
        # elif file_extension == '.txt':
        #     return self.read_txt(file_path)
        else:
            # print(f"Định dạng file không được hỗ trợ: {file_extension}")
            return ""
    #endregion
    
    #region Text Processing Methods
    def chunk_text(self, text: str, max_tokens: int = 800000) -> list:
        """Chia văn bản thành các phần nhỏ"""
        max_chars = max_tokens * 4
        chunks = []
        
        if len(text) <= max_chars:
            return [text]
        
        while len(text) > max_chars:
            split_pos = text.rfind('.', 0, max_chars)
            if split_pos == -1:
                split_pos = text.rfind(' ', 0, max_chars)
            if split_pos == -1:
                split_pos = max_chars
            
            chunks.append(text[:split_pos + 1])
            text = text[split_pos + 1:].strip()
        
        if text:
            chunks.append(text)
            
        return chunks
    
    def generate_intriguing_abstract(self, summary: str) -> str:
        """
        Tạo abstract hấp dẫn từ summary sử dụng LLM
        
        Args:
            summary (str): Bản tóm tắt đầy đủ
            
        Returns:
            str: Abstract hấp dẫn
        """
        prompt = f"""
        Based on the following research paper summary, create an intriguing and engaging abstract that:
        1. Captures the reader's attention with compelling opening
        2. Highlights the most fascinating aspects and novel contributions
        3. Emphasizes the significance and potential impact
        4. Uses engaging language while maintaining academic rigor
        5. Keep it between 150-200 words
        6. Make it searchable by including key technical terms and concepts
        
        The abstract should make researchers want to read the full paper.
        
        Summary to transform:
        {summary}
        
        Generate only the intriguing abstract:
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=102400,
                    temperature=0.3,
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Lỗi khi tạo abstract: {e}")
            return ""
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Trích xuất từ khóa quan trọng từ text
        
        Args:
            text (str): Văn bản cần trích xuất từ khóa
            
        Returns:
            List[str]: Danh sách từ khóa
        """
        prompt = f"""
        Extract the most important keywords and key phrases from this research text.
        Focus on:
        1. Technical terms and concepts
        2. Research methodologies
        3. Application domains
        4. Novel contributions
        5. Important findings
        
        Return only a comma-separated list of keywords/phrases (10-15 items max).
        
        Text:
        {text}
        
        Keywords:
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=102400,
                    temperature=0.1,
                )
            )
            keywords = [kw.strip() for kw in response.text.strip().split(',')]
            return keywords[:15]  # Giới hạn 15 keywords
        except Exception as e:
            print(f"Lỗi khi trích xuất keywords: {e}")
            return []
    #endregion
    
    #region Paper Type Detection Methods
    def enhanced_detect_paper_type(self, paper_text: str, paper_metadata: dict) -> str:
        """
        Enhanced detection with better text extraction
        """
        # Try to extract abstract specifically
        abstract = self.extract_abstract(paper_text)
        introduction = self.extract_introduction(paper_text)
        
        # Use structured content for better detection
        structured_content = f"""
        ABSTRACT:
        {abstract}
        
        # INTRODUCTION (first part):
        # {introduction[:1500]}
        """
        
        detection_prompt = f"""
        Based on the abstract and introduction below, classify this paper into ONE type and identify whether the paper proposes a new direction or not:
        
        **PAPER TYPE CLASSIFICATION CRITERIA:**
        
        1. **survey** - Reviews existing literature comprehensively
           - Abstract mentions: "survey", "review", "comprehensive analysis", "state-of-the-art"
           - Introduction discusses: literature organization, classification schemes
        
        2. **technical** - Presents new methods, algorithms, or systems
           - Abstract mentions: "propose", "develop", "present", "algorithm", "method"
           - Introduction discusses: technical problem, proposed solution
        
        3. **theoretical** - Mathematical analysis, proofs, formal models
           - Abstract mentions: "prove", "theorem", "analysis", "mathematical", "formal"
           - Introduction discusses: theoretical framework, mathematical foundation
        
        4. **empirical** - Data-driven studies with statistical analysis
           - Abstract mentions: "study", "experiment", "data", "statistical", "findings"
           - Introduction discusses: research questions, methodology, participants
        
        5. **case_study** - Detailed analysis of specific applications
           - Abstract mentions: "case study", "application", "practice", "experience"
           - Introduction discusses: specific context, real-world scenario
        
        6. **position** - Argues for viewpoint or future direction
           - Abstract mentions: "argue", "position", "vision", "future", "should"
           - Introduction discusses: current problems, proposed direction
        
        7. **short** - Brief communication or work-in-progress
           - Paper length indicators, workshop venue, preliminary results

        **INDICATORS OF NEW DIRECTIONS (Look for these):**
        - Explicit statements like "we propose a new framework", "this paper introduces a novel approach"
        - Claims of "first study" or "first investigation" into something
        - Introduction of new terminology or concepts that didn't exist before
        - Significant departure from established methodologies
        - Creation of new research problems or questions
        - Claims of opening up "new avenues for research"
        - Foundational work that others are likely to build upon
        
        **INDICATORS OF INCREMENTAL WORK (Not new directions):**
        - Improvements to existing methods (faster, more accurate, etc.)
        - Applications of known methods to new but similar problems
        - Comparative studies or surveys
        - Minor extensions of previous work
        - Reproductions or validations of existing approaches

        **METADATA:**
        Title: {paper_metadata.get('title', 'Not available')}
        Venue: {paper_metadata.get('venue', 'Not available')}
        
        **CONTENT:**
        {structured_content}
        
        Respond with the paper type and 1 or 0 depending on whether the paper proposes a new direction or not, separated by a comman ','.
        Example responses: 
        "survey,0"
        "theoretical,1"
        """
        return detection_prompt
    
    def extract_abstract(self, paper_text: str) -> str:
        """Extract abstract section from paper text"""
        # Simple extraction - can be improved with better NLP
        text_lower = paper_text.lower()
        
        # Look for abstract section
        abstract_start = text_lower.find('abstract')
        if abstract_start == -1:
            return paper_text[:800]  # First 800 chars if no abstract found
        
        # Find end of abstract (usually next section)
        abstract_end = text_lower.find('introduction', abstract_start)
        if abstract_end == -1:
            abstract_end = abstract_start + 1000
        
        return paper_text[abstract_start:abstract_end]
    
    def extract_introduction(self, paper_text: str) -> str:
        """Extract introduction section from paper text"""
        text_lower = paper_text.lower()
        
        # Look for introduction
        intro_start = text_lower.find('introduction')
        if intro_start == -1:
            intro_start = text_lower.find('1. introduction')
        if intro_start == -1:
            return paper_text[1000:2500]  # Skip abstract, take next part
        
        # Take first part of introduction
        return paper_text[intro_start:intro_start + 2000]
    #endregion
    
    #region Summary Prompt Generation Methods
    def get_survey_prompt(self, citation_key: str, paper_metadata: dict, paper_text: str) -> str:
        """Survey/Review Paper Summary"""
        return f"""
        Analyze this SURVEY/REVIEW paper and provide a focused summary for literature review.
        
        **CITATION REQUIREMENTS**: Always use "\\cite{{{citation_key}}}" when referencing this paper.
        
        **SURVEY PAPER ANALYSIS**:
        
        1. **Review Scope & Objectives** (2-3 sentences)
           - What domain/field does this survey cover?
           - What are the main objectives and research questions?
        
        2. **Literature Coverage** (2-3 sentences)
           - Time period and scope of papers reviewed
           - Selection criteria and methodology for literature inclusion
        
        3. **Classification Framework** (2-3 bullet points)
           - How does the survey organize/categorize the literature?
           - Main taxonomies or classification schemes used
        
        4. **Key Findings & Insights** (3-4 bullet points)
           - Major trends and patterns identified
           - Comparative analysis of different approaches
           - Consensus and disagreements in the field
        
        5. **Research Gaps & Future Directions** (2-3 sentences)
           - What gaps does the survey identify?
           - Recommended future research directions
        
        6. **Survey Contribution** (1-2 sentences)
           - What unique value does this survey provide to the field?
           - How comprehensive and authoritative is it?
        Please extract the most important points from this research paper in bullet format above.
        
        Paper content:
        {paper_text[:22000]}
        
        Focus on the survey's synthesis and meta-analysis rather than individual papers reviewed.
        """
    
    def get_technical_prompt(self, citation_key: str, paper_metadata: dict, paper_text: str) -> str:
        """Technical/Research Paper Summary"""
        return f"""
        Analyze this TECHNICAL/RESEARCH paper and provide a focused summary for literature review.
        
        **CITATION REQUIREMENTS**: Always use "\\cite{{{citation_key}}}" when referencing this paper.
        
        **TECHNICAL PAPER ANALYSIS**:
        
        1. **Research Problem & Motivation** 
           - What specific technical problem does this paper address?
           - Why is this problem important and challenging?
        
        2. **Related Work & Positioning**
           - How does this work relate to existing approaches?
           - What are the limitations of previous solutions?
        
        3. **Technical Approach & Innovation** 
           - What is the core technical method or algorithm?
           - What makes this approach novel or different?
        
        4. **Key Technical Contributions** 
           - Novel algorithms, methods, or techniques
           - System design or architectural innovations
           - Theoretical insights or analysis
        
        5. **Experimental Validation**
           - What experiments were conducted?
           - Key performance metrics and comparison results
        
        6. **Limitations & Scope**
           - Technical limitations or assumptions
           - Scope of applicability
        
        7. **Technical Significance** 
           - How does this advance the technical state-of-the-art?
           - Potential impact on future research
        
        Please extract the most important points from this research paper in bullet format above.
        Paper content:
        {paper_text[:22000]}

        Focus on technical innovations and empirical validation.
        """
    
    def get_theoretical_prompt(self, citation_key: str, paper_metadata: dict, paper_text: str) -> str:
        """Theoretical Paper Summary"""
        return f"""
        Analyze this THEORETICAL paper and provide a focused summary for literature review.
        
        **CITATION REQUIREMENTS**: Always use "\\cite{{{citation_key}}}" when referencing this paper.
        
        **THEORETICAL PAPER ANALYSIS**:
        
        1. **Theoretical Problem & Context** (2-3 sentences)
           - What theoretical question or problem is addressed?
           - What is the theoretical context and motivation?
        
        2. **Mathematical Framework** (2-3 sentences)
           - What mathematical or formal framework is used?
           - Key assumptions and theoretical foundations
        
        3. **Main Theoretical Results** (3-4 bullet points)
           - Key theorems, propositions, or formal results
           - Theoretical insights and implications
           - Complexity analysis or bounds (if applicable)
        
        4. **Proof Techniques & Methods** (2-3 sentences)
           - What proof methods or analytical techniques are used?
           - Any novel theoretical approaches or tools
        
        5. **Theoretical Implications** (2-3 sentences)
           - What do the results mean for the field?
           - How do they extend or challenge existing theory?
        
        6. **Limitations & Assumptions** (1-2 sentences)
           - Key assumptions or restrictions
           - Scope of theoretical applicability
        
        7. **Theoretical Significance** (1-2 sentences)
           - How does this contribute to theoretical understanding?
           - Foundation for future theoretical or practical work
        
        Please extract the most important points from this research paper in bullet format above.
        
        Paper content:
        {paper_text[:22000]}
        
        Focus on formal results, proofs, and theoretical insights.
        """
    
    def get_empirical_prompt(self, citation_key: str, paper_metadata: dict, paper_text: str) -> str:
        """Empirical Study Summary"""
        return f"""
        Analyze this EMPIRICAL STUDY and provide a focused summary for literature review.
        
        **CITATION REQUIREMENTS**: Always use "\\cite{{{citation_key}}}" when referencing this paper.
        
        **EMPIRICAL STUDY ANALYSIS**:
        
        1. **Research Questions & Hypotheses** (2-3 sentences)
           - What empirical questions are investigated?
           - What hypotheses are tested?
        
        2. **Study Design & Methodology** (2-3 sentences)
           - What is the experimental or observational design?
           - Data collection and analysis methods
        
        3. **Data & Participants** (2-3 sentences)
           - What data sources or participants were used?
           - Sample size, demographics, or dataset characteristics
        
        4. **Key Empirical Findings** (3-4 bullet points)
           - Main statistical results and effect sizes
           - Significant patterns or relationships discovered
           - Hypothesis testing outcomes
        
        5. **Statistical Analysis** (2-3 sentences)
           - What statistical methods were applied?
           - Significance levels and confidence intervals
        
        6. **Validity & Limitations** (1-2 sentences)
           - Internal and external validity concerns
           - Study limitations and potential biases
        
        7. **Empirical Contribution** (1-2 sentences)
           - What new empirical knowledge is contributed?
           - Implications for theory or practice
        Please extract the most important points from this research paper in bullet format above.
        
        Paper content:
        {paper_text[:22000]}
        
        Focus on empirical methodology, statistical findings, and data-driven insights.
        """
    
    def get_case_study_prompt(self, citation_key: str, paper_metadata: dict, paper_text: str) -> str:
        
        """Case Study Summary"""
        return f"""
        Analyze this CASE STUDY and provide a focused summary for literature review.
        
        **CITATION REQUIREMENTS**: Always use "\\cite{{{citation_key}}}" when referencing this paper.
        
        **CASE STUDY ANALYSIS**:
        
        1. **Case Context & Setting** (2-3 sentences)
           - What is the specific case or context studied?
           - Why was this case selected for study?
        
        2. **Research Questions & Objectives** (2-3 sentences)
           - What questions does the case study address?
           - What insights is it trying to gain?
        
        3. **Case Study Methodology** (2-3 sentences)
           - How was the case study conducted?
           - Data collection and analysis methods
        
        4. **Key Findings & Observations** (3-4 bullet points)
           - Main discoveries or insights from the case
           - Patterns, challenges, or success factors identified
           - Specific outcomes or results
        
        5. **Lessons Learned** (2-3 sentences)
           - What practical lessons emerge from this case?
           - Best practices or recommendations
        
        6. **Generalizability & Limitations** (1-2 sentences)
           - How generalizable are the findings?
           - Context-specific limitations
        
        7. **Case Study Contribution** (1-2 sentences)
           - What unique insights does this case provide?
           - Value for practitioners or researchers

        Please extract the most important points from this research paper in bullet format above.
        
        Paper content:
        {paper_text[:22000]}
        
        Focus on case-specific insights and their broader implications.
        """
    
    def get_position_prompt(self, citation_key: str, paper_metadata: dict, paper_text: str) -> str:
        """Position Paper Summary"""
        return f"""
        Analyze this POSITION PAPER and provide a focused summary for literature review.
        
        **CITATION REQUIREMENTS**: Always use "\\cite{{{citation_key}}}" when referencing this paper.
        
        **POSITION PAPER ANALYSIS**:
        
        1. **Position Statement & Thesis** (2-3 sentences)
           - What is the main position or argument presented?
           - What viewpoint is the paper advocating?
        
        2. **Current State Critique** (2-3 sentences)
           - What problems or limitations in current approaches are identified?
           - What is being challenged or questioned?
        
        3. **Supporting Arguments** (3-4 bullet points)
           - Key arguments and evidence supporting the position
           - Logical reasoning and justifications
           - Examples or case studies used
        
        4. **Proposed Vision/Direction** (2-3 sentences)
           - What alternative approach or future direction is proposed?
           - What should the field focus on instead?
        
        5. **Implications & Impact** (2-3 sentences)
           - What would adopting this position mean for the field?
           - Potential consequences or changes needed
        
        6. **Limitations & Counterarguments** (1-2 sentences)
           - What are the weaknesses in the position?
           - Potential counterarguments or challenges
        
        7. **Position Significance** (1-2 sentences)
           - Why is this position important for the field?
           - Potential influence on future research directions
        Please extract the most important points from this research paper in bullet format above.
        
        Paper content:
        {paper_text[:22000]}
        
        Focus on the argument structure and its implications for the field.
        """
    
    def get_short_prompt(self, citation_key: str, paper_metadata: dict, paper_text: str) -> str:
        """Short Paper/Workshop Paper Summary"""
        return f"""
        Analyze this SHORT/WORKSHOP PAPER and provide a focused summary for literature review.
        
        **CITATION REQUIREMENTS**: Always use "\\cite{{{citation_key}}}" when referencing this paper.
        
        **SHORT PAPER ANALYSIS**:
        
        1. **Core Contribution** (2-3 sentences)
           - What is the main contribution or finding?
           - What specific aspect is being addressed?
        
        2. **Approach & Method** (2-3 sentences)
           - What approach or method is used?
           - Key technical or methodological details
        
        3. **Key Results** (2-3 bullet points)
           - Main findings or outcomes
           - Performance metrics or validation results
        
        4. **Novelty & Significance** (1-2 sentences)
           - What makes this work novel or interesting?
           - Why is it significant despite being brief?
        
        5. **Limitations & Future Work** (1-2 sentences)
           - What are the main limitations?
           - What future work is suggested?
        
        6. **Research Impact** (1-2 sentences)
           - How does this contribute to the broader research area?
           - Potential for future development

        Please extract the most important points from this research paper in bullet format above.
        
        Paper content:
        {paper_text[:22000]}
        
        Focus on the specific contribution and its potential impact.
        """
    
    def get_prompt_by_type(self, paper_type: str, citation_key: str, paper_metadata: dict, paper_text: str) -> str:
        """Get appropriate prompt based on paper type"""
        prompt_map = {
            'survey': self.get_survey_prompt,
            'technical': self.get_technical_prompt,
            'theoretical': self.get_theoretical_prompt,
            'empirical': self.get_empirical_prompt,
            'case_study': self.get_case_study_prompt,
            'position': self.get_position_prompt,
            'short': self.get_short_prompt
        }
        
        prompt_func = prompt_map.get(paper_type, self.get_technical_prompt)  # Default to technical
        return prompt_func(citation_key, paper_metadata, paper_text)
    #endregion
    
    #region Paper Analysis Methods
    def analyze_paper_with_type_detection(self, citation_key: str, paper_metadata: dict, paper_text: str):
        """Complete analysis with automatic type detection"""
        # First detect paper type
        detection_prompt = self.enhanced_detect_paper_type(paper_text, paper_metadata)
        
        # Then get appropriate summary prompt
        # (You would call your AI model here to get the paper type)
        response = self.model.generate_content(
                detection_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=102400,
                    temperature=0.1,
                )
            )
        
        paper_type, is_new_direction = response.text.replace("'", "").replace('"', '').replace(" ", '').split(',')[0], response.text.replace("'", "").replace('"', '').replace(" ", '').split(',')[-1]
        # paper_type = response.candidates[0].content.parts.text.strip().lower()
        print(f"Detected paper type: {paper_type}")
        # For now, return both detection and summary prompts

        prompt_summarize = self.get_prompt_by_type(paper_type, citation_key, paper_metadata, paper_text)
        response = self.model.generate_content(
                prompt_summarize,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=102400,
                    temperature=0.1,
                )
            )
            
        return response.text, paper_type, is_new_direction
    #endregion
    
    #region Document Management Methods    
    def generate_document_id(self, file_path: str) -> str:
        """Tạo ID duy nhất cho document"""
        # Sử dụng hash của file path để tạo ID nhất quán
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def is_paper_already_processed(self, file_path: str):
        """Kiểm tra xem paper đã được xử lý chưa"""
        doc_id = self.generate_document_id(file_path)
        try:
            results = self.collection.get(ids=[doc_id], include=["success",
            "doc_id",
            "summary",
            "intriguing_abstract",
            "keywords",
            "file_path",
            "citation_key",
            "metadata",
            "file_name"])
            return len(results['ids']),  results
        except:
            return 0, None
    
    #region RAG System Methods
    def save_to_rag(self, file_path: str, summary: str, intriguing_abstract: str, keywords: List[str]) -> str:
        """
        Lưu summary và abstract vào RAG system
        
        Args:
            file_path (str): Đường dẫn file gốc
            summary (str): Bản tóm tắt đầy đủ
            intriguing_abstract (str): Abstract hấp dẫn
            keywords (List[str]): Danh sách từ khóa
            
        Returns:
            str: ID của document trong RAG
        """
        # Tạo ID duy nhất
        doc_id = self.generate_document_id(file_path)
        
        # Tạo embedding cho abstract (dùng để search)
        abstract_embedding = self.embedding_model.encode(intriguing_abstract).tolist()
        
        # Metadata bao gồm cả summary đầy đủ
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "created_at": datetime.now().isoformat(),
            "keywords": ", ".join(keywords) if isinstance(keywords, list) else str(keywords), 
            "summary_length": len(summary),
            "abstract_length": len(intriguing_abstract),
            "full_summary": summary  # Lưu summary vào metadata
        }
        
        # Lưu vào ChromaDB với cả abstract và summary
        self.collection.add(
            documents=[intriguing_abstract],  # Dùng abstract để search
            metadatas=[metadata],
            embeddings=[abstract_embedding],
            ids=[doc_id]
        )
        
        # Vẫn lưu backup vào file để dễ đọc
        os.makedirs(f"paper_data/{self.query.replace(' ', '_')}/full_summary", exist_ok=True)
        summary_file = f"paper_data/{self.query.replace(' ', '_')}/full_summary/{doc_id}_full_summary.txt"
        os.makedirs(f"paper_data/{self.query.replace(' ', '_')}", exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"File: {file_path}\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Keywords: {', '.join(keywords)}\n")
            f.write("=" * 50 + "\n")
            f.write("INTRIGUING ABSTRACT:\n")
            f.write("=" * 50 + "\n")
            f.write(intriguing_abstract + "\n\n")
            f.write("=" * 50 + "\n")
            f.write("FULL SUMMARY:\n")
            f.write("=" * 50 + "\n")
            f.write(summary)
        
        # Save all keywords to a single JSON file
        all_keywords_json = f"paper_data/{self.query.replace(' ', '_')}/keywords/all_paper_keywords.json"
        # Load existing data if present
        if os.path.exists(all_keywords_json):
            try:
                with open(all_keywords_json, 'r', encoding='utf-8') as jf:
                    all_keywords = json.load(jf)
            except Exception:
                all_keywords = {}
        else:
            all_keywords = {}
        # Use file_path as key for uniqueness
        all_keywords[os.path.basename(file_path)] = keywords
        with open(all_keywords_json, 'w', encoding='utf-8') as jf:
            json.dump(all_keywords, jf, ensure_ascii=False, indent=2)
        
        return doc_id
    
    def search_similar_papers(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm papers tương tự trong RAG system
        
        Args:
            query (str): Câu truy vấn
            n_results (int): Số kết quả trả về
            
        Returns:
            List[Dict]: Danh sách kết quả tìm kiếm
        """
        try:
            # Tạo embedding cho query
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Tìm kiếm trong ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Định dạng kết quả
            formatted_results = []
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                result = {
                    'abstract': results['documents'][0][i],
                    'full_summary': metadata.get('full_summary', ''),  # Lấy summary từ metadata
                    'metadata': metadata,
                    'similarity_score': 1 - results['distances'][0][i],  # Chuyển distance thành similarity
                    'doc_id': results['ids'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {e}")
            return []
    
    def get_full_summary(self, doc_id: str) -> str:
        """
        Lấy bản tóm tắt đầy đủ từ doc_id (từ database)
        
        Args:
            doc_id (str): ID của document
            
        Returns:
            str: Bản tóm tắt đầy đủ
        """
        try:
            # Lấy từ ChromaDB trước
            results = self.collection.get(
                ids=[doc_id],
                include=['metadatas']
            )
            
            if results['metadatas'] and len(results['metadatas']) > 0:
                metadata = results['metadatas'][0]
                if 'full_summary' in metadata:
                    return metadata['full_summary']
            
            # Fallback: đọc từ file nếu không có trong database
            summary_file = f"paper_data/{self.query.replace(' ', '_')}/summaries/{doc_id}_full_summary.txt"
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return "Không tìm thấy bản tóm tắt đầy đủ."
                
        except Exception as e:
            print(f"Lỗi khi đọc summary: {e}")
            return ""
    
    def list_all_papers(self) -> List[Dict[str, Any]]:
        """
        Liệt kê tất cả papers trong database
        
        Returns:
            List[Dict]: Danh sách thông tin papers
        """
        try:
            results = self.collection.get(include=['metadatas', 'documents'])
            
            papers = []
            for i in range(len(results['ids'])):
                paper_info = {
                    'doc_id': results['ids'][i],
                    'file_name': results['metadatas'][i].get('file_name', ''),
                    'file_path': results['metadatas'][i].get('file_path', ''),
                    'created_at': results['metadatas'][i].get('created_at', ''),
                    'keywords': results['metadatas'][i].get('keywords', []),
                    'abstract': results['documents'][i],
                    'summary_length': results['metadatas'][i].get('summary_length', 0)
                }
                papers.append(paper_info)
            
            return papers
            
        except Exception as e:
            print(f"Lỗi khi liệt kê papers: {e}")
            return []
    
    def delete_paper(self, doc_id: str) -> bool:
        """
        Xóa paper khỏi database
        
        Args:
            doc_id (str): ID của document cần xóa
            
        Returns:
            bool: True nếu xóa thành công
        """
        try:
            # Xóa từ ChromaDB
            self.collection.delete(ids=[doc_id])
            
            # Xóa file backup nếu có
            summary_file = f"paper_data/{self.query.replace(' ', '_')}/summaries/{doc_id}_full_summary.txt"
            if os.path.exists(summary_file):
                os.remove(summary_file)
            
            print(f"Đã xóa paper với ID: {doc_id}")
            return True
            
        except Exception as e:
            print(f"Lỗi khi xóa paper: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê database
        
        Returns:
            Dict: Thông tin thống kê
        """
        try:
            all_papers = self.list_all_papers()
            
            if not all_papers:
                return {
                    'total_papers': 0,
                    'total_summaries': 0,
                    'average_summary_length': 0,
                    'oldest_paper': None,
                    'newest_paper': None,
                    'all_keywords': []
                }
            
            stats = {
                'total_papers': len(all_papers),
                'total_summaries': sum(1 for p in all_papers if p['summary_length'] > 0),
                'average_summary_length': sum(p['summary_length'] for p in all_papers) / len(all_papers),
                'oldest_paper': min(all_papers, key=lambda x: x['created_at'])['created_at'],
                'newest_paper': max(all_papers, key=lambda x: x['created_at'])['created_at']
            }
            
            # Tổng hợp tất cả keywords
            all_keywords = []
            for paper in all_papers:
                keywords = paper['keywords']
                if isinstance(keywords, str):
                    all_keywords.extend([kw.strip() for kw in keywords.split(',')])
                elif isinstance(keywords, list):
                    all_keywords.extend(keywords)
            
            # Đếm frequency của keywords
            keyword_freq = {}
            for keyword in all_keywords:
                if keyword.strip():
                    keyword_freq[keyword.strip()] = keyword_freq.get(keyword.strip(), 0) + 1
            
            stats['all_keywords'] = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            
            return stats
            
        except Exception as e:
            print(f"Lỗi khi lấy thống kê: {e}")
            return {}
    #endregion
    
    #region Paper Processing Methods
    def get_supported_files(self, folder_path: str) -> List[str]:
        """
        Lấy danh sách file được hỗ trợ trong folder
        
        Args:
            folder_path (str): Đường dẫn thư mục
            
        Returns:
            List[str]: Danh sách file được hỗ trợ
        """
        supported_extensions = ['.pdf', '.docx', '.txt']
        supported_files = []
        
        for ext in supported_extensions:
            pattern = os.path.join(folder_path, f"*{ext}")
            files = glob.glob(pattern)
            supported_files.extend(files)
        
        return sorted(supported_files)
    
    def ensure_metadata_completeness(self, metadata: dict, file_path: str) -> dict:
        """
        Ensure all required metadata fields are present and not null/empty.
        If missing, fill from original metadata source and log a warning.
        """
        required_fields = [
            'title', 'authors', 'published_date', 'venue', 'abstract', 'file_path'
        ]
        # If metadata is None, try to load
        if metadata is None:
            metadata = self._find_metadata_by_file_path(file_path) or {}
        # Try to fill missing fields
        original_meta = self._find_metadata_by_file_path(file_path) or {}
        for field in required_fields:
            value = metadata.get(field)
            if value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, list) and not value):
                print(field)
                # Try to fill from original_meta
                fixed_value = original_meta.get(field)
                if fixed_value is not None:
                    metadata[field] = fixed_value
                    print(f"⚠️  Warning: metadata['{field}'] was missing/null for {file_path}. Filled from original metadata.")
                else:
                    # If still missing, set to default
                    metadata[field] = [] if field == 'keywords' or field == 'authors' else 'Not available'
                    print(f"⚠️  Warning: metadata['{field}'] is missing/null for {file_path} and could not be filled. Set to default.")
        return metadata
    
    def summarize_paper(self, file_path: str) -> Dict[str, Any]:
        """
        Tóm tắt paper và lưu vào RAG system
        
        Args:
            file_path (str): Đường dẫn đến file paper
            summary_type (str): Loại tóm tắt
            
        Returns:
            Dict: Thông tin về quá trình tóm tắt và RAG
        """
        paper_text = self.read_paper(file_path)
        metadata = self._find_metadata_by_file_path(file_path)
        metadata = self.ensure_metadata_completeness(metadata, file_path)
        citation_key = self.create_citation_key(metadata)

        
        if not paper_text:
            # return {"error": "Không thể đọc được nội dung paper.", "file_path": file_path}
            paper_text = metadata['abstract']
        
        # Chia văn bản nếu quá dài
        # chunks = self.chunk_text(paper_text)
        
        # if len(chunks) == 1:
        #     # Paper ngắn, tóm tắt trực tiếp
        summary, paper_type, is_new_direction = self.analyze_paper_with_type_detection(citation_key, metadata, paper_text)
        
        # else:
        #     # Paper dài, tóm tắt từng phần rồi tổng hợp
        #     chunk_summaries = []
        #     for i, chunk in enumerate(chunks):
        #         chunk_summary = self.summarize_text(chunk, "key_points")
        #         if chunk_summary:
        #             chunk_summaries.append(chunk_summary)
            
        #     # Tổng hợp các phần tóm tắt
        #     combined_summary = "\n\n".join(chunk_summaries)
        #     summary = self.summarize_text(combined_summary, summary_type)
        
        if not summary:
            return {"error": "Không thể tạo tóm tắt.", "file_path": file_path}
        
        # Tạo intriguing abstract
        intriguing_abstract = self.generate_intriguing_abstract(summary)
        
        # Trích xuất keywords
        keywords = self.extract_keywords(summary)
        
        # Lưu vào RAG system
        doc_id = self.save_to_rag(file_path, summary, intriguing_abstract, keywords)
        
        metadata['summary'] = summary
        metadata['keywords'] = keywords
        metadata['is_new_direction'] = is_new_direction
        metadata['paper_type'] = paper_type
        return {
            "success": True,
            "doc_id": doc_id,
            "summary": summary,
            "intriguing_abstract": intriguing_abstract,
            "keywords": keywords,
            "file_path": file_path,
            "citation_key": citation_key,
            "metadata": metadata,
            "file_name": os.path.basename(file_path)
        }
    
    def process_folder(self, folder_path: str, 
                      skip_existing: bool = True, delay_seconds: float = 1.0, metadata_file='') -> Dict[str, Any]:
        """
        Xử lý tất cả papers trong một folder
        
        Args:
            folder_path (str): Đường dẫn thư mục chứa papers
            summary_type (str): Loại tóm tắt
            skip_existing (bool): Bỏ qua file đã được xử lý
            delay_seconds (float): Thời gian chờ giữa các lần xử lý (để tránh rate limit)
            
        Returns:
            Dict: Thông tin về quá trình xử lý
        """
        self.processing_stats = {
            'total_files': 0,
            'processed_successfully': 0,
            'failed_files': [],
            'skipped_files': [],
            'start_time': datetime.now(),
            'end_time': None
        }
        # if "core_papers/" in folder_path:
        #     folder_path = folder_path.replace("core_papers/", "")
        # if not os.path.exists(folder_path):
        #     return {"error": f"Thư mục không tồn tại: {folder_path}"}
        
        # Lấy danh sách file được hỗ trợ
        # supported_files = self.get_supported_files(folder_path)

        if metadata_file == '':
            supported_files = self.metadata_cache.keys()
        else:
            try:
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Kiểm tra cấu trúc JSON có "metadata" key không
                        if "metadata" in data:
                            metadata = data["metadata"]
                        else:
                            metadata = data
                    print(f"✅ Loaded metadata for {len(metadata)} papers")
                else:
                    print(f"⚠️  Metadata file not found: {metadata_file}")
                    metadata = {}
            except Exception as e:
                print(f"❌ Error loading metadata: {e}")
            supported_files = metadata.keys()
        print(len(supported_files))

        self.processing_stats['total_files'] = len(supported_files)

        # if not supported_files:
        #     return {"error": "Không tìm thấy file được hỗ trợ trong thư mục"}
        
        print(f"Tìm thấy {len(supported_files)} file được hỗ trợ trong thư mục: {folder_path}")
        print("=" * 80)
        
        processed_results = []
        checkpoint_file = f"paper_data/{self.query.replace(' ', '_')}/keywords/processed_checkpoint.json"
        # Load checkpoint if exists
        already_processed_files = set()
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as cp:
                    checkpoint_data = json.load(cp)
                    # checkpoint_data is a list of dicts with 'file_path' key
                    processed_results = checkpoint_data
                    already_processed_files = set([r.get('file_path') for r in processed_results if r.get('file_path')])
                    # Check for missing/null fields and fix from metadata
                    for entry in processed_results:
                        file_path = entry.get('file_path')
                        if not file_path or file_path.endswith('.txt'):
                            continue
                        # Check for null or missing fields
                        fields_to_check = ['metadata', 'citation_key', 'file_name', 'keywords', 'summary', 'intriguing_abstract']
                        for field in fields_to_check:
                            if field not in entry or entry[field] is None:
                                print(f"⚠️  Warning: '{field}' is null/missing for {file_path}. Attempting to fix from metadata file.")
                                if field == 'metadata':
                                    entry['metadata'] = self._find_metadata_by_file_path(file_path)
                                elif field == 'citation_key':
                                    entry['citation_key'] = self.create_citation_key(entry.get('metadata'), file_path)
                                elif field == 'file_name':
                                    entry['file_name'] = os.path.basename(file_path)
                                elif field == 'keywords':
                                    meta = entry.get('metadata')
                                    entry['keywords'] = meta.get('keywords', []) if meta else []
                                elif field == 'summary':
                                    entry['summary'] = ''
                                elif field == 'intriguing_abstract':
                                    entry['intriguing_abstract'] = ''
                        # Ensure metadata completeness
                        entry['metadata'] = self.ensure_metadata_completeness(entry.get('metadata'), file_path)
                print(f"🔄 Resuming from checkpoint: {len(already_processed_files)} files already processed.")
            except Exception as e:
                print(f"⚠️  Error loading checkpoint: {e}")
        
        for i, file_path in enumerate(supported_files):
            file_name = os.path.basename(file_path)
            if file_path in already_processed_files:
                print(f"  ⏭️  Bỏ qua - File đã được lưu trong checkpoint")
                self.processing_stats['skipped_files'].append(file_path)
                continue
            print(f"\n[{i+1}/{len(supported_files)}] Đang xử lý: {file_name}")
            # Kiểm tra xem file đã được xử lý chưa
            exising_flag, summary_ifTrue = self.is_paper_already_processed(file_path)
            if skip_existing and exising_flag>0:
                print(f"  ⏭️  Bỏ qua - File đã được xử lý trước đó")
                self.processing_stats['skipped_files'].append(file_path)
                processed_results.append(summary_ifTrue)
                # Save checkpoint after skip
                with open(checkpoint_file, 'w', encoding='utf-8') as cp:
                    json.dump(processed_results, cp, ensure_ascii=False, indent=2)
                continue
            result = self.summarize_paper(file_path)
            # print(file_name)
            file_metadata = result['metadata']
            self.metadata_cache[file_name] = file_metadata
            with open(self.metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_cache, f, indent=4)
            if result.get("success"):
                print(f"  ✅ Xử lý thành công - ID: {result['doc_id']}")
                print(f"     Keywords: {', '.join(result['keywords'][:3])}...")
                self.processing_stats['processed_successfully'] += 1
                processed_results.append(result)
            else:
                # print(f"  ❌ Lỗi: {result.get('error', 'Không xác định')}")
                # self.processing_stats['failed_files'].append({
                #     'file_path': file_path,
                #     'error': result.get('error', 'Không xác định')
                # })
                continue
            # Save checkpoint after each iteration
            with open(checkpoint_file, 'w', encoding='utf-8') as cp:
                json.dump(processed_results, cp, ensure_ascii=False, indent=2)
            if i < len(supported_files) - 1:  # Không delay cho file cuối cùng
                time.sleep(delay_seconds)
        
        self.processing_stats['end_time'] = datetime.now()
        
        # Tạo báo cáo tổng kết
        # self.print_processing_summary()
        
        return processed_results
    #endregion
#endregion

#region Main Function and Usage Example    
# Ví dụ sử dụng
def main():
    if len(sys.argv) != 2:
        print("Usage: python writing/summarize.py \"your research query\"")
        print("Example: python writing/summarize.py \"federated learning privacy\"")
        return
    query = sys.argv[1]
    # Thay thế bằng Gemini API key của bạn
    load_dotenv(Path(".env"))
    API_KEY = os.getenv("API_KEY") 
    rag_db_path = f"paper_data/{query.replace(' ', '_')}/rag_database"
    os.makedirs(f"paper_data/{query.replace(' ', '_')}/summaries/", exist_ok=True)
    os.makedirs(f"paper_data/{query.replace(' ', '_')}/keywords/", exist_ok=True)
    os.makedirs(f"paper_data/{query.replace(' ', '_')}/rag_database/", exist_ok=True)
    # Khởi tạo summarizer với RAG
    summarizer = PaperSummarizerRAG(query, API_KEY, rag_db_path)
    
    # Đường dẫn đến folder chứa papers
    papers_folder = f"paper_data/{query.replace(' ', '_')}"
    
    # Xử lý tất cả papers trong folder
    result = summarizer.process_folder(
        folder_path=papers_folder,

        skip_existing=True,  # Bỏ qua file đã xử lý
        delay_seconds=0.0    # Chờ 2 giây giữa các lần xử lý'
    )
if __name__ == "__main__":
    main()
#endregion
