# Workflow
- Step 1: Crawl papers related to the topic with semantic scholar search api (with constraints to speed up the process)
- Step 2: Crawl papers that cited crawled papers from step 1, merge all papers together into a pool
- Step 3: Download, summarize, and create survey graph from pool of papers
- Step 4: Travel in graph to get taxonomies and development directions, download papers PDF along the way, create outline with proofs
- Step 5: Write paper from outline and proofs

RUN SCRIPTS FOR FULL PIPELINE:
```bash
./run.sh "your research topic query" \"your research keywords\"  number_of_paper_to_crawl
# example
./run.sh "A SURVEY ON ADVERSARIAL RECOMMENDER SYSTEMS" "adversarial attacks, recommender systems, adversarial ML" 500
```
# 1. Setup environment

## Install dependencies
```bash
pip install -r requirements.txt
```

## Configure API Keys
This project requires several API keys to function. Create a `.env` file in the project root directory:

```bash
cp .env.example .env
```

Then edit the `.env` file and add your API keys:

```bash
# OpenAI API Key (required for LLM operations)
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_actual_openai_key_here

# Gemini API Key (alternative LLM provider)
# Get your API key from: https://aistudio.google.com/app/apikey
API_KEY=your_actual_gemini_key_here

# Semantic Scholar API Key (required for paper crawling)
# Get your API key from: https://www.semanticscholar.org/product/api
SEMANTIC_SCHOLAR_API_KEY=your_actual_semantic_scholar_key_here

# Embedding Token (HuggingFace or SiliconFlow)
# Get your token from: https://huggingface.co/settings/tokens
EMBED_TOKEN=your_actual_embed_token_here
```

**Important:** Never commit your `.env` file to version control. It's already included in `.gitignore`.
# 2. Run survey_crawler.py to crawl papers from semantic scholar
```bash
cd /media/aiserver/New Volume/HDD_linux/bear/SurveyG
python scripts/survey_crawler.py "your research topic query" number_of_paper_to_crawl
```
# 3. Create survey graph
```bash
python scripts/create_survey_graph.py "your research topic query"
```

# 4. Fetch cited papers
```bash
python scripts/fetch_cited_by_batch.py "your research topic query"
```

# 5. Download papers, prepare metadata file 
```bash
python scripts/pdf_downloader.py "your research topic query"
```
# 6. Summarize all papers, get summary, paper type for metadata 
```bash
python writing/summarize.py "your research topic query"
```

# 7. Travel in graph to create K + 3 summaries and create paper outline
- Create 3 taxonomy summaries (by layer) and K development direction (DFS from core papers in layer 1 to layer 2 and 3)
- Download paper PDF files along the way 
- Save all the paths traveled
- Merge K + 3 summaries and create outline 
```bash
python scripts/traversal.py "your research topic query"
```
# 8. Write paper
```bash
cd writing
python writing_survey.py "your research topic query"
cd ..
```
# 9. Compile paper
```bash
cd paper_data/{"your research topic query"}/literature_review_output
pdflatex literature_review.tex
```
