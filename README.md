# Workflow
- Step 1: Crawl papers related to the topic with semantic scholar search api (with constraints to speed up the process)
- Step 2: Crawl papers that cited crawled papers from step 1, merge all papers together into a pool
- Step 3: Create survey graph from pool of papers
- Step 4: Travel in graph to get taxonomies and development directions, download papers PDF along the way, create outline with proofs
- Step 5: Write paper from outline and proofs

# 1. Setup environment
```bash
pip install -r requirements.txt
```
# 2. Run survey_crawler.py to crawl papers from semantic scholar
```bash
python scripts/survey_crawler.py "your research topic query" number_of_paper_to_crawl
```
# 3. Fetch cited papers
```bash
python scripts/fetch_cited_by_batch.py "your research topic query"
```
# 4. Summarize all papers, get metadata 
```bash
python writing/summarize.py "your research topic query"
```
# 5. Create survey graph
```bash
python scripts/create_survey_graph.py "your research topic query"
```
# 6. Travel in graph to create K + 3 summaries and create paper outline
- Create 3 taxonomy summaries (by layer) and K development direction (DFS from core papers in layer 1 to layer 2 and 3)
- Download paper PDF files along the way 
- Save all the paths traveled
- Merge K + 3 summaries and create outline 
```bash
python scripts/traversal.py "your research topic query"
```
# 7. Write paper
```bash
cd writing
python writing_survey.py "your research topic query"
```
# 8. Compile paper
```bash
cd paper_data/{"your research topic query"}/literature_review_output
pdflatex literature_review.tex
```