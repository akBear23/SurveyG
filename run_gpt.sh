#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 \"research topic query\" \"your research keywords\" number_of_papers"
    exit 1
fi

ORIGINAL_QUERY="$1"
KEYWORDS="$2"
NUM_PAPERS="$3"

# Convert query to directory-friendly format (replace spaces with underscores)
DIR_QUERY=$(echo "$ORIGINAL_QUERY" | tr ' ' '_' | tr -d ':')
BASE_DIR="/media/aiserver/New Volume/HDD_linux/bear/SurveyG"
ABLATION_STUDY='_gpt'
echo "Starting survey generation for: '$ORIGINAL_QUERY'"
echo "Using directory name: $DIR_QUERY"
echo "Running ablation study: $ABLATION_STUDY"

# # # Step 2: Crawl papers
# echo "Step 2: Running crawler..."
# python "$BASE_DIR/scripts/survey_crawler.py" "$ORIGINAL_QUERY" "$KEYWORDS" "$NUM_PAPERS"

# # Step 6: Create survey graph
# echo "Step 6: Creating survey graph..."
# python "$BASE_DIR/scripts/create_survey_graph.py" "$ORIGINAL_QUERY"

# # # # Step 3: Fetch cited papers
# echo "Step 3: Fetching cited papers..."
# python "$BASE_DIR/scripts/fetch_cited_by_batch.py" "$ORIGINAL_QUERY"

# # Step 4: Download PDFs & prepare metadata
# echo "Step 4: Downloading PDFs..."
# python "$BASE_DIR/scripts/pdf_downloader.py" "$ORIGINAL_QUERY"

# # Step 5: Generate summaries and metadata
# echo "Step 5: Generating summaries..."
# python "$BASE_DIR/writing/summarize.py" "$ORIGINAL_QUERY"

# # Step 7: Traverse graph for summaries
# echo "Step 7: Traversing graph..."
# python "$BASE_DIR/scripts/traversal_gpt.py" "$ORIGINAL_QUERY" "$ABLATION_STUDY"

# Step 8: Write survey paper
echo "Step 8: Writing survey paper..."
(
    python writing/writing_survey_gpt.py "$ORIGINAL_QUERY" "$ABLATION_STUDY"
)

# Step 9: Compile LaTeX document
echo "Step 9: Compiling LaTeX..."
PAPER_DIR="$BASE_DIR/paper_data/$DIR_QUERY/literature_review_output"
mkdir -p "$PAPER_DIR"  # Ensure directory exists
(
    cd "$PAPER_DIR" && \
    pdflatex literature_review.tex
)

echo "Survey generation complete! Output located at: $PAPER_DIR/literature_review.pdf"