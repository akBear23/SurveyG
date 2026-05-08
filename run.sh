# #!/bin/bash

# # Check if both arguments are provided
# if [ $# -ne 3 ]; then
#     echo "Usage: $0 \"research topic query\" \"your research keywords\" number_of_papers"
#     exit 1
# fi

# ORIGINAL_QUERY="$1"
# KEYWORDS="$2"
# NUM_PAPERS="$3"

# # Convert query to directory-friendly format (replace spaces with underscores)
# DIR_QUERY=$(echo "$ORIGINAL_QUERY" | tr ' ' '_' | tr -d ':')
# BASE_DIR="/path/to/SurveyG"

# echo "Starting survey generation for: '$ORIGINAL_QUERY'"
# echo "Using directory name: $DIR_QUERY"

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
# python "$BASE_DIR/scripts/traversal.py" "$ORIGINAL_QUERY"

# # Step 8: Write survey paper
# echo "Step 8: Writing survey paper..."
# (
#     python writing/writing_survey.py "$ORIGINAL_QUERY"
# )

# # Step 9: Compile LaTeX document
# echo "Step 9: Compiling LaTeX..."
# PAPER_DIR="$BASE_DIR/paper_data/$DIR_QUERY/literature_review_output"
# mkdir -p "$PAPER_DIR"  # Ensure directory exists
# (
#     cd "$PAPER_DIR" && \
#     pdflatex literature_review.tex
# )

# echo "Survey generation complete! Output located at: $PAPER_DIR/literature_review.pdf"

#!/bin/bash

# Check if arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 \"research topic query\" \"your research keywords\" number_of_papers [top_k]"
    echo "Example: $0 \"federated learning\" \"privacy,security\" 100"
    echo "Example: $0 \"federated learning\" \"privacy,security\" 100 20  # Run with top_k=20"
    exit 1
fi

ORIGINAL_QUERY="$1"
KEYWORDS="$2"
NUM_PAPERS="$3"
TOP_K="${4:-}"  # Optional 4th argument, empty string if not provided

# Convert query to directory-friendly format (replace spaces with underscores)
DIR_QUERY=$(echo "$ORIGINAL_QUERY" | tr ' ' '_' | tr -d ':')
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine output directory based on top_k
if [ -n "$TOP_K" ]; then
    OUTPUT_DIR_QUERY="${DIR_QUERY}_top${TOP_K}"
    echo "Running with top_k=${TOP_K}"
else
    OUTPUT_DIR_QUERY="${DIR_QUERY}"
    echo "Running with automatic K selection"
fi

echo "Starting survey generation for: '$ORIGINAL_QUERY'"
echo "Base directory: $DIR_QUERY"
echo "Output directory: $OUTPUT_DIR_QUERY"

# Only run steps 2-5 if this is the first run (base folder doesn't exist)
if [ ! -d "$BASE_DIR/paper_data/$DIR_QUERY/info" ] || [ -z "$TOP_K" ]; then
    echo "Running full pipeline (crawl, download, summarize)..."
    
    # Step 2: Crawl papers
    echo "Step 2: Running crawler..."
    python "$BASE_DIR/scripts/survey_crawler.py" "$ORIGINAL_QUERY" "$KEYWORDS" "$NUM_PAPERS"

    # Step 3: Fetch cited papers
    echo "Step 3: Fetching cited papers..."
    python "$BASE_DIR/scripts/fetch_cited_by_batch.py" "$ORIGINAL_QUERY"

    # Step 4: Download PDFs & prepare metadata
    echo "Step 4: Downloading PDFs..."
    python "$BASE_DIR/scripts/pdf_downloader.py" "$ORIGINAL_QUERY"

    # Step 5: Generate summaries and metadata
    echo "Step 5: Generating summaries..."
    python "$BASE_DIR/writing/summarize.py" "$ORIGINAL_QUERY"
else
    echo "Reusing existing data from $DIR_QUERY (skipping crawl, download, summarize)"
fi

# Step 6: Create survey graph (with optional top_k)
echo "Step 6: Creating survey graph..."
if [ -n "$TOP_K" ]; then
    python "$BASE_DIR/scripts/create_survey_graph.py" "$ORIGINAL_QUERY" "$TOP_K"
else
    python "$BASE_DIR/scripts/create_survey_graph.py" "$ORIGINAL_QUERY"
fi

# Step 7: Traverse graph for summaries and generate survey paper outline
echo "Step 7: Traversing graph..."
if [ -n "$TOP_K" ]; then
    python "$BASE_DIR/scripts/traversal.py" "$ORIGINAL_QUERY" "" "$TOP_K"
else
    python "$BASE_DIR/scripts/traversal.py" "$ORIGINAL_QUERY"
fi

# Step 8: Write survey paper, for each section and subsection, write, evaluate and use rag to improve the survey paper
echo "Step 8: Writing survey paper..."
if [ -n "$TOP_K" ]; then
    python "$BASE_DIR/writing/writing_survey.py" "$ORIGINAL_QUERY" "" "$TOP_K"
else
    python "$BASE_DIR/writing/writing_survey.py" "$ORIGINAL_QUERY"
fi

# Step 9: Compile LaTeX document
echo "Step 9: Compiling LaTeX..."
PAPER_DIR="$BASE_DIR/paper_data/$OUTPUT_DIR_QUERY/literature_review_output"
mkdir -p "$PAPER_DIR"  # Ensure directory exists
(
    cd "$PAPER_DIR" && \
    pdflatex literature_review.tex
)

echo "Survey generation complete! Output located at: $PAPER_DIR/literature_review.pdf"

# Run Streamlit app (optional)
# streamlit run streamlit_demo.py "$ORIGINAL_QUERY"