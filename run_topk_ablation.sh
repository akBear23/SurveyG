#!/bin/bash

# Top-K Ablation Study Script
# This script runs the survey generation pipeline with different top_k values
# to evaluate the impact of the number of foundation papers (Layer 1)

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"research topic query\" [top_k_values...]"
    echo "Example: $0 \"federated learning\" 10 15 20 25"
    echo "If no top_k values provided, will run with: 10, 15, 20, 25"
    exit 1
fi

ORIGINAL_QUERY="$1"
shift  # Remove first argument

# Default top_k values if none provided
if [ $# -eq 0 ]; then
    TOP_K_VALUES=(6 8 10 12 14)
else
    TOP_K_VALUES=("$@")
fi

DIR_QUERY=$(echo "$ORIGINAL_QUERY" | tr ' ' '_' | tr -d ':')
BASE_DIR="/media/aiserver/New Volume/HDD_linux/bear/SurveyG"

echo "=================================="
echo "Top-K Ablation Study"
echo "=================================="
echo "Query: $ORIGINAL_QUERY"
echo "Base directory: $DIR_QUERY"
echo "Top-K values to test: ${TOP_K_VALUES[*]}"
echo "=================================="

# Check if base data exists
if [ ! -d "$BASE_DIR/paper_data/$DIR_QUERY/info" ]; then
    echo "ERROR: Base data not found at paper_data/$DIR_QUERY"
    echo "Please run the full pipeline first without top_k parameter:"
    echo "  ./run.sh \"$ORIGINAL_QUERY\" \"keywords\" num_papers"
    exit 1
fi

echo "✓ Base data found, reusing crawled papers, PDFs, and summaries"
echo ""

# Create ablation results directory
ABLATION_DIR="$BASE_DIR/paper_data/${DIR_QUERY}_ablation_topk"
mkdir -p "$ABLATION_DIR"

# Save experiment configuration
cat > "$ABLATION_DIR/experiment_config.txt" << EOF
Top-K Ablation Study
====================
Query: $ORIGINAL_QUERY
Base Directory: $DIR_QUERY
Top-K Values: ${TOP_K_VALUES[*]}
Date: $(date)

This experiment tests the impact of varying the number of foundation papers (Layer 1)
on the quality and coverage of the generated literature review.
EOF

echo "Experiment configuration saved to: $ABLATION_DIR/experiment_config.txt"
echo ""

# Run pipeline for each top_k value
for TOP_K in "${TOP_K_VALUES[@]}"; do
    echo "========================================"
    echo "Running experiment with top_k=$TOP_K"
    echo "========================================"
    
    OUTPUT_DIR="${DIR_QUERY}_top${TOP_K}"
    
    # Step 6: Create survey graph with specific top_k
    echo "Step 6: Creating survey graph (top_k=$TOP_K)..."
    python "$BASE_DIR/scripts/create_survey_graph.py" "$ORIGINAL_QUERY" "$TOP_K"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Graph creation failed for top_k=$TOP_K"
        continue
    fi

    # Step 7: Traverse graph if outline does not exist
    if [ ! -f "$BASE_DIR/paper_data/$OUTPUT_DIR/literature_review_output/survey_outline.json" ]; then
        echo "Step 7: Traversing graph (top_k=$TOP_K)..."
        python "$BASE_DIR/scripts/traversal.py" "$ORIGINAL_QUERY" "" "$TOP_K"
    fi
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Graph traversal failed for top_k=$TOP_K"
        continue
    fi
    # echo "Step 7: Traversing graph (top_k=$TOP_K)..."
    # python "$BASE_DIR/scripts/traversal.py" "$ORIGINAL_QUERY" "" "$TOP_K"
    
    # if [ $? -ne 0 ]; then
    #     echo "ERROR: Graph traversal failed for top_k=$TOP_K"
    #     continue
    # fi
    
    # Step 8: Write survey paper
    echo "Step 8: Writing survey paper (top_k=$TOP_K)..."
    python "$BASE_DIR/writing/writing_survey.py" "$ORIGINAL_QUERY" "" "$TOP_K"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Survey writing failed for top_k=$TOP_K"
        continue
    fi
    
    # Step 9: Compile LaTeX document
    echo "Step 9: Compiling LaTeX (top_k=$TOP_K)..."
    PAPER_DIR="$BASE_DIR/paper_data/$OUTPUT_DIR/literature_review_output"
    
    if [ -f "$PAPER_DIR/literature_review.tex" ]; then
        (
            cd "$PAPER_DIR" && \
            pdflatex -interaction=nonstopmode literature_review.tex > /dev/null 2>&1
        )
        
        if [ -f "$PAPER_DIR/literature_review.pdf" ]; then
            echo "✓ PDF generated successfully for top_k=$TOP_K"
            # Copy to ablation directory for easy comparison
            cp "$PAPER_DIR/literature_review.pdf" "$ABLATION_DIR/literature_review_top${TOP_K}.pdf"
        else
            echo "⚠ LaTeX compilation failed for top_k=$TOP_K"
        fi
    else
        echo "⚠ LaTeX file not found for top_k=$TOP_K"
    fi
    
    echo "✓ Completed experiment with top_k=$TOP_K"
    echo ""
done

echo "========================================"
echo "Ablation Study Complete!"
echo "========================================"
echo "Results saved to: $ABLATION_DIR"
echo ""
echo "Generated PDFs:"
ls -lh "$ABLATION_DIR"/*.pdf 2>/dev/null || echo "No PDFs found"
echo ""
echo "Individual results:"
for TOP_K in "${TOP_K_VALUES[@]}"; do
    OUTPUT_DIR="${DIR_QUERY}_top${TOP_K}"
    RESULT_DIR="$BASE_DIR/paper_data/$OUTPUT_DIR"
    if [ -d "$RESULT_DIR" ]; then
        echo "  top_k=$TOP_K: $RESULT_DIR"
    fi
done
echo ""
echo "To compare results, see the PDFs in: $ABLATION_DIR"
