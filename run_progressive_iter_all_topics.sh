#!/bin/bash
################################################################################
# Progressive Iteration Ablation for Multiple Topics
################################################################################
# This script runs progressive iteration ablation for all three topics,
# saving intermediate versions after each iteration (iter0-iter5).
#
# Key Features:
# - Runs iterations 0-5 sequentially for each topic
# - Saves checkpoints after each iteration
# - Can resume from any checkpoint
# - Uploads results to Hugging Face
#
# Usage:
#   ./run_progressive_iter_all_topics.sh [start_iter] [max_iter]
#
# Examples:
#   ./run_progressive_iter_all_topics.sh 0 5      # Run all iterations 0-5
#   ./run_progressive_iter_all_topics.sh 3 5      # Resume from iteration 3
################################################################################

set -e  # Exit on error

# Default iteration range
START_ITER=${1:-0}
MAX_ITER=${2:-5}

# Topics to process
TOPICS=(
    # "Retrieval-Augmented_Generation_for_Large_Language_Models"
    # "A_survey_on_Visual_Transformer"
    "knowledge_graph_embedding"
)

# Hugging Face settings
HF_REPO="AnKhanh/SurveyG"
HF_FOLDER="progressive_iteration_ablation"

echo "================================================================================"
echo "PROGRESSIVE ITERATION ABLATION - ALL TOPICS"
echo "================================================================================"
echo "Iteration Range: ${START_ITER} → ${MAX_ITER}"
echo "Topics: ${#TOPICS[@]}"
echo "Hugging Face: ${HF_REPO}/${HF_FOLDER}"
echo "================================================================================"
echo ""

# Process each topic
for TOPIC in "${TOPICS[@]}"; do
    echo ""
    echo "################################################################################"
    echo "# TOPIC: ${TOPIC}"
    echo "################################################################################"
    echo ""
    
    # Setup directories
    BASE_DIR="paper_data/${TOPIC}"
    ABLATION_DIR="${BASE_DIR}_iter_ablation"
    
    echo "📁 Directories:"
    echo "   Base: ${BASE_DIR}"
    echo "   Ablation: ${ABLATION_DIR}"
    echo ""
    
    # Check if base directory exists
    if [ ! -d "${BASE_DIR}" ]; then
        echo "❌ ERROR: Base directory not found: ${BASE_DIR}"
        echo "   Skipping topic: ${TOPIC}"
        continue
    fi
    
    # Create ablation directory structure if needed
    if [ ! -d "${ABLATION_DIR}" ]; then
        echo "📂 Creating ablation directory structure..."
        mkdir -p "${ABLATION_DIR}"
        
        # Create symlinks/copy necessary data
        echo "🔗 Setting up data links..."
        
        # Copy or symlink paths
        if [ -d "${BASE_DIR}/paths" ]; then
            if [ ! -e "${ABLATION_DIR}/paths" ]; then
                ln -s "$(realpath ${BASE_DIR}/paths)" "${ABLATION_DIR}/paths"
                echo "   ✓ Linked paths directory"
            fi
        fi
        
        # Copy or symlink info
        if [ -d "${BASE_DIR}/info" ]; then
            if [ ! -e "${ABLATION_DIR}/info" ]; then
                ln -s "$(realpath ${BASE_DIR}/info)" "${ABLATION_DIR}/info"
                echo "   ✓ Linked info directory"
            fi
        fi
        
        # Copy literature_review_output for outline
        if [ -d "${BASE_DIR}/literature_review_output" ]; then
            if [ ! -e "${ABLATION_DIR}/literature_review_output" ]; then
                ln -s "$(realpath ${BASE_DIR}/literature_review_output)" "${ABLATION_DIR}/literature_review_output"
                echo "   ✓ Linked literature_review_output directory"
            fi
        fi
        
        echo "   ✓ Directory structure ready"
    fi
    
    # Run progressive iteration ablation
    echo ""
    echo "🚀 Running progressive iteration ablation..."
    echo "   Iterations: ${START_ITER} → ${MAX_ITER}"
    echo ""
    
    python writing/writing_survey_progressive_iter.py \
        "${TOPIC}" \
        "${ABLATION_DIR}" \
        ${START_ITER} \
        ${MAX_ITER}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Completed: ${TOPIC}"
        echo ""
        
        # Upload to Hugging Face
        echo "📤 Uploading to Hugging Face..."
        
        # Upload each iteration
        for iter in $(seq ${START_ITER} ${MAX_ITER}); do
            ITER_DIR="${ABLATION_DIR}/literature_review_output_iter${iter}"
            SNAPSHOT_DIR="${ITER_DIR}_snapshot_iter${iter}"
            
            if [ -d "${SNAPSHOT_DIR}" ]; then
                echo "   Uploading iteration ${iter}..."
                
                huggingface-cli upload \
                    "${HF_REPO}" \
                    "${SNAPSHOT_DIR}" \
                    "${HF_FOLDER}/${TOPIC}/iter${iter}" \
                    --repo-type dataset \
                    --commit-message "Progressive iteration ${iter} - ${TOPIC}" \
                    || echo "⚠️  Warning: Upload failed for iteration ${iter}"
            fi
        done
        
        echo "   ✓ Upload complete"
        
    else
        echo "❌ FAILED: ${TOPIC}"
        echo "   Continuing with next topic..."
    fi
    
    echo ""
    echo "################################################################################"
    echo ""
done

# Final summary
echo ""
echo "================================================================================"
echo "🎉 PROGRESSIVE ITERATION ABLATION COMPLETE"
echo "================================================================================"
echo "Processed Topics: ${#TOPICS[@]}"
echo "Iteration Range: ${START_ITER} → ${MAX_ITER}"
echo ""
echo "📊 Results Summary:"
for TOPIC in "${TOPICS[@]}"; do
    ABLATION_DIR="paper_data/${TOPIC}_iter_ablation"
    echo ""
    echo "  ${TOPIC}:"
    
    if [ -d "${ABLATION_DIR}" ]; then
        for iter in $(seq ${START_ITER} ${MAX_ITER}); do
            ITER_DIR="${ABLATION_DIR}/literature_review_output_iter${iter}"
            
            if [ -d "${ITER_DIR}" ]; then
                # Check for PDF
                if [ -f "${ITER_DIR}/literature_review.pdf" ]; then
                    SIZE=$(du -h "${ITER_DIR}/literature_review.pdf" | cut -f1)
                    echo "    ✓ Iteration ${iter}: PDF (${SIZE})"
                else
                    echo "    ✓ Iteration ${iter}: LaTeX only"
                fi
            else
                echo "    ✗ Iteration ${iter}: Not found"
            fi
        done
    else
        echo "    ✗ Ablation directory not found"
    fi
done

echo ""
echo "📁 Output Location:"
echo "   Local: paper_data/*_iter_ablation/"
echo "   Hugging Face: ${HF_REPO}/${HF_FOLDER}"
echo ""
echo "================================================================================"
echo "Done! 🎉"
echo "================================================================================"
