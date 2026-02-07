#!/bin/bash

# Multi-Topic Top-K Ablation Study with Hugging Face Upload
# This script runs ablation studies for multiple research topics and uploads results

set -e  # Exit on error

# Configuration
BASE_DIR="/media/aiserver/New Volume/HDD_linux/bear/SurveyG"
HF_REPO="AnKhanh/SurveyG"
UPLOAD_DIR="$BASE_DIR/hf_upload_staging"

# Topics to run
TOPICS=(
    # "Retrieval-Augmented_Generation_for_Large_Language_Models"
    # "A_survey_on_Visual_Transformer"
    "knowledge_graph_embedding"
)

# Top-K values to test for each topic
TOP_K_VALUES=(14)

echo "=========================================="
echo "Multi-Topic Top-K Ablation Study"
echo "=========================================="
echo "Topics: ${#TOPICS[@]}"
for topic in "${TOPICS[@]}"; do
    echo "  - $topic"
done
echo "Top-K values: ${TOP_K_VALUES[*]}"
echo "Hugging Face repo: $HF_REPO"
echo "=========================================="
echo ""

# Create upload staging directory
mkdir -p "$UPLOAD_DIR"

# Track all generated outputs
declare -a ALL_OUTPUTS

# Run ablation for each topic
for TOPIC in "${TOPICS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing Topic: $TOPIC"
    echo "=========================================="
    
    DIR_QUERY=$(echo "$TOPIC" | tr ' ' '_' | tr -d ':')
    
    # Check if base data exists
    if [ ! -d "$BASE_DIR/paper_data/$DIR_QUERY/info" ]; then
        echo "WARNING: Base data not found for '$TOPIC'"
        echo "Skipping this topic. Please run the full pipeline first:"
        echo "  ./run.sh \"$TOPIC\" \"keywords\" num_papers"
        continue
    fi
    
    echo "✓ Base data found for '$TOPIC'"
    echo "Running ablation study..."
    
    # Run the ablation script for this topic
    bash "$BASE_DIR/run_topk_ablation.sh" "$TOPIC" "${TOP_K_VALUES[@]}"
    
    if [ $? -eq 0 ]; then
        echo "✓ Ablation study completed for '$TOPIC'"
        
        # Collect all output directories for this topic
        for TOP_K in "${TOP_K_VALUES[@]}"; do
            OUTPUT_DIR="${DIR_QUERY}_top${TOP_K}"
            RESULT_DIR="$BASE_DIR/paper_data/$OUTPUT_DIR/literature_review_output"
            
            if [ -d "$RESULT_DIR" ]; then
                ALL_OUTPUTS+=("$OUTPUT_DIR")
                echo "  ✓ Found results: $OUTPUT_DIR"
            fi
        done
        
        # Also collect ablation summary directory
        ABLATION_DIR="$BASE_DIR/paper_data/${DIR_QUERY}_ablation_topk"
        if [ -d "$ABLATION_DIR" ]; then
            ALL_OUTPUTS+=("${DIR_QUERY}_ablation_topk")
            echo "  ✓ Found ablation summary: ${DIR_QUERY}_ablation_topk"
        fi
    else
        echo "✗ Ablation study failed for '$TOPIC'"
    fi
    
    echo ""
done

echo "=========================================="
echo "All Ablation Studies Complete!"
echo "=========================================="
echo "Total output directories: ${#ALL_OUTPUTS[@]}"
echo ""

# Prepare data for Hugging Face upload
echo "=========================================="
echo "Preparing data for Hugging Face upload"
echo "=========================================="

# Clear staging directory
rm -rf "$UPLOAD_DIR"
mkdir -p "$UPLOAD_DIR"

# Copy all literature_review_output directories to staging
for OUTPUT_NAME in "${ALL_OUTPUTS[@]}"; do
    SOURCE_DIR="$BASE_DIR/paper_data/$OUTPUT_NAME"
    
    if [ -d "$SOURCE_DIR/literature_review_output" ]; then
        DEST_DIR="$UPLOAD_DIR/$OUTPUT_NAME/literature_review_output"
        mkdir -p "$DEST_DIR"
        
        echo "Copying: $OUTPUT_NAME/literature_review_output"
        cp -r "$SOURCE_DIR/literature_review_output/"* "$DEST_DIR/" 2>/dev/null || true
        
        # Create a metadata file
        cat > "$DEST_DIR/metadata.json" << EOF
{
    "experiment_name": "$OUTPUT_NAME",
    "upload_date": "$(date -Iseconds)",
    "description": "Literature review output from SurveyG ablation study"
}
EOF
    elif [ -d "$SOURCE_DIR" ] && [[ "$OUTPUT_NAME" == *"_ablation_topk" ]]; then
        # This is an ablation summary directory
        DEST_DIR="$UPLOAD_DIR/$OUTPUT_NAME"
        mkdir -p "$DEST_DIR"
        
        echo "Copying: $OUTPUT_NAME (ablation summary)"
        cp -r "$SOURCE_DIR/"* "$DEST_DIR/" 2>/dev/null || true
    fi
done

# Create overall README for the upload
cat > "$UPLOAD_DIR/README.md" << 'EOF'
# SurveyG Ablation Study Results

This repository contains the results of Top-K ablation studies for literature review generation using SurveyG.

## Experiments

### Topics Analyzed
1. **Retrieval-Augmented Generation for Large Language Models**
2. **A Survey on Visual Transformer**
3. **Knowledge Graph Embedding**

### Ablation Study: Top-K Foundation Papers

This study investigates the impact of varying the number of foundation papers (Layer 1) on the quality and coverage of generated literature reviews.

**Top-K values tested:** 6, 8, 10, 12, 14

### Directory Structure

```
{topic_name}_top{K}/literature_review_output/
├── literature_review.tex       # LaTeX source
├── literature_review.pdf       # Generated PDF (if compilation succeeded)
├── references.bib              # Bibliography
├── outline.json                # Survey outline structure
└── metadata.json               # Experiment metadata

{topic_name}_ablation_topk/
├── experiment_config.txt       # Experiment configuration
├── literature_review_top6.pdf  # Comparative PDFs for each K value
├── literature_review_top8.pdf
├── literature_review_top10.pdf
├── literature_review_top12.pdf
└── literature_review_top14.pdf
```

### Key Findings

Compare the generated PDFs across different top-K values to observe:
- **Coverage**: How many research directions are identified
- **Depth**: Detail level in each section
- **Coherence**: Logical flow and organization
- **Citations**: Distribution and relevance of cited papers

### Usage

To reproduce these results:
```bash
./run_multi_topic_ablation.sh
```

### Generated Files

Each experiment includes:
- **LaTeX source** with validated citations and proper formatting
- **PDF output** (when LaTeX compilation succeeds)
- **Bibliography** with all cited papers
- **Outline** showing the survey structure
- **Metadata** tracking experiment parameters

### Citation

If you use these results or the SurveyG tool, please cite:
```bibtex
@software{surveyg2024,
  title={SurveyG: Automated Literature Review Generation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/SurveyG}
}
```

### Contact

For questions or issues, please open an issue on the GitHub repository.

---
*Generated on $(date)*
EOF

# Create experiment summary
cat > "$UPLOAD_DIR/EXPERIMENT_SUMMARY.txt" << EOF
SurveyG Multi-Topic Ablation Study
===================================

Date: $(date)
Hugging Face Repository: $HF_REPO

Topics Processed: ${#TOPICS[@]}
$(for topic in "${TOPICS[@]}"; do echo "  - $topic"; done)

Top-K Values: ${TOP_K_VALUES[*]}

Total Experiments: $((${#TOPICS[@]} * ${#TOP_K_VALUES[@]}))

Output Directories Generated: ${#ALL_OUTPUTS[@]}
$(for output in "${ALL_OUTPUTS[@]}"; do echo "  - $output"; done)

Files Uploaded:
$(find "$UPLOAD_DIR" -type f | wc -l) files

Total Size:
$(du -sh "$UPLOAD_DIR" | cut -f1)

===================================
EOF

echo ""
echo "✓ Staging directory prepared: $UPLOAD_DIR"
echo ""
du -sh "$UPLOAD_DIR"
echo ""

# Upload to Hugging Face
echo "=========================================="
echo "Uploading to Hugging Face"
echo "=========================================="

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Create Python upload script
cat > "$BASE_DIR/upload_to_hf.py" << 'EOFPYTHON'
import os
import sys
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_to_huggingface(upload_dir, repo_id):
    """Upload directory to Hugging Face Hub"""
    print(f"Initializing upload to {repo_id}...")
    
    api = HfApi()
    
    # Check if logged in
    try:
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception as e:
        print("✗ Not logged in to Hugging Face")
        print("Please run: huggingface-cli login")
        return False
    
    # Create repository if it doesn't exist
    try:
        print(f"Creating/accessing repository: {repo_id}")
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"✓ Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"✗ Error creating repository: {e}")
        return False
    
    # Upload folder
    try:
        print(f"Uploading files from {upload_dir}...")
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload ablation study results - {os.path.basename(upload_dir)}"
        )
        print(f"✓ Upload complete!")
        print(f"✓ View at: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python upload_to_hf.py <upload_dir> <repo_id>")
        sys.exit(1)
    
    upload_dir = sys.argv[1]
    repo_id = sys.argv[2]
    
    success = upload_to_huggingface(upload_dir, repo_id)
    sys.exit(0 if success else 1)
EOFPYTHON

# Run the upload
python "$BASE_DIR/upload_to_hf.py" "$UPLOAD_DIR" "$HF_REPO"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ SUCCESS!"
    echo "=========================================="
    echo "All experiments completed and uploaded!"
    echo ""
    echo "View results at:"
    echo "https://huggingface.co/datasets/$HF_REPO"
    echo ""
    echo "Local results preserved in:"
    echo "$UPLOAD_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "⚠ Upload Failed"
    echo "=========================================="
    echo "Experiments completed successfully, but upload failed."
    echo "Results are still available locally at:"
    echo "$UPLOAD_DIR"
    echo ""
    echo "To upload manually:"
    echo "1. Login: huggingface-cli login"
    echo "2. Run: python upload_to_hf.py \"$UPLOAD_DIR\" \"$HF_REPO\""
    echo "=========================================="
fi

# Clean up
echo ""
read -p "Keep staging directory? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning up staging directory..."
    rm -rf "$UPLOAD_DIR"
    echo "✓ Cleaned up"
fi

echo ""
echo "Script complete!"
