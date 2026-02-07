#!/usr/bin/env python
"""
Progressive Iteration Ablation Script
=====================================
This script saves intermediate versions of each section/subsection after every iteration,
allowing you to resume from any checkpoint and progressively improve without re-running.

Key Features:
- Saves version after each iteration (iter0, iter1, iter2, ...)
- Can resume from any iteration checkpoint
- Avoids re-running entire process
- Preserves all intermediate states

Usage:
    python writing_survey_progressive_iter.py "query" "output_base_dir" start_iter max_iter
    
Example:
    # Run iterations 0-5, saving each version
    python writing_survey_progressive_iter.py "knowledge_graph_embedding" \
        "paper_data/knowledge_graph_embedding_iter_ablation" 0 5
    
    # Resume from iteration 3, continue to iteration 5
    python writing_survey_progressive_iter.py "knowledge_graph_embedding" \
        "paper_data/knowledge_graph_embedding_iter_ablation" 3 5
"""

import sys
import os
import shutil
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from pathlib import Path
from dotenv import load_dotenv


def copy_checkpoints_between_iterations(src_dir, dst_dir):
    """
    Copy checkpoint files from one iteration to another to resume improvement.
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    # Copy all .tex checkpoint files
    copied = 0
    for filename in os.listdir(src_dir):
        if filename.endswith('_checkpoint.tex'):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)
            copied += 1
    
    return copied


def save_iteration_snapshot(save_dir, iteration, metadata=None):
    """
    Save a snapshot of the current iteration state.
    Creates a versioned copy of all outputs.
    """
    snapshot_dir = f"{save_dir}_snapshot_iter{iteration}"
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # Copy all files from current save_dir to snapshot
    for filename in os.listdir(save_dir):
        src_path = os.path.join(save_dir, filename)
        dst_path = os.path.join(snapshot_dir, filename)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
    
    # Save metadata
    if metadata:
        metadata_path = os.path.join(snapshot_dir, f"iter{iteration}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved iteration {iteration} snapshot: {snapshot_dir}")
    return snapshot_dir


def main():
    """
    Main entry point for progressive iteration ablation.
    Runs multiple iterations sequentially, saving checkpoints after each.
    """
    
    if len(sys.argv) < 5:
        print("Usage: python writing_survey_progressive_iter.py \"query\" \"output_base_dir\" start_iter max_iter")
        print("")
        print("Arguments:")
        print("  query: Research topic (e.g., 'knowledge_graph_embedding')")
        print("  output_base_dir: Base directory for ablation outputs")
        print("  start_iter: Starting iteration (0 for fresh, or resume from checkpoint)")
        print("  max_iter: Maximum iteration to run (0-5)")
        print("")
        print("Example:")
        print("  # Run iterations 0-5 (6 versions total)")
        print("  python writing_survey_progressive_iter.py \"knowledge_graph_embedding\" \\")
        print("         \"paper_data/knowledge_graph_embedding_iter_ablation\" 0 5")
        print("")
        print("  # Resume from iteration 3, continue to 5")
        print("  python writing_survey_progressive_iter.py \"knowledge_graph_embedding\" \\")
        print("         \"paper_data/knowledge_graph_embedding_iter_ablation\" 3 5")
        return 1
    
    query = sys.argv[1]
    output_base_dir = sys.argv[2]
    start_iter = int(sys.argv[3])
    max_iter = int(sys.argv[4])
    
    # Validation
    if start_iter < 0 or max_iter < 0 or start_iter > max_iter:
        print("❌ ERROR: Invalid iteration range")
        print(f"   start_iter={start_iter}, max_iter={max_iter}")
        print("   Requirements: 0 <= start_iter <= max_iter")
        return 1
    
    # Load API key
    load_dotenv(Path(".env"))
    API_KEY = os.getenv("API_KEY")
    
    if not API_KEY:
        print("❌ ERROR: API_KEY not found in .env file")
        return 1
    
    print("=" * 80)
    print("PROGRESSIVE ITERATION ABLATION")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Iteration Range: {start_iter} → {max_iter} ({max_iter - start_iter + 1} runs)")
    print(f"Output Base: {output_base_dir}")
    print("=" * 80)
    print("")
    
    try:
        from writing.writing_survey import LiteratureReviewGenerator
        import google.generativeai as genai
        
        class ProgressiveIterationGenerator(LiteratureReviewGenerator):
            """
            Generator that supports progressive iteration with checkpointing.
            Saves intermediate states after each improvement iteration.
            """
            
            def __init__(self, query, api_key, ablation_base, current_iteration):
                # Initialize base paths
                query_dir = query.replace(' ', '_').replace(':', '')
                
                # Call parent constructor
                super().__init__(query, api_key, ablation_study='', top_k=None)
                
                # Override paths to use ablation base
                self.output_query_dir = ablation_base
                self.base_query_dir = f"paper_data/{query_dir}"
                
                # Set current iteration count
                self.max_improvement_iterations = current_iteration
                self.current_iteration = current_iteration
                
                # Output directory for this iteration
                self.save_dir = f"{ablation_base}/literature_review_output_iter{current_iteration}"
                os.makedirs(self.save_dir, exist_ok=True)
                
                # Data paths from base directory
                self.keyword_dir = f"{self.base_query_dir}/keywords"
                self.graph_path = f"{ablation_base}/info/paper_citation_graph.json"
                self.node_info_path = f"{self.keyword_dir}/processed_checkpoint.json"
                
                # Verify required files
                if not os.path.exists(self.graph_path):
                    raise FileNotFoundError(f"Graph not found: {self.graph_path}")
                if not os.path.exists(self.node_info_path):
                    raise FileNotFoundError(f"Node info not found: {self.node_info_path}")
                
                # Reload graph
                self.G, self.id2node_info = self.load_graph(self.graph_path, self.node_info_path)
                
                # Load paths data from ablation base
                paths_dir = f"{ablation_base}/paths"
                self.layer_method_group_json = json.load(open(f"{paths_dir}/layer_method_group_summary.json", "r"))
                self.develop_direction = json.load(open(f"{paths_dir}/layer1_seed_taxonomy.json", "r"))
                self.community_summary = json.load(open(f"{paths_dir}/communities_summary.json", "r"))
                
                # Outline path
                self.outline_path = f"{self.save_dir}/survey_outline.json"
                
                # Copy outline if needed
                source_outline = f"{paths_dir}/../literature_review_output/survey_outline.json"
                if not os.path.exists(source_outline):
                    source_outline = f"paper_data/{query_dir}/literature_review_output/survey_outline.json"
                
                if os.path.exists(source_outline) and not os.path.exists(self.outline_path):
                    shutil.copy2(source_outline, self.outline_path)
                
                print(f"  ✓ Initialized iteration {current_iteration}")
                print(f"  ✓ Output: {self.save_dir}")
        
        # Run iterations progressively
        for iteration in range(start_iter, max_iter + 1):
            print("")
            print("=" * 80)
            print(f"ITERATION {iteration}/{max_iter}")
            print("=" * 80)
            print("")
            
            # Initialize generator for this iteration
            print(f"📝 Initializing generator for iteration {iteration}...")
            generator = ProgressiveIterationGenerator(
                query=query,
                api_key=API_KEY,
                ablation_base=output_base_dir,
                current_iteration=iteration
            )
            
            # If resuming (iteration > 0), copy checkpoints from previous iteration
            if iteration > 0:
                prev_dir = f"{output_base_dir}/literature_review_output_iter{iteration-1}"
                curr_dir = generator.save_dir
                
                if os.path.exists(prev_dir):
                    print(f"📋 Resuming from iteration {iteration-1}...")
                    copied = copy_checkpoints_between_iterations(prev_dir, curr_dir)
                    print(f"  ✓ Copied {copied} checkpoint files from previous iteration")
                else:
                    print(f"⚠️  Warning: Previous iteration directory not found: {prev_dir}")
                    print(f"   Starting fresh for iteration {iteration}")
            
            # Generate review with current iteration count
            print(f"🚀 Running iteration {iteration} (improvement loops: {iteration})...")
            print("")
            
            start_time = datetime.now()
            
            review_data = generator.generate_complete_literature_review(
                output_base_dir,
                f"A Comprehensive Literature Review: {query}"
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Save results
            if "error" not in review_data:
                generator.save_literature_review(review_data)
                
                # Save metadata for this iteration
                metadata = {
                    'iteration': iteration,
                    'max_iterations': max_iter,
                    'query': query,
                    'output_base_dir': output_base_dir,
                    'generation_date': end_time.isoformat(),
                    'duration_seconds': duration,
                    'sections_count': len(review_data.get('sections', {}))
                }
                
                metadata_path = f"{generator.save_dir}/iteration_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Create snapshot of this iteration
                save_iteration_snapshot(generator.save_dir, iteration, metadata)
                
                print("")
                print("=" * 80)
                print(f"✅ Iteration {iteration} Completed!")
                print("=" * 80)
                print(f"📁 Output: {generator.save_dir}")
                print(f"⏱️  Duration: {duration:.1f}s ({duration/60:.1f} min)")
                print(f"📄 Sections: {len(review_data.get('sections', {}))}")
                
                # Check PDF
                pdf_path = f"{generator.save_dir}/literature_review.pdf"
                if os.path.exists(pdf_path):
                    size = os.path.getsize(pdf_path) / 1024
                    print(f"📄 PDF: {size:.1f} KB")
                
                print("=" * 80)
                
            else:
                print(f"❌ Error in iteration {iteration}: {review_data['error']}")
                print("Stopping ablation study.")
                return 1
        
        # Final summary
        print("")
        print("=" * 80)
        print("🎉 PROGRESSIVE ITERATION ABLATION COMPLETED")
        print("=" * 80)
        print(f"✓ Completed iterations: {start_iter} → {max_iter}")
        print(f"✓ Total versions saved: {max_iter - start_iter + 1}")
        print("")
        print("📁 Output directories:")
        for i in range(start_iter, max_iter + 1):
            output_dir = f"{output_base_dir}/literature_review_output_iter{i}"
            snapshot_dir = f"{output_dir}_snapshot_iter{i}"
            print(f"  - Iteration {i}:")
            print(f"      Working: {output_dir}")
            print(f"      Snapshot: {snapshot_dir}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
