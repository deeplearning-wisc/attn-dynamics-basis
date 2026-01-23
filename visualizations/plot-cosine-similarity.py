import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def extract_step_number(revision_dir):
    """Extract step number from revision directory name."""
    return int(revision_dir.split('revision_step')[-1])

def load_cosine_similarity_data(results_dir):
    """Load cosine similarity data from all revision steps."""
    attention_data = {}
    layer_output_data = {}
    layer_mid_data = {}
    
    # Find all revision directories
    revision_dirs = glob.glob(os.path.join(results_dir, "revision_step*"))
    
    for revision_dir in revision_dirs:
        step = extract_step_number(revision_dir)
        
        # Load attention comparison data
        attention_file = os.path.join(revision_dir, "matrix_theoretical", "attention_comparison_summary.json")
        if os.path.exists(attention_file):
            with open(attention_file, 'r') as f:
                attention_data[step] = json.load(f)
        
        # Load layer mid (pre-MLP) comparison data
        layer_mid_file = os.path.join(revision_dir, "matrix_theoretical", "cosine_similarity_summary_mids.json")
        if os.path.exists(layer_mid_file):
            with open(layer_mid_file, 'r') as f:
                layer_mid_data[step] = json.load(f)
        
        # Load layer output comparison data
        layer_output_file = os.path.join(revision_dir, "matrix_theoretical", "cosine_similarity_summary.json")
        if os.path.exists(layer_output_file):
            with open(layer_output_file, 'r') as f:
                layer_output_data[step] = json.load(f)
    
    return attention_data, layer_mid_data, layer_output_data

def extract_layer_similarities(data_dict):
    """Extract cosine similarities for all layers from a data dictionary."""
    similarities = []
    for layer_key, layer_data in data_dict.items():
        if 'cosine_similarity' in layer_data:
            similarities.append(layer_data['cosine_similarity'])
    return similarities

def create_heatmap_data(data_dict, steps_list):
    """Create 2D array for heatmap from layer data."""
    # Get all layer numbers and sort them
    all_layers = set()
    for step_data in data_dict.values():
        for layer_key in step_data.keys():
            if layer_key.startswith('layer_'):
                layer_num = int(layer_key.split('_')[1])
                all_layers.add(layer_num)
    
    layers = sorted(all_layers)
    n_layers = len(layers)
    n_steps = len(steps_list)
    
    # Create 2D array: rows = layers, columns = steps
    heatmap_data = np.full((n_layers, n_steps), np.nan)
    
    # Fill the array
    for step_idx, step in enumerate(steps_list):
        if step in data_dict:
            step_data = data_dict[step]
            for layer_idx, layer_num in enumerate(layers):
                layer_key = f'layer_{layer_num}'
                if layer_key in step_data and 'cosine_similarity' in step_data[layer_key]:
                    heatmap_data[layer_idx, step_idx] = step_data[layer_key]['cosine_similarity']
    
    return heatmap_data, layers

def plot_cosine_similarity_heatmaps(
    results_dir,
    output_file="cosine_similarity_heatmaps.pdf",
    title="Cosine Similarity Across Checkpoints",
):
    # Load data
    attention_data, layer_mid_data, layer_output_data = load_cosine_similarity_data(results_dir)
    
    # Sort steps
    attention_steps = sorted(attention_data.keys())
    layer_mid_steps = sorted(layer_mid_data.keys())
    layer_output_steps = sorted(layer_output_data.keys())
    
    # Create heatmap data
    attention_heatmap, attention_layers = create_heatmap_data(attention_data, attention_steps)
    layer_mid_heatmap, layer_mid_layers = create_heatmap_data(layer_mid_data, layer_mid_steps)
    layer_output_heatmap, layer_output_layers = create_heatmap_data(layer_output_data, layer_output_steps)
    
    # Create subplots (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5))
    
    # Plot attention comparison heatmap (left)
    if attention_steps and not np.isnan(attention_heatmap).all():
        im1 = ax1.imshow(attention_heatmap, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax1.set_title('Attention Mapping Cosine Similarity', fontsize=22)
        ax1.set_xlabel('Training Step', fontsize=20)
        ax1.set_ylabel('Layer Number', fontsize=20)
        
        # Set x-axis ticks and labels
        step_indices = range(len(attention_steps))
        ax1.set_xticks(step_indices)
        ax1.set_xticklabels([f'{step}' for step in attention_steps], rotation=45, ha='right')
        
        # Set y-axis ticks and labels (layers from top to bottom, 1-indexed)
        ax1.set_yticks(range(len(attention_layers)))
        ax1.set_yticklabels([f'{layer+1}' for layer in attention_layers])
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Cosine Similarity', fontsize=18)
    
    # Plot layer mid (pre-MLP) comparison heatmap (middle)
    if layer_mid_steps and not np.isnan(layer_mid_heatmap).all():
        im2 = ax2.imshow(layer_mid_heatmap, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('No MLP Embedding Cosine Similarity', fontsize=22)
        ax2.set_xlabel('Training Step', fontsize=20)
        ax2.set_ylabel('Layer Number', fontsize=20)
        
        # Set x-axis ticks and labels
        step_indices = range(len(layer_mid_steps))
        ax2.set_xticks(step_indices)
        ax2.set_xticklabels([f'{step}' for step in layer_mid_steps], rotation=45, ha='right')
        
        # Set y-axis ticks and labels (layers from top to bottom, 1-indexed)
        ax2.set_yticks(range(len(layer_mid_layers)))
        ax2.set_yticklabels([f'{layer+1}' for layer in layer_mid_layers])
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Cosine Similarity', fontsize=18)
    
    # Plot layer output comparison heatmap (right)
    if layer_output_steps and not np.isnan(layer_output_heatmap).all():
        im3 = ax3.imshow(layer_output_heatmap, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax3.set_title('Embedding Mapping Cosine Similarity', fontsize=22)
        ax3.set_xlabel('Training Step', fontsize=20)
        ax3.set_ylabel('Layer Number', fontsize=20)
        
        # Set x-axis ticks and labels
        step_indices = range(len(layer_output_steps))
        ax3.set_xticks(step_indices)
        ax3.set_xticklabels([f'{step}' for step in layer_output_steps], rotation=45, ha='right')
        
        # Set y-axis ticks and labels (layers from top to bottom, 1-indexed)
        ax3.set_yticks(range(len(layer_output_layers)))
        ax3.set_yticklabels([f'{layer+1}' for layer in layer_output_layers])
        
        # Add colorbar
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Cosine Similarity', fontsize=18)
    
    # Add overall title
    fig.suptitle(title, fontsize=26)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    
def main():
    """Main function to create the plot."""
    parser = argparse.ArgumentParser(
        description="Plot cosine similarity heatmaps across checkpoints."
    )
    parser.add_argument(
        "--fineweb",
        action="store_true",
        help="Use FineWeb results directory, filename, and title.",
    )
    args = parser.parse_args()

    if args.fineweb:
        results_dir = "results_fw"
        output_file = "cosine_similarity_heatmaps_fw.pdf"
        title = "Cosine Similarity Across Checkpoints (FineWeb)"
    else:
        results_dir = "results"
        output_file = "cosine_similarity_heatmaps.pdf"
        title = "Cosine Similarity Across Checkpoints"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    plot_cosine_similarity_heatmaps(results_dir, output_file, title)

if __name__ == "__main__":
    main()
