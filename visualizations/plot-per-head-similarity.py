import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def extract_step_number(step_name):
    return int(step_name.replace('step', ''))

def load_per_head_data(results_dir, layer_numbers=None):
    per_head_data = {}
    
    # Find all revision directories
    revision_dirs = glob.glob(os.path.join(results_dir, "revision_step*"))
    
    for revision_dir in revision_dirs:
        # Extract step name from revision directory (e.g., "revision_step0" -> "step0")
        step_name = os.path.basename(revision_dir).replace('revision_', '')
        
        # Find all layer directories within this revision
        layer_dirs = glob.glob(os.path.join(revision_dir, "layer_*"))
        
        for layer_dir in layer_dirs:
            layer_num = int(os.path.basename(layer_dir).split('_')[1])
            
            # Skip if not in requested layers
            if layer_numbers is not None and layer_num not in layer_numbers:
                continue
            
            # Load the per_head_results.json file
            results_file = os.path.join(layer_dir, "per_head_results.json")
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    step_data = json.load(f)
                
                # Initialize layer data if not exists
                if layer_num not in per_head_data:
                    per_head_data[layer_num] = {}
                
                per_head_data[layer_num][step_name] = step_data
    
    return per_head_data

def create_per_head_heatmap_data(layer_data):
    # Get all steps and sort them
    steps_list = sorted(layer_data.keys(), key=extract_step_number)
    
    # Get number of heads from first step
    first_step = steps_list[0]
    head_keys = sorted([k for k in layer_data[first_step].keys() if k.startswith('head_')],
                      key=lambda x: int(x.split('_')[1]))
    n_heads = len(head_keys)
    head_indices = [int(k.split('_')[1]) for k in head_keys]
    
    n_steps = len(steps_list)
    
    # Create 2D array: rows = heads, columns = steps
    heatmap_data = np.full((n_heads, n_steps), np.nan)
    
    # Fill the array
    for step_idx, step_name in enumerate(steps_list):
        step_data = layer_data[step_name]
        for head_idx, head_key in enumerate(head_keys):
            if head_key in step_data and 'cosine_similarity' in step_data[head_key]:
                heatmap_data[head_idx, step_idx] = step_data[head_key]['cosine_similarity']
    
    return heatmap_data, steps_list, head_indices

def plot_single_layer_heatmap(layer_data, layer_num, output_file):
    heatmap_data, steps_list, head_indices = create_per_head_heatmap_data(layer_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    
    # Set labels
    ax.set_title(
        f'Per-Head Attention Cosine Similarity - Layer {layer_num+1}',
        fontsize=22,
    )
    ax.set_xlabel('Training Step', fontsize=20)
    ax.set_ylabel('Head Index', fontsize=20)
    
    # Set x-axis ticks and labels
    step_numbers = [extract_step_number(s) for s in steps_list]
    ax.set_xticks(range(len(steps_list)))
    ax.set_xticklabels([f'{num}' for num in step_numbers], rotation=45, ha='right')
    
    # Set y-axis ticks and labels (1-indexed)
    ax.set_yticks(range(len(head_indices)))
    ax.set_yticklabels([f'{head+1}' for head in head_indices])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=18)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Heatmap for layer {layer_num+1} saved as {output_file}")
    
    # Print summary statistics
    print(f"\nLayer {layer_num+1} Summary:")
    print(f"  Steps: {len(steps_list)} ({step_numbers[0]} to {step_numbers[-1]})")
    print(f"  Heads: {len(head_indices)}")
    print(f"  Min similarity: {np.nanmin(heatmap_data):.4f}")
    print(f"  Max similarity: {np.nanmax(heatmap_data):.4f}")
    print(f"  Mean similarity: {np.nanmean(heatmap_data):.4f}")
    
    # Find best and worst heads on average
    mean_per_head = np.nanmean(heatmap_data, axis=1)
    best_head = np.nanargmax(mean_per_head)
    worst_head = np.nanargmin(mean_per_head)
    print(f"  Best head (avg): {head_indices[best_head]+1} (similarity: {mean_per_head[best_head]:.4f})")
    print(f"  Worst head (avg): {head_indices[worst_head]+1} (similarity: {mean_per_head[worst_head]:.4f})")

def plot_multi_layer_comparison(per_head_data, output_file):
    layer_nums = sorted(per_head_data.keys())
    n_layers = len(layer_nums)
    
    if n_layers == 0:
        print("No data to plot")
        return
    
    # Determine grid layout
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Flatten axes array for easier indexing
    if n_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_layers > 1 else [axes]
    
    # Compute global min/max for consistent color scale
    global_min = float('inf')
    global_max = float('-inf')
    heatmap_data_list = []
    
    for layer_num in layer_nums:
        heatmap_data, steps_list, head_indices = create_per_head_heatmap_data(per_head_data[layer_num])
        heatmap_data_list.append((heatmap_data, steps_list, head_indices))
        global_min = min(global_min, np.nanmin(heatmap_data))
        global_max = max(global_max, np.nanmax(heatmap_data))
    
    # Plot each layer
    for idx, layer_num in enumerate(layer_nums):
        ax = axes[idx]
        heatmap_data, steps_list, head_indices = heatmap_data_list[idx]
        
        im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', 
                      interpolation='nearest', vmin=global_min, vmax=global_max)
        
        ax.set_title(f'Layer {layer_num+1}', fontsize=18)
        ax.set_xlabel('Training Step', fontsize=16)
        ax.set_ylabel('Head Index', fontsize=16)
        
        # Set x-axis ticks and labels (show all training steps)
        step_numbers = [extract_step_number(s) for s in steps_list]
        ax.set_xticks(range(len(steps_list)))
        ax.set_xticklabels([f'{step_numbers[i]}' for i in range(len(steps_list))], rotation=45, ha='right', fontsize=14)
        
        # Set y-axis ticks and labels (show all head indices, 1-indexed)
        ax.set_yticks(range(len(head_indices)))
        ax.set_yticklabels([f'{head+1}' for head in head_indices], fontsize=14)
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Cosine Similarity', fontsize=18)
    
    # Add overall title
    fig.suptitle(
        'Per-Head Attention Cosine Similarity Across Training Steps',
        fontsize=22,
        y=0.98,
    )
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Multi-layer comparison heatmap saved as {output_file}")

def plot_head_evolution(
    per_head_data, layer_num, selected_heads=None, output_file=None
):
    if layer_num not in per_head_data:
        print(f"No data for layer {layer_num+1}")
        return
    
    layer_data = per_head_data[layer_num]
    heatmap_data, steps_list, head_indices = create_per_head_heatmap_data(layer_data)
    
    step_numbers = [extract_step_number(s) for s in steps_list]
    
    # Determine which heads to plot
    if selected_heads is None:
        selected_heads = head_indices
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each selected head
    for head_idx in selected_heads:
        if head_idx in head_indices:
            data_idx = head_indices.index(head_idx)
            ax.plot(step_numbers, heatmap_data[data_idx, :], 
                   marker='o', label=f'Head {head_idx+1}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Training Step', fontsize=18)
    ax.set_ylabel('Cosine Similarity', fontsize=18)
    ax.set_title(
        f'Head Evolution Across Training - Layer {layer_num+1}',
        fontsize=20,
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Head evolution plot saved as {output_file}")
    else:
        plt.show()

def main():
    """Main function to create per-head visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize per-head attention cosine similarity')
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results_per_head',
        help='Directory containing per-head results'
    )
    parser.add_argument('--output_dir', type=str, default='/nobackup/sim/attn-dynamics',
                       help='Directory to save output plots')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='Specific layers to plot (default: all available)')
    parser.add_argument('--plot_type', type=str, choices=['single', 'multi', 'evolution', 'all'],
                       default='multi', help='Type of plot to create')
    parser.add_argument('--selected_heads', type=int, nargs='+', default=None,
                       help='Specific heads to plot for evolution plot')
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return
    
    print(f"Loading per-head data from {args.results_dir}...")
    per_head_data = load_per_head_data(args.results_dir, args.layers)
    
    if not per_head_data:
        print("No per-head data found!")
        return
    
    layer_nums = sorted(per_head_data.keys())
    print(f"Found data for layers: {[l+1 for l in layer_nums]}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create plots based on plot_type
    if args.plot_type in ['single', 'all']:
        # Create individual heatmaps for each layer
        for layer_num in layer_nums:
            output_file = os.path.join(
                args.output_dir, f"per_head_heatmap_layer_{layer_num}.pdf"
            )
            plot_single_layer_heatmap(
                per_head_data[layer_num],
                layer_num,
                output_file,
            )
    
    if args.plot_type in ['multi', 'all']:
        # Create multi-layer comparison plot
        if len(layer_nums) > 1:
            output_file = os.path.join(
                args.output_dir, "per_head_multi_layer_comparison.pdf"
            )
            plot_multi_layer_comparison(
                per_head_data, output_file
            )
        else:
            print("Skipping multi-layer plot (only one layer available)")
    
    if args.plot_type in ['evolution', 'all']:
        # Create evolution plots for each layer
        for layer_num in layer_nums:
            output_file = os.path.join(
                args.output_dir, f"per_head_evolution_layer_{layer_num}.pdf"
            )
            plot_head_evolution(
                per_head_data,
                layer_num,
                args.selected_heads,
                output_file,
            )
    
    print(f"\nAll plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
