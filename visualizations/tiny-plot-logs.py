import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
from pathlib import Path

def load_training_logs(log_file):
    with open(log_file, 'r') as f:
        logs = json.load(f)
    return logs

def plot_cosine_similarities(logs, output_dir, log_file_name):
    # Extract epochs
    epochs = [log['epoch'] for log in logs]
    
    # Extract cosine similarities data
    cosine_sims = logs[0]['cosine_similarities']
    
    # Group similarities by parameter type
    param_types = {}
    for key in cosine_sims.keys():
        if key.startswith('layer_'):
            # Extract parameter type (e.g., 'W_Q_bar' or 'V_B_bar_Phi_bar')
            param_type = key.split('_', 2)[2]
            if param_type not in param_types:
                param_types[param_type] = []
            param_types[param_type].append(key)
        elif key == 'W_O_B_bar':
            # Handle W_O_B_bar
            if 'W_O_B_bar' not in param_types:
                param_types['W_O_B_bar'] = []
            param_types['W_O_B_bar'].append(key)
    
    # Create the plot
    plt.figure(figsize=(7, 4))
    
    colors = ['red', 'blue', 'green']
    
    names = ['$Attention$', '$Value$', '$Output$']

    for i, (param_type, keys) in enumerate(param_types.items()):
        color = colors[i % len(colors)]
        
        # Extract data for all layers of this parameter type
        layer_data = {}
        for key in keys:
            values = [log['cosine_similarities'][key] for log in logs]
            layer_data[key] = values
        
        # Calculate min and max across all layers for this parameter type
        all_values = []
        for values in layer_data.values():
            all_values.extend(values)
        
        if all_values:
            min_values = [min([layer_data[key][epoch_idx] for key in keys]) for epoch_idx in range(len(epochs))]
            max_values = [max([layer_data[key][epoch_idx] for key in keys]) for epoch_idx in range(len(epochs))]
            
            # Plot the shaded region
            plt.fill_between(epochs, min_values, max_values, alpha=0.3, color=color, 
                           label=f'{names[i]} (range across layers)')
            
            # Plot individual layer lines
            for key in keys:
                layer_num = key.split('_')[1] if key.startswith('layer_') else 'output'
                plt.plot(epochs, layer_data[key], color=color, alpha=0.6, linewidth=1)
    
    plt.title(f'Cosine Similarities Large LR', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Cosine Similarity', fontsize=18)
    plt.legend(fontsize=14, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)  
    plt.tight_layout()
    
    # Save the plot
    base_filename = log_file_name.replace('.json', '')
    plot_path = os.path.join(output_dir, f"cosine_similarities_{base_filename}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cosine similarities plot to: {plot_path}")
    
    # Print minimum cosine similarities analysis
    print(f"\nMinimum cosine similarities analysis for {log_file_name}:")
    print("=" * 60)
    
    for param_type, keys in param_types.items():
        # Collect all values for this parameter type across all epochs and layers
        all_param_values = []
        for key in keys:
            values = [log['cosine_similarities'][key] for log in logs]
            all_param_values.extend(values)
        
        if all_param_values:
            min_overall = min(all_param_values)
            max_overall = max(all_param_values)
            
            # Find which epoch and layer had the minimum
            min_epoch = None
            min_layer = None
            for epoch_idx, log in enumerate(logs):
                for key in keys:
                    value = log['cosine_similarities'][key]
                    if value == min_overall:
                        min_epoch = log['epoch']
                        min_layer = key.split('_')[1] if key.startswith('layer_') else 'output'
                        break
                if min_epoch is not None:
                    break
            
            print(f"{param_type}:")
            print(f"  Min across all layers and epochs: {min_overall:.6f}")
            print(f"  Max across all layers and epochs: {max_overall:.6f}")
            print(f"  Min occurred at epoch {min_epoch}, layer {min_layer}")
            
            # Also show min for each layer across all epochs
            print(f"  Min per layer across all epochs:")
            for key in keys:
                layer_values = [log['cosine_similarities'][key] for log in logs]
                layer_min = min(layer_values)
                layer_name = key.split('_')[1] if key.startswith('layer_') else 'output'
                print(f"    Layer {layer_name}: {layer_min:.6f}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Plot training logs from tiny-self-attn')
    parser.add_argument('--log_file', type=str, help='Log file to plot')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for plots')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot log file
    if not os.path.exists(args.log_file):
        print(f"Error: Log file {args.log_file} not found")
        return
    
    log_file_name = os.path.basename(args.log_file)
    logs = load_training_logs(args.log_file)
    plot_cosine_similarities(logs, args.output_dir, log_file_name)

if __name__ == "__main__":
    main()
