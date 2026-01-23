import os
import torch
import numpy as np
import sys
from transformer_lens import HookedTransformer
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns

def load_pythia_model(model_name="EleutherAI/pythia-1.4b-deduped", device=None, revision="step1"):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {model_name} on {device}...")
    if revision:
        print(f"Using revision: {revision}")
    
    # Set cache directory for model storage
    cache_dir = "models"
    os.makedirs(cache_dir, exist_ok=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)

    model = HookedTransformer.from_pretrained(
        model_name, hf_model=base_model, device=device,
        torch_dtype=torch.float32, revision=revision
    )
    
    return model

def load_openwebtext_subset(n_examples=100000, max_length=128, seed=42):
    # Set random seed for reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # Load OpenWebText dataset with streaming to avoid disk quota issues
    dataset = load_dataset("Skylion007/openwebtext", revision="convert/parquet", 
                          split="train", streaming=True)
    
    # Collect examples
    examples = []
    for example in tqdm(dataset, desc="Loading examples", total=n_examples):
        if len(examples) >= n_examples: 
            break
        
        text = example['text']
        if len(text) >= 4*max_length: # estimate tokens based on avg of 4 chars 
            examples.append(text)
    
    print(f"Loaded {len(examples)} examples from OpenWebText")
    return examples

def compute_layer_inputs_for_tokens(model, token_ids, device=None):
    if device is None:
        device = model.cfg.device
    
    print(f"Computing layer inputs for {len(token_ids)} tokens...")
    
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    layer_inputs = {}
    for layer_idx in range(n_layers):
        layer_inputs[f'layer_{layer_idx}'] = torch.zeros(len(token_ids), d_model, device=device)
    
    batch_size = 64 
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(token_ids), batch_size), desc="Processing token batches"):
            batch_end = min(batch_start + batch_size, len(token_ids))
            batch_tokens = token_ids[batch_start:batch_end]
            
            batch_tensor = torch.tensor(batch_tokens, device=device).unsqueeze(1)  # [batch_size, 1]
            
            logits, cache = model.run_with_cache(batch_tensor)
            
            for layer_idx in range(n_layers):
                layer_input = cache['resid_pre', layer_idx]
                layer_inputs[f'layer_{layer_idx}'][batch_start:batch_end] = layer_input[:, 0, :]
    
    print(f"Computed layer inputs for {len(token_ids)} tokens across {n_layers} layers")
    return layer_inputs

def compute_covariance_batched(matrix, batch_size=1000, verbose=False):
    n, d = matrix.shape
    device = matrix.device
    
    if verbose:
        print(f"  Computing {n}x{n} covariance in batches of {batch_size}...")
    
    # Center each row (subtract row means)
    row_means = matrix.mean(dim=1, keepdim=True)  # [n, 1]
    centered = matrix - row_means  # [n, d]
    
    # Compute covariance: cov[i,j] = (1/(d-1)) * sum_k(centered[i,k] * centered[j,k])
    # = (1/(d-1)) * centered @ centered.T
    # We compute this in row batches to avoid materializing the full n x n matrix at once
    
    cov = torch.zeros(n, n, device=device)
    
    num_batches = (n + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, n, batch_size)):
        if verbose and batch_idx % 10 == 0:
            print(f"    Batch {batch_idx+1}/{num_batches}")
        
        i_end = min(i + batch_size, n)
        batch_centered = centered[i:i_end]  # [batch_size, d]
        
        # Compute covariance for this batch of rows with all rows
        cov[i:i_end, :] = torch.matmul(batch_centered, centered.T) / (d - 1)
    
    return cov

def compute_per_head_attention_comparison(model, layer_inputs, theoretical_matrix, token_ids, 
                                          target_layer=0, device=None):
   if device is None:
        device = next(iter(layer_inputs.values())).device
    
    print(f"Computing per-head attention comparison for layer {target_layer} with {len(token_ids)} tokens...")
    
    n_tokens = len(token_ids)
    d_model = model.cfg.d_model
    n_heads = model.cfg.n_heads
    d_head = d_model // n_heads
    
    per_head_results = {}
    
    print("Extracting per-head key*query parameter matrices...")
    
    layer_name = f'layer_{target_layer}'
    
    with torch.no_grad():
        W_K = model.blocks[target_layer].attn.W_K  # [n_heads, d_model, d_head]
        W_Q = model.blocks[target_layer].attn.W_Q  # [n_heads, d_model, d_head]
        M_per_head = torch.einsum("hmd,hnd->hmn", W_Q, W_K)  # (n_heads, d_model, d_model)
    
    print(f"Comparing per-head key*query matrices with theoretical Q_bar...")
    
    prev_layer_outputs = layer_inputs[layer_name]  # [n_tokens, d_model]
    
    # Precompute theoretical covariance once
    row_norms = torch.norm(theoretical_matrix, p=2, dim=1, keepdim=True)
    
    # Only normalize rows with non-zero norms
    theoretical_matrix_scaled = torch.zeros_like(theoretical_matrix)
    non_zero_mask = (row_norms.squeeze() != 0)
    theoretical_matrix_scaled[non_zero_mask] = theoretical_matrix[non_zero_mask] / row_norms[non_zero_mask]
    
    print("  Computing theoretical covariance once...")
    theoretical_cov = compute_covariance_batched(theoretical_matrix_scaled, verbose=True)  # [n_tokens, n_tokens]
    
    # Normalize to prevent numerical issues
    theoretical_cov_norm = torch.norm(theoretical_cov, p='fro')
    theoretical_cov = theoretical_cov / theoretical_cov_norm
    
    for head_idx in range(n_heads):
        head_name = f'head_{head_idx}'
        print(f"\n  Processing {head_name}...")
        
        # Get the key*query matrix for this head
        actual_matrix = M_per_head[head_idx]  # [d_model, d_model]
        
        # Transform by layer inputs
        actual_matrix = torch.matmul(prev_layer_outputs, actual_matrix)  # [n_tokens, d_model]
        actual_matrix = torch.matmul(actual_matrix, prev_layer_outputs.T)  # [n_tokens, n_tokens]
        
        actual_cov = compute_covariance_batched(actual_matrix, verbose=True)  # [n_tokens, n_tokens]
        
        # Normalize to prevent numerical issues
        actual_cov_norm = torch.norm(actual_cov, p='fro')
        actual_cov = actual_cov / actual_cov_norm
        
        dot_product = torch.trace(torch.matmul(actual_cov.T, theoretical_cov))
        norm_actual = torch.norm(actual_cov, p='fro')
        norm_theoretical = torch.norm(theoretical_cov, p='fro')
        
        cosine_similarity = dot_product / (norm_actual * norm_theoretical)
        
        per_head_results[head_name] = {
            'cosine_similarity': float(cosine_similarity.item()),
            'head_idx': head_idx
        }
        
        print(f"  Layer {target_layer}, {head_name}: cosine_similarity={cosine_similarity:.4f}")
        
        del actual_matrix, actual_cov
        torch.cuda.empty_cache()
    
    return per_head_results

def compute_theoretical_matrices(model, examples, token_indices, max_seq_len=512, batch_size=8, device=None):
    """Simplified version that only computes Q_bar"""
    if device is None:
        device = model.cfg.device
    
    print(f"Computing theoretical Q_bar matrix from {len(examples)} examples...")
    print(f"Batch size: {batch_size}, Max sequence length: {max_seq_len}")
    
    vocab_size = model.cfg.d_vocab
    reduced_vocab_size = len(token_indices)
    seq_length = max_seq_len
    
    B_bar = torch.zeros(reduced_vocab_size, reduced_vocab_size, device=device)
    Phi_bar = torch.zeros(reduced_vocab_size, reduced_vocab_size, device=device)
    total_tokens = 0
    
    def create_uniform_attention_mask(seq_length):
        A = torch.zeros(seq_length, seq_length, device=device)
        for i in range(seq_length):
            A[i, :i+1] = 1.0 / (i + 1) 
        return A
    
    attention_mask = create_uniform_attention_mask(seq_length)
    
    processed_examples = 0
    
    for batch_start in tqdm(range(0, len(examples), batch_size), desc="Computing B_bar and Phi_bar"):
        batch_examples = examples[batch_start:batch_start + batch_size]
        
        batch_tokens = []
        valid_indices = []
        
        for i, example in enumerate(batch_examples):
            tokens = model.to_tokens(example, truncate=True)
            batch_tokens.append(tokens)
            valid_indices.append(batch_start + i)
        
        truncated_tokens = torch.zeros(len(batch_tokens), max_seq_len, dtype=torch.long, device=device)
        
        for i, tokens in enumerate(batch_tokens):
            seq_len = min(tokens.shape[1], max_seq_len)
            truncated_tokens[i, :seq_len] = tokens[0, :seq_len]
        
        token_indices_tensor = torch.tensor(token_indices, device=device)
        
        x_onehot = torch.zeros(len(batch_tokens), max_seq_len, len(token_indices), device=device, dtype=torch.float)
        
        token_mask = (truncated_tokens.unsqueeze(-1) == token_indices_tensor.unsqueeze(0).unsqueeze(0))
        x_onehot = token_mask.float()
        
        y_tokens = torch.zeros_like(truncated_tokens)
        y_tokens[:, :-1] = truncated_tokens[:, 1:]
        
        y_onehot = torch.zeros(len(batch_tokens), max_seq_len, len(token_indices), device=device, dtype=torch.float)
        y_token_mask = (y_tokens.unsqueeze(-1) == token_indices_tensor.unsqueeze(0).unsqueeze(0))
        y_onehot = y_token_mask.float()
        
        r = y_onehot - (1.0 / vocab_size)
        
        for batch_idx, example_idx in enumerate(valid_indices):
            seq_len = max_seq_len
            
            x_seq = x_onehot[batch_idx, :seq_len, :]
            r_seq = r[batch_idx, :seq_len, :]
            
            B_bar += torch.matmul(r_seq.T, x_seq)
            
            attended_context = torch.matmul(attention_mask[:seq_len, :seq_len], x_seq)
            
            Phi_bar += torch.matmul(r_seq.T, attended_context)
            
            total_tokens += seq_len
            processed_examples += 1
    
    if total_tokens > 0:
        B_bar = B_bar / total_tokens
        Phi_bar = Phi_bar / total_tokens
    
    print("Computing G_bar and Q_bar...")
    
    B_bar_Phi_bar = torch.matmul(B_bar.T, Phi_bar).T
    G_bar = torch.matmul(B_bar_Phi_bar.T, B_bar.T)
    
    t_coords = torch.arange(seq_length, device=device).view(-1, 1, 1)
    i_coords = torch.arange(seq_length, device=device).view(1, -1, 1)
    j_coords = torch.arange(seq_length, device=device).view(1, 1, -1)
    
    diagonal_mask = (i_coords == j_coords).float()
    off_diagonal_mask = (i_coords != j_coords).float()
    
    valid_mask = ((i_coords <= t_coords) & (j_coords <= t_coords)).float()
    
    diagonal_terms = (1.0 / (t_coords + 1)) * (1.0 - 1.0 / (t_coords + 1))
    off_diagonal_terms = -1.0 / ((t_coords + 1) * (t_coords + 1))
    
    J = (diagonal_mask * diagonal_terms + off_diagonal_mask * off_diagonal_terms) * valid_mask
    
    Q_bar = torch.zeros(reduced_vocab_size, reduced_vocab_size, device=device)
    
    for batch_start in tqdm(range(0, len(examples), batch_size), desc="Computing Q_bar"):
        batch_examples = examples[batch_start:batch_start + batch_size]
        
        batch_tokens = []
        
        for i, example in enumerate(batch_examples):
            tokens = model.to_tokens(example, truncate=True)
            batch_tokens.append(tokens)
        
        truncated_tokens = torch.zeros(len(batch_tokens), max_seq_len, dtype=torch.long, device=device)
        
        for i, tokens in enumerate(batch_tokens):
            seq_len = min(tokens.shape[1], max_seq_len)
            truncated_tokens[i, :seq_len] = tokens[0, :seq_len]
        
        token_indices_tensor = torch.tensor(token_indices, device=device)
        
        token_mask = (truncated_tokens.unsqueeze(-1) == token_indices_tensor.unsqueeze(0).unsqueeze(0))
        x_onehot = token_mask.float()
        
        y_tokens = torch.zeros_like(truncated_tokens)
        y_tokens[:, :-1] = truncated_tokens[:, 1:]
        
        y_token_mask = (y_tokens.unsqueeze(-1) == token_indices_tensor.unsqueeze(0).unsqueeze(0))
        y_onehot = y_token_mask.float()
        
        r = y_onehot - (1.0 / vocab_size)
        
        for batch_idx in range(len(batch_tokens)):
            seq_len = max_seq_len
            x_seq = x_onehot[batch_idx, :seq_len, :]
            r_seq = r[batch_idx, :seq_len, :]
            
            intermediate = torch.matmul(x_seq, G_bar)
            intermediate = torch.matmul(intermediate, r_seq.T)
            
            jacobian_applied = torch.einsum('tjk,tk->tj', J[:seq_len, :seq_len, :seq_len], intermediate.T).T
            
            Q_bar += torch.matmul(torch.matmul(x_seq.T, jacobian_applied), x_seq)
    
    print(f"Q_bar computation completed")
    return Q_bar

def plot_per_head_heatmap(results_by_step, target_layer, output_dir="results"):
    # Extract step names and sort them
    step_names = sorted(results_by_step.keys(), key=lambda x: int(x.replace('step', '')))
    
    # Get number of heads from first result
    first_result = results_by_step[step_names[0]]
    n_heads = len(first_result)
    
    # Create matrix: rows are heads, columns are steps
    similarity_matrix = np.zeros((n_heads, len(step_names)))
    
    for step_idx, step_name in enumerate(step_names):
        per_head_results = results_by_step[step_name]
        for head_idx in range(n_heads):
            head_name = f'head_{head_idx}'
            similarity_matrix[head_idx, step_idx] = per_head_results[head_name]['cosine_similarity']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(similarity_matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Head Index', fontsize=14)
    ax.set_title(f'Per-Head Attention Cosine Similarity - Layer {target_layer}', fontsize=16)
    
    # Set ticks
    ax.set_xticks(range(len(step_names)))
    ax.set_xticklabels(step_names, rotation=45, ha='right')
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels(range(n_heads))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=12)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    
    plot_filename = os.path.join(output_dir, f"per_head_heatmap_layer_{target_layer}.pdf")
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Per-head heatmap saved to: {plot_filename}")
    
    # PNG
    plot_filename_png = os.path.join(output_dir, f"per_head_heatmap_layer_{target_layer}.png")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(similarity_matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Head Index', fontsize=14)
    ax.set_title(f'Per-Head Attention Cosine Similarity - Layer {target_layer}', fontsize=16)
    ax.set_xticks(range(len(step_names)))
    ax.set_xticklabels(step_names, rotation=45, ha='right')
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels(range(n_heads))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_filename_png, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Per-head heatmap also saved to: {plot_filename_png}")

def main():
    parser = argparse.ArgumentParser(description='Per-head attention analysis across training steps')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-1.4b-deduped', 
                       help='Name of the Pythia model to load')
    parser.add_argument('--steps', type=str, nargs='+', default=['step1', 'step2', 'step4', 'step8', 
                                                                   'step16', 'step32', 'step64', 'step128', 'step256',
                                                                   'step512'],
                       help='List of training steps to analyze')
    parser.add_argument('--target_layer', type=int, default=0,
                       help='Which layer to analyze (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to load model on (auto-detect if None)')
    parser.add_argument('--output_dir', type=str, default='results_per_head',
                       help='Directory to save results')
    parser.add_argument('--n_examples', type=int, default=100000,
                       help='Number of examples to analyze from OpenWebText')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length for analysis')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for processing examples')
    parser.add_argument('--top_tokens', type=int, default=None,
                       help='Number of top tokens to analyze (default: use all tokens in vocab)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Per-head attention analysis across training steps")
    print(f"Target device: {device}")
    print(f"Target layer: {args.target_layer}")
    print(f"Training steps: {args.steps}")
    
    # Load data once (used for all steps)
    print(f"\nLoading OpenWebText data...")
    examples = load_openwebtext_subset(
        n_examples=args.n_examples, 
        max_length=args.max_seq_len
    )
    
    # Compute theoretical matrix using step0 model
    model_step0 = load_pythia_model(args.model_name, device, revision='step0')
    
    # Set top_tokens to vocab_size if not specified
    if args.top_tokens is None:
        args.top_tokens = model_step0.cfg.d_vocab
        print(f"Using all tokens in vocabulary: {args.top_tokens} tokens")
    elif args.top_tokens > model_step0.cfg.d_vocab:
        print(f"Warning: top_tokens ({args.top_tokens}) exceeds vocab_size ({model_step0.cfg.d_vocab}), using vocab_size")
        args.top_tokens = model_step0.cfg.d_vocab
    
    # Get top tokens or use all tokens if top_tokens equals vocab size
    if args.top_tokens == model_step0.cfg.d_vocab:
        print(f"Using all {model_step0.cfg.d_vocab} tokens from vocabulary (skipping token counting)")
        top_tokens = list(range(model_step0.cfg.d_vocab))
    else:
        print(f"Finding top {args.top_tokens} tokens...")
        token_counts = {}
        for example in tqdm(examples, desc="Counting tokens"):
            tokens = model_step0.to_tokens(example, truncate=True)
            for token_id in tokens[0].cpu().numpy():
                token_counts[token_id] = token_counts.get(token_id, 0) + 1
        
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        top_tokens = [token_id for token_id, count in sorted_tokens[:args.top_tokens]]
        print(f"Found top {len(top_tokens)} tokens")
    
    # Compute theoretical Q_bar
    Q_bar = compute_theoretical_matrices(
        model=model_step0,
        examples=examples,
        token_indices=top_tokens,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device
    )
    
    del model_step0
    torch.cuda.empty_cache()
    
    # Now iterate over all steps
    results_by_step = {}
    
    for step in args.steps:
        print(f"\n{'='*80}")
        print(f"Processing {step}...")
        print(f"{'='*80}")
        
        # Load model for this step
        model = load_pythia_model(args.model_name, device, revision=step)
        
        # Compute layer inputs
        layer_inputs = compute_layer_inputs_for_tokens(
            model=model,
            token_ids=top_tokens,
            device=device
        )
        
        # Compute per-head attention comparison
        per_head_results = compute_per_head_attention_comparison(
            model=model,
            layer_inputs=layer_inputs,
            theoretical_matrix=Q_bar,
            token_ids=top_tokens,
            target_layer=args.target_layer,
            device=device
        )
        
        results_by_step[step] = per_head_results
        
        # Save intermediate results with revision-based folder structure
        revision_output_dir = os.path.join(args.output_dir, f"revision_{step}", f"layer_{args.target_layer}")
        os.makedirs(revision_output_dir, exist_ok=True)
        
        results_file = os.path.join(revision_output_dir, "per_head_results.json")
        with open(results_file, 'w') as f:
            json.dump(per_head_results, f, indent=2)
        
        print(f"Results for {step} saved to: {results_file}")
        
        # Clean up
        del model
        del layer_inputs
        torch.cuda.empty_cache()
    
    # Create visualization
    print(f"\n{'='*80}")
    print("Creating visualization...")
    print(f"{'='*80}")
    
    # Save visualization and combined results at the output_dir level
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_output_file = os.path.join(args.output_dir, f"per_head_heatmap_layer_{args.target_layer}.pdf")
    plot_per_head_heatmap(results_by_step, args.target_layer, args.output_dir)
    
    # Save combined results for this layer
    combined_file = os.path.join(args.output_dir, f"combined_results_layer_{args.target_layer}.json")
    with open(combined_file, 'w') as f:
        json.dump(results_by_step, f, indent=2)
    
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"Combined results saved to: {combined_file}")

if __name__ == "__main__":
    main()

