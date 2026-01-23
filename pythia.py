import os
import torch
import numpy as np
import subprocess
import sys
from transformer_lens import HookedTransformer
import argparse
import json
import random
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

def load_fineweb_subset(n_examples=100000, max_length=128, seed=42):
    print(f"Loading {n_examples} examples from FineWeb...")
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT",
                          split="train", streaming=True)
    
    # Collect examples
    examples = []
    for example in tqdm(dataset, desc="Loading examples", total=n_examples):
        if len(examples) >= n_examples: 
            break
        
        text = example['text']
        if len(text) >= 4*max_length: # estimate tokens based on avg of 4 chars 
            examples.append(text)
    
    print(f"Loaded {len(examples)} examples from FineWeb")
    return examples

def find_top_tokens(model, examples, top_k=5000, device=None):
    if device is None:
        device = model.cfg.device
    
    print(f"Finding top {top_k} most common tokens from {len(examples)} examples...")
    
    token_counts = {}
    
    # Count tokens across all examples
    for example in tqdm(examples, desc="Counting tokens"):
        tokens = model.to_tokens(example, truncate=True)
        for token_id in tokens[0].cpu().numpy():
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
    
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    top_tokens = [token_id for token_id, count in sorted_tokens[:top_k]]
    
    print(f"Found top {len(top_tokens)} tokens (total unique tokens: {len(token_counts)})")
    return top_tokens

def compute_layer_outputs_for_tokens(model, token_ids, device=None):
    if device is None:
        device = model.cfg.device
    
    print(f"Computing layer outputs for {len(token_ids)} tokens...")
    
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    layer_outputs = {}
    layer_mids = {}
    layer_inputs = {}
    for layer_idx in range(n_layers):
        layer_outputs[f'layer_{layer_idx}'] = torch.zeros(len(token_ids), d_model, device=device)
        layer_mids[f'layer_{layer_idx}'] = torch.zeros(len(token_ids), d_model, device=device)
        layer_inputs[f'layer_{layer_idx}'] = torch.zeros(len(token_ids), d_model, device=device)
    
    batch_size = 64 
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(token_ids), batch_size), desc="Processing token batches"):
            batch_end = min(batch_start + batch_size, len(token_ids))
            batch_tokens = token_ids[batch_start:batch_end]
            
            batch_tensor = torch.tensor(batch_tokens, device=device).unsqueeze(1)  # [batch_size, 1]
            
            logits, cache = model.run_with_cache(batch_tensor)
            
            for layer_idx in range(n_layers):
                layer_output = cache['resid_post', layer_idx]  # [batch_size, 1, d_model]
                layer_input = cache['resid_pre', layer_idx]
                attn_out = cache['attn_out', layer_idx]  # [batch_size, 1, d_model]
                layer_mid = layer_input + attn_out
                
                layer_outputs[f'layer_{layer_idx}'][batch_start:batch_end] = layer_output[:, 0, :]
                layer_mids[f'layer_{layer_idx}'][batch_start:batch_end] = layer_mid[:, 0, :]
                layer_inputs[f'layer_{layer_idx}'][batch_start:batch_end] = layer_input[:, 0, :]
    print(f"Computed layer outputs for {len(token_ids)} tokens across {n_layers} layers")
    return layer_inputs, layer_mids, layer_outputs

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

def compute_attention_comparison(model, layer_inputs, theoretical_matrix, token_ids, device=None, output_dir="results"):
    if device is None:
        device = next(iter(layer_inputs.values())).device
    
    print(f"Computing attention comparison for {len(token_ids)} tokens...")
    
    n_tokens = len(token_ids)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_heads = model.cfg.n_heads
    d_head = d_model // n_heads
    
    attention_results = {}
    
    batch_size = 64
    actual_attn_scores = {}
    
    print("Extracting actual key*query parameter matrices...")
    
    actual_key_query_matrices = {}
    
    with torch.no_grad():
        for layer_idx in range(n_layers):
            layer_name = f'layer_{layer_idx}'
            
            W_K = model.blocks[layer_idx].attn.W_K  # [n_heads, d_model, d_head]
            W_Q = model.blocks[layer_idx].attn.W_Q  # [n_heads, d_model, d_head]
            M_per_head = torch.einsum("hmd,hnd->hmn", W_Q, W_K)  # (n_heads, d_model, d_model)

            key_query_matrix = M_per_head.mean(dim=0)  # [d_model, d_model]
            
            actual_key_query_matrices[layer_name] = key_query_matrix
                
    print("Comparing actual key*query matrices with theoretical Q_bar transformed to d_model x d_model...")
    
    # Precompute theoretical covariance once
    row_norms = torch.norm(theoretical_matrix, p=2, dim=1, keepdim=True)
    
    # Only normalize rows with non-zero norms
    theoretical_matrix_scaled = torch.zeros_like(theoretical_matrix)
    non_zero_mask = (row_norms.squeeze() != 0)
    theoretical_matrix_scaled[non_zero_mask] = theoretical_matrix[non_zero_mask] / row_norms[non_zero_mask]
    
    print("  Computing theoretical covariance once...")
    theoretical_cov = compute_covariance_batched(theoretical_matrix_scaled, verbose=True)
    
    # Normalize to prevent numerical issues
    theoretical_cov_norm = torch.norm(theoretical_cov, p='fro')
    theoretical_cov = theoretical_cov / theoretical_cov_norm
    
    for layer_idx in range(n_layers):
        layer_name = f'layer_{layer_idx}'
        print(f"\n  Processing {layer_name}...")
        
        actual_matrix = actual_key_query_matrices[layer_name]  # [d_model, d_model]
        prev_layer_outputs = layer_inputs[layer_name]  # [n_tokens, d_model]
    
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
        
        attention_results[layer_name] = {
            'cosine_similarity': float(cosine_similarity.item())
        }
        
        print(f"  {layer_name}: cosine_similarity={cosine_similarity:.4f}")
        
        # Clean up to free memory
        del actual_matrix, actual_cov
        torch.cuda.empty_cache()
    
    os.makedirs(output_dir, exist_ok=True)
    
    attention_file = os.path.join(output_dir, "attention_comparison_results.pt")
    torch.save(attention_results, attention_file)
    
    summary_stats = {}
    for layer_name, results in attention_results.items():
        summary_stats[layer_name] = {
            'cosine_similarity': results['cosine_similarity']
        }
    
    summary_file = os.path.join(output_dir, "attention_comparison_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\nAttention comparison results saved to: {attention_file}")
    print(f"Summary statistics saved to: {summary_file}")
    
    layer_similarities = [(layer_name, results['cosine_similarity']) 
                         for layer_name, results in summary_stats.items()]
    layer_similarities.sort(key=lambda x: x[1], reverse=True) 
    
    print(f"  Best matching layer: {layer_similarities[0][0]} (cosine similarity: {layer_similarities[0][1]:.4f})")
    print(f"  Worst matching layer: {layer_similarities[-1][0]} (cosine similarity: {layer_similarities[-1][1]:.4f})")
    
    return attention_results

def compute_frobenius_cosine_similarity(actual_layer_outputs, theoretical_matrix, token_ids, output_dir="results"):
    n_layers = len(actual_layer_outputs)
    n_tokens = len(token_ids)
    
    similarity_results = {}
    
    # Precompute theoretical covariance once
    row_norms = torch.norm(theoretical_matrix, p=2, dim=1, keepdim=True)
    
    # Only normalize rows with non-zero norms
    B_scaled = torch.zeros_like(theoretical_matrix)
    non_zero_mask = (row_norms.squeeze() != 0)
    B_scaled[non_zero_mask] = theoretical_matrix[non_zero_mask] / row_norms[non_zero_mask]
    
    print("  Computing theoretical covariance once...")
    cov_B = compute_covariance_batched(B_scaled, verbose=True)  # [n_tokens, n_tokens]
    
    # Normalize to prevent numerical issues
    cov_B_norm = torch.norm(cov_B, p='fro')
    cov_B = cov_B / cov_B_norm
    
    for layer_idx in range(n_layers):
        layer_name = f'layer_{layer_idx}'
        print(f"\n  Processing {layer_name}...")
        
        A = actual_layer_outputs[layer_name]  # [n_tokens, d_model]

        cov_A = compute_covariance_batched(A, verbose=True)  # [n_tokens, n_tokens]
        
        # Normalize to prevent numerical issues
        cov_A_norm = torch.norm(cov_A, p='fro')
        cov_A = cov_A / cov_A_norm
        
        plot_covariance_heatmaps(cov_A, cov_B, layer_name, output_dir, size=10, shift=0)
        
        dot_product = torch.trace(torch.matmul(cov_A.T, cov_B))
        norm_A = torch.norm(cov_A, p='fro')
        norm_B = torch.norm(cov_B, p='fro')
        
        cosine_similarity = dot_product / (norm_A * norm_B)
        
        similarity_results[layer_name] = {
            'cosine_similarity': float(cosine_similarity.item()),
            'dot_product': float(dot_product.item())
        }
        
        print(f"  {layer_name}: cosine_similarity={cosine_similarity:.4f}")
        
        del cov_A
        torch.cuda.empty_cache()
    
    os.makedirs(output_dir, exist_ok=True)
    
    similarity_file = os.path.join(output_dir, "cosine_similarity_results.pt")
    torch.save(similarity_results, similarity_file)
    
    summary_stats = {}
    for layer_name, results in similarity_results.items():
        summary_stats[layer_name] = {
            'cosine_similarity': results['cosine_similarity']
        }
    
    summary_file = os.path.join(output_dir, "cosine_similarity_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\nCosine similarity results saved to: {similarity_file}")
    print(f"Summary statistics saved to: {summary_file}")
    
    layer_similarities = [(layer_name, results['cosine_similarity']) 
                         for layer_name, results in summary_stats.items()]
    layer_similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  Best matching layer: {layer_similarities[0][0]} (cosine similarity: {layer_similarities[0][1]:.4f})")
    print(f"  Worst matching layer: {layer_similarities[-1][0]} (cosine similarity: {layer_similarities[-1][1]:.4f})")
    
    return similarity_results

def compute_frobenius_cosine_similarity_mids(actual_layer_mids, theoretical_matrix, token_ids, output_dir="results"):
    n_layers = len(actual_layer_mids)
    n_tokens = len(token_ids)
    
    similarity_results = {}
    
    # Precompute theoretical covariance once
    row_norms = torch.norm(theoretical_matrix, p=2, dim=1, keepdim=True)
    
    # Only normalize rows with non-zero norms
    B_scaled = torch.zeros_like(theoretical_matrix)
    non_zero_mask = (row_norms.squeeze() != 0)
    B_scaled[non_zero_mask] = theoretical_matrix[non_zero_mask] / row_norms[non_zero_mask]
    
    print("  Computing theoretical covariance once...")
    cov_B = compute_covariance_batched(B_scaled, verbose=True)  # [n_tokens, n_tokens]
    
    # Normalize to prevent numerical issues
    cov_B_norm = torch.norm(cov_B, p='fro')
    cov_B = cov_B / cov_B_norm
    
    for layer_idx in range(n_layers):
        layer_name = f'layer_{layer_idx}'
        print(f"\n  Processing {layer_name}...")
        
        A = actual_layer_mids[layer_name]  # [n_tokens, d_model]

        cov_A = compute_covariance_batched(A, verbose=True)  # [n_tokens, n_tokens]
        
        # Normalize to prevent numerical issues
        cov_A_norm = torch.norm(cov_A, p='fro')
        cov_A = cov_A / cov_A_norm
        
        plot_covariance_heatmaps(cov_A, cov_B, layer_name, output_dir, size=10, shift=0, label_suffix="_mids")
        
        dot_product = torch.trace(torch.matmul(cov_A.T, cov_B))
        norm_A = torch.norm(cov_A, p='fro')
        norm_B = torch.norm(cov_B, p='fro')
        
        cosine_similarity = dot_product / (norm_A * norm_B)
        
        similarity_results[layer_name] = {
            'cosine_similarity': float(cosine_similarity.item()),
            'dot_product': float(dot_product.item())
        }
        
        print(f"  {layer_name}: cosine_similarity={cosine_similarity:.4f}")
        
        del cov_A
        torch.cuda.empty_cache()
    
    os.makedirs(output_dir, exist_ok=True)
    
    similarity_file = os.path.join(output_dir, "cosine_similarity_results_mids.pt")
    torch.save(similarity_results, similarity_file)
    
    summary_stats = {}
    for layer_name, results in similarity_results.items():
        summary_stats[layer_name] = {
            'cosine_similarity': results['cosine_similarity']
        }
    
    summary_file = os.path.join(output_dir, "cosine_similarity_summary_mids.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\nCosine similarity results (mids) saved to: {similarity_file}")
    print(f"Summary statistics (mids) saved to: {summary_file}")
    
    layer_similarities = [(layer_name, results['cosine_similarity']) 
                         for layer_name, results in summary_stats.items()]
    layer_similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  Best matching layer: {layer_similarities[0][0]} (cosine similarity: {layer_similarities[0][1]:.4f})")
    print(f"  Worst matching layer: {layer_similarities[-1][0]} (cosine similarity: {layer_similarities[-1][1]:.4f})")
    
    return similarity_results

def compute_theoretical_matrices(model, examples, token_indices, max_seq_len=512, batch_size=8, device=None, output_dir="results"):

    if device is None:
        device = model.cfg.device
    
    print(f"Computing theoretical matrices from {len(examples)} examples...")
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
    failed_examples = 0
    
    for batch_start in tqdm(range(0, len(examples), batch_size), desc="Computing theoretical matrices"):
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
        
        token_to_idx = {token_id: idx for idx, token_id in enumerate(token_indices)}
        
        token_indices_tensor = torch.tensor(token_indices, device=device)  # [len(token_indices)]
        
        x_onehot = torch.zeros(len(batch_tokens), max_seq_len, len(token_indices), device=device, dtype=torch.float)
        
        token_mask = (truncated_tokens.unsqueeze(-1) == token_indices_tensor.unsqueeze(0).unsqueeze(0))
        x_onehot = token_mask.float()
        
        y_tokens = torch.zeros_like(truncated_tokens)
        y_tokens[:, :-1] = truncated_tokens[:, 1:]  # Shift by 1
        
        y_onehot = torch.zeros(len(batch_tokens), max_seq_len, len(token_indices), device=device, dtype=torch.float)
        y_token_mask = (y_tokens.unsqueeze(-1) == token_indices_tensor.unsqueeze(0).unsqueeze(0))
        y_onehot = y_token_mask.float()
        
        r = y_onehot - (1.0 / vocab_size)
        
        for batch_idx, example_idx in enumerate(valid_indices):
            seq_len = max_seq_len
            
            x_seq = x_onehot[batch_idx, :seq_len, :]  # [seq_len, len(token_indices)]
            r_seq = r[batch_idx, :seq_len, :]  # [seq_len, len(token_indices)]
            
            B_bar += torch.matmul(r_seq.T, x_seq)  # (len(token_indices), len(token_indices))
            
            attended_context = torch.matmul(attention_mask[:seq_len, :seq_len], x_seq)  # [seq_len, len(token_indices)]
            
            Phi_bar += torch.matmul(r_seq.T, attended_context)  # (len(token_indices), len(token_indices))
            
            total_tokens += seq_len
            processed_examples += 1
    
    if total_tokens > 0:
        B_bar = B_bar / total_tokens
        Phi_bar = Phi_bar / total_tokens
    
    print("Computing derived matrices...")
    
    B_bar_Phi_bar = torch.matmul(B_bar.T, Phi_bar).T
    
    G_bar = torch.matmul(B_bar_Phi_bar.T, B_bar.T)
    
    t_coords = torch.arange(seq_length, device=device).view(-1, 1, 1)  # [seq_length, 1, 1]
    i_coords = torch.arange(seq_length, device=device).view(1, -1, 1)  # [1, seq_length, 1]
    j_coords = torch.arange(seq_length, device=device).view(1, 1, -1)  # [1, 1, seq_length]
    
    diagonal_mask = (i_coords == j_coords).float()  # [1, seq_length, seq_length]
    off_diagonal_mask = (i_coords != j_coords).float()  # [1, seq_length, seq_length]
    
    valid_mask = ((i_coords <= t_coords) & (j_coords <= t_coords)).float()  # [seq_length, seq_length, seq_length]
    
    diagonal_terms = (1.0 / (t_coords + 1)) * (1.0 - 1.0 / (t_coords + 1))  # [seq_length, 1, 1]
    
    off_diagonal_terms = -1.0 / ((t_coords + 1) * (t_coords + 1))  # [seq_length, 1, 1]
    
    J = (diagonal_mask * diagonal_terms + off_diagonal_mask * off_diagonal_terms) * valid_mask

    B_bar_Phi_bar = torch.matmul(B_bar.T, Phi_bar).T
    G_bar = torch.matmul(torch.matmul(B_bar.T, Phi_bar).T, B_bar.T)

    Q_bar = torch.zeros(reduced_vocab_size, reduced_vocab_size, device=device)
    
    for batch_start in tqdm(range(0, len(examples), batch_size), desc="Computing Q_bar"):
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
        
        token_indices_tensor = torch.tensor(token_indices, device=device)  # [len(token_indices)]
        
        token_mask = (truncated_tokens.unsqueeze(-1) == token_indices_tensor.unsqueeze(0).unsqueeze(0))
        x_onehot = token_mask.float()
        
        y_tokens = torch.zeros_like(truncated_tokens)
        y_tokens[:, :-1] = truncated_tokens[:, 1:]  # Shift by 1
        
        y_token_mask = (y_tokens.unsqueeze(-1) == token_indices_tensor.unsqueeze(0).unsqueeze(0))
        y_onehot = y_token_mask.float()
        
        r = y_onehot - (1.0 / vocab_size)
        
        for batch_idx in range(len(batch_tokens)):
            seq_len = max_seq_len
            x_seq = x_onehot[batch_idx, :seq_len, :]  # [seq_len, len(token_indices)]
            r_seq = r[batch_idx, :seq_len, :]  # [seq_len, len(token_indices)]
            
            intermediate = torch.matmul(x_seq, G_bar)  # [seq_len, len(token_indices)]
            intermediate = torch.matmul(intermediate, r_seq.T)  # [seq_len, seq_len]
            
            jacobian_applied = torch.einsum('tjk,tk->tj', J[:seq_len, :seq_len, :seq_len], intermediate.T).T  # [seq_len, seq_len]
            
            Q_bar += torch.matmul(torch.matmul(x_seq.T, jacobian_applied), x_seq)

    theoretical_matrices = {
        'B_bar': B_bar,
        'Phi_bar': Phi_bar,
        'B_bar_Phi_bar': B_bar_Phi_bar,
        'G_bar': G_bar,
        'Q_bar': Q_bar,
        'metadata': {
            'n_examples': processed_examples,
            'total_tokens': total_tokens,
            'vocab_size': vocab_size,
            'reduced_vocab_size': len(token_indices),
            'max_seq_len': max_seq_len,
            'batch_size': batch_size
        }
    }

    return theoretical_matrices

def main():
    parser = argparse.ArgumentParser(description='Load and setup Pythia 1B deduped model')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-1.4b-deduped', 
                       help='Name of the Pythia model to load')
    parser.add_argument('--revision', type=str, default="step0",
                       help='Model revision to load (e.g., step0, step1, step1000, step10000)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to load model on (auto-detect if None)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save model information')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['openwebtext', 'fineweb'],
        default='openwebtext',
        help='Dataset to analyze',
    )
    parser.add_argument('--n_examples', type=int, default=100000,
                       help='Number of examples to analyze from the dataset')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length for analysis')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for processing examples')
    parser.add_argument('--top_tokens', type=int, default=None,
                       help='Number of top tokens to analyze (default: use all tokens in vocab)')
    
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = 'results_fw' if args.dataset == 'fineweb' else 'results'
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Setting up Pythia 1B deduped model...")
    print(f"Target device: {device}")
    
    # Modify output directory to include revision
    if args.revision:
        args.output_dir = os.path.join(args.output_dir, f"revision_{args.revision}")
    
    # Load model
    model = load_pythia_model(args.model_name, device, revision=args.revision)
    
    # Set top_tokens to vocab_size if not specified
    if args.top_tokens is None:
        args.top_tokens = model.cfg.d_vocab
        print(f"Using all tokens in vocabulary: {args.top_tokens} tokens")
    elif args.top_tokens > model.cfg.d_vocab:
        print(f"Warning: top_tokens ({args.top_tokens}) exceeds vocab_size ({model.cfg.d_vocab}), using vocab_size")
        args.top_tokens = model.cfg.d_vocab

    dataset_label = "FineWeb" if args.dataset == "fineweb" else "OpenWebText"
    print(f"Number of examples: {args.n_examples}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Batch size: {args.batch_size}")
    
    # Load dataset subset
    if args.dataset == "fineweb":
        examples = load_fineweb_subset(
            n_examples=args.n_examples,
            max_length=args.max_seq_len,
        )
    else:
        examples = load_openwebtext_subset(
            n_examples=args.n_examples,
            max_length=args.max_seq_len,
        )
    
    # Find top tokens or use all tokens if top_tokens equals vocab size
    if args.top_tokens == model.cfg.d_vocab:
        print(f"Using all {model.cfg.d_vocab} tokens from vocabulary (skipping token counting)")
        top_tokens = list(range(model.cfg.d_vocab))
    else:
        print(f"Finding top {args.top_tokens} most common tokens...")
        top_tokens = find_top_tokens(
            model=model,
            examples=examples,
            top_k=args.top_tokens,
            device=device
        )

    print(f"\nComputing theoretical matrices from {dataset_label}...")
    
    # Compute theoretical matrices
    theoretical_matrices = compute_theoretical_matrices(
        model=model,
        examples=examples,
        token_indices=top_tokens,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir
    )
        
    print(f"Theoretical matrices computation completed and saved to {args.output_dir}/")
    
    print(f"\nComputing layer outputs for top {args.top_tokens} tokens...")
    
    # Compute layer outputs for these tokens
    layer_inputs, layer_mids, layer_outputs = compute_layer_outputs_for_tokens(
        model=model,
        token_ids=top_tokens,
        device=device
    )
    
    matrix_output_dir = os.path.join(args.output_dir, "matrix_theoretical")
    
    # Compute attention comparison between actual and theoretical attention patterns
    print(f"\nComputing attention comparison...")
    attention_results = compute_attention_comparison(
        model=model,
        layer_inputs=layer_inputs,
        theoretical_matrix=theoretical_matrices['Q_bar'],
        token_ids=top_tokens,
        device=device,
        output_dir=matrix_output_dir
    )
    
    # Compute Frobenius cosine similarity between actual and theoretical layer outputs
    similarity_results = compute_frobenius_cosine_similarity(
        actual_layer_outputs=layer_outputs,
        theoretical_matrix=theoretical_matrices['B_bar_Phi_bar'],
        token_ids=top_tokens,
        output_dir=matrix_output_dir
    )
    
    # Compute Frobenius cosine similarity between actual and theoretical layer mids
    similarity_results_mids = compute_frobenius_cosine_similarity_mids(
        actual_layer_mids=layer_mids,
        theoretical_matrix=theoretical_matrices['B_bar_Phi_bar'],
        token_ids=top_tokens,
        output_dir=matrix_output_dir
    )
         
    print(f"Layer outputs computation completed")
    
    print(f"Model comparison completed and saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
