import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import json
from datasets import load_from_disk
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='Train a residual transformer on TinyStories data with BPE tokenizer')
parser.add_argument('--seq_length', type=int, default=200, help='Sequence length')
parser.add_argument('--per_device_batch_size', type=int, default=128, help='Batch size per device')
parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
parser.add_argument('--log_epochs', type=float, default=0.1, help='Log metrics every N epochs')
parser.add_argument('--save_weights_epochs', type=int, default=1, help='Save weights every N epochs (default: 1 = only at integer epochs)')
parser.add_argument('--layers', type=int, default=1, help='Number of transformer layers')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--grad_accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps')

args = parser.parse_args()

raw_dataset = load_from_disk('processed_data/tiny_stories')

if torch.cuda.is_available():
    num_available_gpus = torch.cuda.device_count()
    num_gpus_to_use = min(args.num_gpus, num_available_gpus)
    device = torch.device('cuda')
    use_multi_gpu = num_gpus_to_use > 1
else:
    device = torch.device('cpu')
    num_gpus_to_use = 0
    use_multi_gpu = False

def setup_reproducibility(seed=42):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.utils.deterministic.fill_uninitialized_memory = True

setup_reproducibility(args.seed)

tokenizer = Tokenizer.from_file('processed_data/bpe_tokenizer.json')

with open('processed_data/vocabulary.json', 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    token_to_idx = vocab_data['token_to_idx']
    idx_to_token = vocab_data['idx_to_token']
    vocab_size = vocab_data['vocab_size']

total_batch_size = args.per_device_batch_size * max(1, num_gpus_to_use)

def encode_text(text, max_length=200):
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    return token_ids

class TinyStoriesDataset(Dataset):
    def __init__(self, data_tensor, seq_length):
        self.data_tensor = data_tensor
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, idx):
        sequence = self.data_tensor[idx]
        x = sequence[:self.seq_length]
        y = sequence[1:self.seq_length + 1]
        return x, y

class ResidualTransformer(torch.nn.Module):
    def __init__(self, vocab_size, length, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.length = length
        self.num_layers = num_layers
        
        self.W_layers = torch.nn.ParameterList()
        self.P_layers = torch.nn.ParameterList()
        self.V_layers = torch.nn.ParameterList()
        
        variance = 0 * (length ** -3) * (vocab_size ** -2)
        std = torch.sqrt(torch.tensor(variance))
        
        for layer_idx in range(num_layers):
            W = torch.nn.Parameter(torch.randn(vocab_size, vocab_size) * std)
            W.requires_grad = True
            self.W_layers.append(W)
            
            P = torch.nn.Parameter(torch.randn(length) * std)
            P.requires_grad = True
            self.P_layers.append(P)
            
            V = torch.nn.Parameter(torch.randn(vocab_size, vocab_size) * std)
            V.requires_grad = True
            self.V_layers.append(V)
        
        self.W_O = torch.nn.Parameter(torch.randn(vocab_size, vocab_size) * std)
        self.W_O.requires_grad = True
    
    def forward(self, x):
        batch_size, seq_length, vocab_size = x.shape
        
        mask = torch.triu(torch.ones(seq_length, seq_length)*(-torch.inf), diagonal=1).to(x.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        current_x = x
        for layer_idx in range(self.num_layers):
            pos = torch.zeros(seq_length, seq_length).to(x.device)
            for i, val in enumerate(self.P_layers[layer_idx]):
                pos.diagonal(offset=-i).copy_(val)
            pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
            
            attention_scores = torch.matmul(torch.matmul(current_x, self.W_layers[layer_idx]), current_x.transpose(-2, -1))
            A = F.softmax(attention_scores + mask + pos, dim=-1)
            
            attended = torch.matmul(A, current_x)
            current_x = torch.matmul(attended, self.V_layers[layer_idx]) + current_x
        
        return torch.matmul(current_x, self.W_O)
    
    def get_layer_weights(self, layer_idx):
        return {
            'W': self.W_layers[layer_idx],
            'P': self.P_layers[layer_idx],
            'V': self.V_layers[layer_idx]
        }

seq_length = args.seq_length
model = ResidualTransformer(vocab_size=vocab_size, length=seq_length, num_layers=args.layers).to(device)

if use_multi_gpu:
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus_to_use)))

encoded_texts = []
for i, example in enumerate(raw_dataset):
    encoded = encode_text(example['text'], max_length=seq_length + 1)
    if len(encoded) >= seq_length + 1:
        encoded_texts.append(encoded)

data_tensor = torch.tensor(encoded_texts, dtype=torch.long)
num_samples = len(data_tensor)
num_complete_batches = num_samples // (total_batch_size * args.grad_accumulation_steps)
truncated_samples = min(num_complete_batches, 50) * total_batch_size * args.grad_accumulation_steps
data_tensor = data_tensor[:truncated_samples]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

dataset_obj = TinyStoriesDataset(data_tensor, seq_length)
dataloader = DataLoader(
    dataset_obj, 
    batch_size=total_batch_size, 
    shuffle=True, 
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=True,  
    persistent_workers=True  
)

def create_uniform_attention_mask(seq_length):
    A = torch.zeros(seq_length, seq_length)
    for i in range(seq_length):
        for j in range(i + 1):
            A[i, j] = 1.0 / (i + 1)
    return A

attention_mask = create_uniform_attention_mask(seq_length).to(device)

J = torch.zeros(seq_length, seq_length, seq_length).to(device)
for t in range(seq_length):    
    for i in range(seq_length):
        for j in range(seq_length):
            if i == j:
                J[t, i, j] = 1/(t+1) * (1 - 1/(t+1))
            else:
                J[t, i, j] = -1/((t+1)*(t+1))
    J[t, t+1:, :] = 0.0
    J[t, :, t+1:] = 0.0
J = J.to(device)

# Use 4 batches for precomputation (total 2048 examples)
num_precompute_batches = 4
total_examples = 0

# Initialize accumulators
B_bar = torch.zeros(vocab_size, vocab_size).to(device)
Phi_bar = torch.zeros(vocab_size, vocab_size).to(device)

dataloader_iter = iter(dataloader)
for batch_idx in range(num_precompute_batches):
    batch_data = next(dataloader_iter)
    x, y = batch_data
    x = x.to(device)
    y = y.to(device)
    
    # Convert to one-hot
    x_onehot = F.one_hot(x, num_classes=vocab_size).float().to(device)
    y_onehot = F.one_hot(y, num_classes=vocab_size).float().to(device)
    
    # Create r: y with 1/vocab_size subtracted from every element
    r = y_onehot - (1.0 / vocab_size)
    
    # Accumulate B_bar
    for b in range(x.shape[0]):
        B_bar += torch.matmul(x_onehot[b].T, r[b])
    
    # Accumulate Phi_bar
    attended_context = torch.matmul(attention_mask, x_onehot)
    for b in range(x.shape[0]):
        Phi_bar += torch.matmul(r[b].T, attended_context[b])
    
    total_examples += x.shape[0]

# Normalize by total number of tokens
total_tokens = total_examples * seq_length
B_bar = B_bar / total_tokens
Phi_bar = Phi_bar / total_tokens

# Compute derived matrices
B_bar_Phi_bar = torch.matmul(B_bar, Phi_bar).T
G_bar = torch.matmul(torch.matmul(B_bar, Phi_bar).T, B_bar)
Sigma_B = torch.matmul(B_bar.T, B_bar)

Q_bar = torch.zeros(vocab_size, vocab_size).to(device)
dataloader_iter = iter(dataloader)
for batch_idx in range(num_precompute_batches):
    batch_data = next(dataloader_iter)
    x, y = batch_data
    x = x.to(device)
    y = y.to(device)
    
    x_onehot = F.one_hot(x, num_classes=vocab_size).float().to(device)
    y_onehot = F.one_hot(y, num_classes=vocab_size).float().to(device)
    r = y_onehot - (1.0 / vocab_size)
    
    for b in range(x.shape[0]):
        Q_bar += torch.matmul(torch.matmul(x_onehot[b].T, torch.einsum('tjk,tk->tj', J, torch.matmul(torch.matmul(x_onehot[b], G_bar), r[b].T).T)), x_onehot[b])

config_name = f"layers{args.layers}_vocab{vocab_size}_lr{args.lr:.6f}_batch{total_batch_size}_seed{args.seed}"
base_output_folder = "results_tiny_bpe"
base_weights_folder = "weights_tiny_bpe"
output_folder = os.path.join(base_output_folder, config_name)
weights_folder = os.path.join(base_weights_folder, config_name)

os.makedirs(output_folder, exist_ok=True)
os.makedirs(weights_folder, exist_ok=True)

def compute_training_log_entry(epoch, loss, model, B_bar, B_bar_Phi_bar, Q_bar):
    def cosine_similarity_matrix(A, B):
        trace_AT_B = torch.trace(torch.matmul(A.T, B))
        norm_A = torch.norm(A, p='fro')
        norm_B = torch.norm(B, p='fro')
        return trace_AT_B / (norm_A * norm_B)
    
    log_entry = {
        'epoch': epoch,
        'loss': loss,
        'cosine_similarities': {}
    }
    
    actual_model = model.module if hasattr(model, 'module') else model
    
    for layer_idx in range(actual_model.num_layers):
        layer_weights = actual_model.get_layer_weights(layer_idx)
        W_Q_bar_sim = cosine_similarity_matrix(layer_weights['W'].cpu().detach() * np.power(10, 10), Q_bar.cpu().detach())
        V_B_bar_Phi_bar_sim = cosine_similarity_matrix(layer_weights['V'].cpu().detach(), B_bar_Phi_bar.cpu().detach())
        log_entry['cosine_similarities'][f'layer_{layer_idx}_W_Q_bar'] = W_Q_bar_sim.item()
        log_entry['cosine_similarities'][f'layer_{layer_idx}_V_B_bar_Phi_bar'] = V_B_bar_Phi_bar_sim.item()
    
    W_O_B_bar_sim = cosine_similarity_matrix(actual_model.W_O.cpu().detach(), B_bar.cpu().detach())
    log_entry['cosine_similarities']['W_O_B_bar'] = W_O_B_bar_sim.item()
    
    print(f"Epoch {epoch} - Loss: {loss:.4f}")
    print(f"  Output Layer - W_O/B_bar: {W_O_B_bar_sim:.4f}")
    for layer_idx in range(actual_model.num_layers):
        W_sim = log_entry['cosine_similarities'][f'layer_{layer_idx}_W_Q_bar']
        V_sim = log_entry['cosine_similarities'][f'layer_{layer_idx}_V_B_bar_Phi_bar']
        print(f"  Layer {layer_idx} - W/Q_bar: {W_sim:.4f}, V/B_bar_Phi_bar: {V_sim:.4f}")
    
    return log_entry

def save_weights(epoch, model, weights_folder, vocab_size, B_bar, Phi_bar, B_bar_Phi_bar, G_bar, Q_bar):
    actual_model = model.module if hasattr(model, 'module') else model
    num_layers = actual_model.num_layers

    weights_filename = f"weights_epoch{epoch:.2f}.pt"
    weights_dict = {
        'W_O': actual_model.W_O.cpu().detach(),
        'B_bar': B_bar.cpu().detach(),
        'Phi_bar': Phi_bar.cpu().detach(),
        'B_bar_Phi_bar': B_bar_Phi_bar.cpu().detach(),
        'Q_bar': Q_bar.cpu().detach(),
        'Sigma_B': Sigma_B.cpu().detach(),
        'parameters': {
            'vocab_size': vocab_size,
            'seq_length': seq_length,
            'batch_size': total_batch_size,
            'num_layers': num_layers,
            'epoch': epoch
        }
    }
    
    for layer_idx in range(num_layers):
        layer_weights = actual_model.get_layer_weights(layer_idx)
        weights_dict[f'layer_{layer_idx}_W'] = layer_weights['W'].cpu().detach()
        weights_dict[f'layer_{layer_idx}_V'] = layer_weights['V'].cpu().detach()
        weights_dict[f'layer_{layer_idx}_P'] = layer_weights['P'].cpu().detach()
    
    torch.save(weights_dict, os.path.join(weights_folder, weights_filename))
    
    def cosine_similarity_matrix(A, B):
        trace_AT_B = torch.trace(torch.matmul(A.T, B))
        norm_A = torch.norm(A, p='fro')
        norm_B = torch.norm(B, p='fro')
        return trace_AT_B / (norm_A * norm_B)

    similarities = {}
    W_O_B_bar_sim = cosine_similarity_matrix(actual_model.W_O.cpu().detach(), B_bar.cpu().detach())
    similarities['W_O_B_bar'] = W_O_B_bar_sim.item()
    
    for layer_idx in range(num_layers):
        layer_weights = actual_model.get_layer_weights(layer_idx)
        W_Q_bar_sim = cosine_similarity_matrix(layer_weights['W'].cpu().detach() * np.power(10, 10), Q_bar.cpu().detach())
        V_B_bar_Phi_bar_sim = cosine_similarity_matrix(layer_weights['V'].cpu().detach(), B_bar_Phi_bar.cpu().detach())
        similarities[f'layer_{layer_idx}_W_Q_bar'] = W_Q_bar_sim.item()
        similarities[f'layer_{layer_idx}_V_B_bar_Phi_bar'] = V_B_bar_Phi_bar_sim.item()
    
    return similarities

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
optimizer.zero_grad()

cosine_similarities = {
    'W_O_B_bar': [],
    'epochs': []
}

training_logs = []

actual_model = model.module if hasattr(model, 'module') else model
for layer_idx in range(actual_model.num_layers):
    cosine_similarities[f'layer_{layer_idx}_W_Q_bar'] = []
    cosine_similarities[f'layer_{layer_idx}_V_B_bar_Phi_bar'] = []

batches_per_epoch = len(dataloader)
log_interval = args.log_epochs
accumulation_step = 0
last_logged_epoch = 0.0 

for epoch in range(args.epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        x_onehot = F.one_hot(batch_x, num_classes=vocab_size).float().to(device)
        y_hat = model(x_onehot)
        y_hat_flat = y_hat.view(-1, vocab_size)
        y_flat = batch_y.reshape(-1).to(device)
        loss = criterion(y_hat_flat, y_flat)
        
        # Scale loss by accumulation steps
        loss = loss / args.grad_accumulation_steps
        loss.backward()
        
        accumulation_step += 1
        
        # Update weights only every grad_accumulation_steps
        if accumulation_step % args.grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * args.grad_accumulation_steps
        num_batches += 1
        
        # Calculate current epoch
        current_epoch = epoch + (num_batches / batches_per_epoch)
        
        # Calculate next logging target based on last logged epoch
        next_log_target = last_logged_epoch + log_interval
        
        # Log if we've reached or passed the next logging target and did an optimizer step
        if current_epoch >= next_log_target and accumulation_step % args.grad_accumulation_steps == 0:
            avg_loss = epoch_loss / num_batches
            progress_fraction = num_batches / batches_per_epoch
            print(f"Epoch {next_log_target:.1f}, Average Loss: {avg_loss:.4f}, Progress: {num_batches}/{batches_per_epoch} ({progress_fraction:.4f}), Accum step: {accumulation_step}")
            
            log_entry = compute_training_log_entry(next_log_target, avg_loss, model, B_bar, B_bar_Phi_bar, Q_bar)
            training_logs.append(log_entry)
            
            # Only save weights at integer epochs
            is_integer_epoch = abs(next_log_target - round(next_log_target)) < 1e-6
            should_save_weights = is_integer_epoch and (round(next_log_target) % args.save_weights_epochs == 0)
            
            if should_save_weights:
                similarities = save_weights(next_log_target, model, weights_folder, vocab_size, B_bar, Phi_bar, B_bar_Phi_bar, G_bar, Q_bar)
            else:
                similarities = log_entry['cosine_similarities']
            
            cosine_similarities['W_O_B_bar'].append(similarities['W_O_B_bar'])
            for layer_idx in range(actual_model.num_layers):
                cosine_similarities[f'layer_{layer_idx}_W_Q_bar'].append(similarities[f'layer_{layer_idx}_W_Q_bar'])
                cosine_similarities[f'layer_{layer_idx}_V_B_bar_Phi_bar'].append(similarities[f'layer_{layer_idx}_V_B_bar_Phi_bar'])
            cosine_similarities['epochs'].append(next_log_target)
            
            # Save logs incrementally
            logs_file = os.path.join(output_folder, "training_logs.json")
            with open(logs_file, 'w') as f:
                json.dump(training_logs, f, indent=2)
            
            last_logged_epoch = next_log_target
    
    # Handle any remaining accumulated gradients at end of epoch
    if accumulation_step % args.grad_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Epoch summary
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch + 1} complete, Average Loss: {avg_loss:.4f}")

print("Training complete! Logs saved to:", os.path.join(output_folder, "training_logs.json"))

