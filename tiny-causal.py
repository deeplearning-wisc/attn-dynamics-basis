import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='Causal analysis by removing theoretical matrix projections')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--eval_batch_size', type=int, default=2048, help='Batch size for loss evaluation')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_reproducibility(seed=42):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

setup_reproducibility(args.seed)

# Load vocabulary
with open('filtered_data/vocabulary.json', 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    token_to_idx = vocab_data['token_to_idx']
    vocab_size = vocab_data['vocab_size']

vocab = [token for token in vocab if token not in [' ', '\n']]
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
vocab_size = len(vocab)

def tokenize_text(text):
    text = text.lower()
    tokens = []
    current_word = ""
    
    for char in text:
        if char.isalpha():
            current_word += char
        else:
            if current_word:
                tokens.append(current_word)
                current_word = ""
            tokens.append(char)
    
    if current_word:
        tokens.append(current_word)
    
    tokens = [token for token in tokens if token not in [' ', '\n']]
    return tokens

def encode_text(text, max_length=200):
    tokens = tokenize_text(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return [token_to_idx[token] for token in tokens]

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
        
        for layer_idx in range(num_layers):
            W = torch.nn.Parameter(torch.zeros(vocab_size, vocab_size))
            self.W_layers.append(W)
            
            P = torch.nn.Parameter(torch.zeros(length))
            self.P_layers.append(P)
            
            V = torch.nn.Parameter(torch.zeros(vocab_size, vocab_size))
            self.V_layers.append(V)
        
        self.W_O = torch.nn.Parameter(torch.zeros(vocab_size, vocab_size))
    
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

# Load checkpoint
checkpoint = torch.load(args.checkpoint, map_location=device)

num_layers = checkpoint['parameters']['num_layers']
seq_length = checkpoint['parameters']['seq_length']

# Create model and load weights
model = ResidualTransformer(vocab_size=vocab_size, length=seq_length, num_layers=num_layers).to(device)
model.W_O.data = checkpoint['W_O'].to(device)

for layer_idx in range(num_layers):
    model.W_layers[layer_idx].data = checkpoint[f'layer_{layer_idx}_W'].to(device)
    model.V_layers[layer_idx].data = checkpoint[f'layer_{layer_idx}_V'].to(device)
    model.P_layers[layer_idx].data = checkpoint[f'layer_{layer_idx}_P'].to(device)

model.eval()

# Load theoretical matrices from checkpoint
B_bar = checkpoint['B_bar'].to(device)
Phi_bar = checkpoint['Phi_bar'].to(device)
B_bar_Phi_bar = checkpoint['B_bar_Phi_bar'].to(device)
Q_bar = checkpoint['Q_bar'].to(device)

# Load dataset
filtered_dataset = load_from_disk('filtered_data/filtered_tiny_stories')

# Load training dataset for loss evaluation (same as training)
encoded_texts = []
for i, example in enumerate(filtered_dataset):
    tokens = tokenize_text(example['text'])
    if len(tokens) >= seq_length + 1:
        encoded = encode_text(example['text'], max_length=seq_length + 1)
        encoded_texts.append(encoded)

loss_data_tensor = torch.tensor(encoded_texts, dtype=torch.long)
num_samples = len(loss_data_tensor)
num_complete_batches = num_samples // args.eval_batch_size
truncated_samples = num_complete_batches * args.eval_batch_size
loss_data_tensor = loss_data_tensor[:truncated_samples]

loss_dataset = TinyStoriesDataset(loss_data_tensor, seq_length)
loss_dataloader = DataLoader(
    loss_dataset,
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=0
)

def compute_projection(W, T):
    # Frobenius inner product: <W, T> = trace(W^T @ T)
    inner_product = torch.trace(torch.matmul(W.T, T))
    norm_T_squared = torch.norm(T, p='fro') ** 2
    
    # Projection coefficient
    coeff = inner_product / (norm_T_squared + 1e-8)
    
    # Projected component
    projection = coeff * T
    return projection

# Compute projections for each weight
projections = {}

# W_O projection onto B_bar
projections['W_O_onto_B_bar'] = compute_projection(model.W_O, B_bar)
print(f"W_O projection norm: {torch.norm(projections['W_O_onto_B_bar'], p='fro'):.4f}")

for layer_idx in range(num_layers):
    layer_weights = model.get_layer_weights(layer_idx)
    
    # V projection onto B_bar_Phi_bar
    projections[f'layer_{layer_idx}_V_onto_B_bar_Phi_bar'] = compute_projection(
        layer_weights['V'], B_bar_Phi_bar
    )
    print(f"Layer {layer_idx} V projection norm: {torch.norm(projections[f'layer_{layer_idx}_V_onto_B_bar_Phi_bar'], p='fro'):.4f}")
    
    # W projection onto Q_bar (scaled)
    W_scaled = layer_weights['W'] * np.power(10, 10)
    projections[f'layer_{layer_idx}_W_onto_Q_bar'] = compute_projection(
        W_scaled, Q_bar
    ) / np.power(10, 10)
    print(f"Layer {layer_idx} W projection norm: {torch.norm(projections[f'layer_{layer_idx}_W_onto_Q_bar'], p='fro'):.4f}")

def compute_loss_on_dataset(model, dataloader, vocab_size, device):
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x_onehot = F.one_hot(batch_x, num_classes=vocab_size).float()
            
            y_hat = model(x_onehot)
            y_hat_flat = y_hat.view(-1, vocab_size)
            y_flat = batch_y.reshape(-1)
            
            loss = criterion(y_hat_flat, y_flat)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def run_inference_variants(model, projections, num_layers, loss_dataloader, vocab_size, device):
    results = {}
    
    # 1. Original model
    results['original'] = {
        'dataset_loss': compute_loss_on_dataset(model, loss_dataloader, vocab_size, device)
    }
    
    # 2. Remove W_O projection
    original_W_O = model.W_O.data.clone()
    model.W_O.data = original_W_O - projections['W_O_onto_B_bar']
    results['W_O_removed'] = {
        'dataset_loss': compute_loss_on_dataset(model, loss_dataloader, vocab_size, device)
    }
    
    model.W_O.data = original_W_O
    
    # 3. Remove V projections for each layer
    for layer_idx in range(num_layers):
        original_V = model.V_layers[layer_idx].data.clone()
        model.V_layers[layer_idx].data = original_V - projections[f'layer_{layer_idx}_V_onto_B_bar_Phi_bar']
        results[f'layer_{layer_idx}_V_removed'] = {
            'dataset_loss': compute_loss_on_dataset(model, loss_dataloader, vocab_size, device)
        }
        
        model.V_layers[layer_idx].data = original_V
    
    # 4. Remove W projections for each layer
    for layer_idx in range(num_layers):
        original_W = model.W_layers[layer_idx].data.clone()
        model.W_layers[layer_idx].data = original_W - projections[f'layer_{layer_idx}_W_onto_Q_bar']
        results[f'layer_{layer_idx}_W_removed'] = {
            'dataset_loss': compute_loss_on_dataset(model, loss_dataloader, vocab_size, device)
        }
        
        model.W_layers[layer_idx].data = original_W
    
    return results

dataset_loss_results = run_inference_variants(
    model, projections, num_layers,
    loss_dataloader=loss_dataloader,
    vocab_size=vocab_size,
    device=device
)

# Extract dataset losses
dataset_losses = {}
for variant_name in dataset_loss_results:
    if 'dataset_loss' in dataset_loss_results[variant_name]:
        dataset_losses[variant_name] = dataset_loss_results[variant_name]['dataset_loss']

print("\nDataset Loss (Cross-Entropy):")
for variant_name in dataset_losses:
    print(f"  {variant_name}: {dataset_losses[variant_name]:.6f}")
