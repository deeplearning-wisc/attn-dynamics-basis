import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import json
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='Train a residual transformer on TinyStories data with common words tokenizer')
parser.add_argument('--seq_length', type=int, default=200, help='Sequence length')
parser.add_argument('--per_device_batch_size', type=int, default=512, help='Batch size per device')
parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
parser.add_argument('--log_epochs', type=int, default=2, help='Save weights and plots every N epochs')
parser.add_argument('--layers', type=int, default=1, help='Number of transformer layers')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

args = parser.parse_args()

filtered_dataset = load_from_disk('filtered_data/filtered_tiny_stories')

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

with open('filtered_data/vocabulary.json', 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
    vocab = vocab_data['vocab']
    token_to_idx = vocab_data['token_to_idx']
    idx_to_token = vocab_data['idx_to_token']
    vocab_size = vocab_data['vocab_size']

original_vocab_size = len(vocab)
vocab = [token for token in vocab if token not in [' ', '\n']]
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
idx_to_token = {idx: token for idx, token in enumerate(vocab)}
vocab_size = len(vocab)

total_batch_size = args.per_device_batch_size * max(1, num_gpus_to_use)

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
for i, example in enumerate(filtered_dataset):
    tokens = tokenize_text(example['text'])
    if len(tokens) >= seq_length + 1:
        encoded = encode_text(example['text'], max_length=seq_length + 1)
        encoded_texts.append(encoded)

data_tensor = torch.tensor(encoded_texts, dtype=torch.long).to(device)
num_samples = len(data_tensor)
num_complete_batches = num_samples // total_batch_size
truncated_samples = num_complete_batches * total_batch_size
data_tensor = data_tensor[:truncated_samples]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

dataset = TinyStoriesDataset(data_tensor, seq_length)
dataloader = DataLoader(
    dataset, 
    batch_size=total_batch_size, 
    shuffle=True, 
    num_workers=0,
    worker_init_fn=seed_worker,
    generator=g
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

# Precompute B_bar, Phi_bar, and Q_bar matrices

batch_data = next(iter(dataloader))
x, y = batch_data
x = x.to(device)
y = y.to(device)

# Convert to one-hot
x_onehot = F.one_hot(x, num_classes=vocab_size).float().to(device)
y_onehot = F.one_hot(y, num_classes=vocab_size).float().to(device)

# Create r: y with 1/vocab_size subtracted from every element
r = y_onehot - (1.0 / vocab_size)

# Build matrix B_bar: sum over batch divided by batch_size*seq_length
B_bar = torch.zeros(vocab_size, vocab_size).to(device)
for b in range(x.shape[0]):
    B_bar += torch.matmul(x_onehot[b].T, r[b])
B_bar = B_bar / (x.shape[0] * seq_length)

# Create Phi_bar: sum over batch divided by number of tokens
attended_context = torch.matmul(attention_mask, x_onehot)
Phi_bar = torch.zeros(vocab_size, vocab_size).to(device)
for b in range(x.shape[0]):
    Phi_bar += torch.matmul(r[b].T, attended_context[b])
Phi_bar = Phi_bar / (x.shape[0] * seq_length)

# Compute derived matrices
B_bar_Phi_bar = torch.matmul(B_bar, Phi_bar).T
G_bar = torch.matmul(torch.matmul(B_bar, Phi_bar).T, B_bar)
Sigma_B = torch.matmul(B_bar.T, B_bar)

# Compute Q_bar
Q_bar = torch.zeros(vocab_size, vocab_size).to(device)
for b in range(x.shape[0]):
    Q_bar += torch.matmul(torch.matmul(x_onehot[b].T, torch.einsum('tjk,tk->tj', J, torch.matmul(torch.matmul(x_onehot[b], G_bar), r[b].T).T)), x_onehot[b])

config_name = f"layers{args.layers}_vocab{vocab_size}_lr{args.lr:.6f}_batch{total_batch_size}_seed{args.seed}"
base_output_folder = "results_tiny"
base_weights_folder = "weights_tiny"
output_folder = os.path.join(base_output_folder, config_name)
weights_folder = os.path.join(base_weights_folder, config_name)

os.makedirs(output_folder, exist_ok=True)
os.makedirs(weights_folder, exist_ok=True)

def save_top_elements(matrix, matrix_name, base_output_folder, vocab_size, idx_to_token, layer_idx=None):
    matrix_np = matrix.cpu().detach().numpy()
    top_elem_folder = os.path.join(base_output_folder, f"precomputed")
    os.makedirs(epoch_folder, exist_ok=True)
    
    if layer_idx is not None:
        layer_folder = os.path.join(top_elem_folder, f"layer_{layer_idx}")
        os.makedirs(layer_folder, exist_ok=True)
        output_file = os.path.join(layer_folder, f"{matrix_name}_top_elements.txt")
        title = f"Top 30 elements for {matrix_name} matrix, Layer {layer_idx}"
    else:
        output_file = os.path.join(top_elem_folder, f"{matrix_name}_top_elements.txt")
        title = f"Top 30 elements for {matrix_name} matrix"
    
    with open(output_file, 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * 50 + "\n\n")
        f.write("ROWS (Input tokens -> Output tokens):\n")
        f.write("-" * 30 + "\n")
        for i in range(vocab_size):
            row = matrix_np[i, :]
            top_indices = np.argsort(row)[-30:][::-1]
            top_values = row[top_indices]
            f.write(f"Row {i} ({idx_to_token[i]}):\n")
            for j, (idx, val) in enumerate(zip(top_indices, top_values)):
                f.write(f"  {j+1}. {idx_to_token[idx]}: {val:.6f}\n")
            f.write("\n")
        f.write("\n" + "=" * 50 + "\n\n")
        f.write("COLUMNS (Output tokens <- Input tokens):\n")
        f.write("-" * 30 + "\n")
        for j in range(vocab_size):
            col = matrix_np[:, j]
            top_indices = np.argsort(col)[-30:][::-1]
            top_values = col[top_indices]
            f.write(f"Column {j} ({idx_to_token[j]}):\n")
            for i, (idx, val) in enumerate(zip(top_indices, top_values)):
                f.write(f"  {i+1}. {idx_to_token[idx]}: {val:.6f}\n")
            f.write("\n")

save_top_elements(B_bar, "B_bar", output_folder, vocab_size, idx_to_token)
save_top_elements(Phi_bar, "Phi_bar", output_folder, vocab_size, idx_to_token)
save_top_elements(B_bar_Phi_bar, "B_bar_Phi_bar", output_folder, vocab_size, idx_to_token)
save_top_elements(G_bar, "G_bar", output_folder, vocab_size, idx_to_token)
save_top_elements(Q_bar, "Q_bar", output_folder, vocab_size, idx_to_token)
save_top_elements(Sigma_B, "Sigma_B", output_folder, vocab_size, idx_to_token)

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

def save_weights_and_plots(epoch, model, base_output_folder, weights_folder, vocab_size, idx_to_token, B_bar, Phi_bar, B_bar_Phi_bar, G_bar, Q_bar):
    actual_model = model.module if hasattr(model, 'module') else model
    num_layers = actual_model.num_layers
    
    epoch_folder = os.path.join(base_output_folder, f"epoch_{epoch}")
    os.makedirs(epoch_folder, exist_ok=True)
    
    for layer_idx in range(num_layers):
        layer_folder = os.path.join(epoch_folder, f"layer_{layer_idx}")
        os.makedirs(layer_folder, exist_ok=True)

    for layer_idx in range(num_layers):
        layer_weights = actual_model.get_layer_weights(layer_idx)
        save_top_elements(layer_weights['W'], "W", output_folder, vocab_size, idx_to_token, epoch, layer_idx)
        save_top_elements(layer_weights['V'], "V", output_folder, vocab_size, idx_to_token, epoch, layer_idx)
    
    save_top_elements(actual_model.W_O, "W_O", output_folder, vocab_size, idx_to_token, epoch)

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

    weights_filename = f"weights_epoch{epoch}.pt"
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
    return similarities

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

cosine_similarities = {
    'W_O_B_bar': [],
    'epochs': []
}

training_logs = []

actual_model = model.module if hasattr(model, 'module') else model
for layer_idx in range(actual_model.num_layers):
    cosine_similarities[f'layer_{layer_idx}_W_Q_bar'] = []
    cosine_similarities[f'layer_{layer_idx}_V_B_bar_Phi_bar'] = []

for epoch in range(args.epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y in dataloader:
        x_onehot = F.one_hot(batch_x, num_classes=vocab_size).float().to(device)
        y_hat = model(x_onehot)
        y_hat_flat = y_hat.view(-1, vocab_size)
        y_flat = batch_y.reshape(-1).to(device)
        loss = criterion(y_hat_flat, y_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    if (epoch + 1) % args.log_epochs == 0:
        log_entry = compute_training_log_entry(epoch + 1, avg_loss, model, B_bar, B_bar_Phi_bar, Q_bar)
        training_logs.append(log_entry)
        similarities = save_weights_and_plots(epoch + 1, model, output_folder, weights_folder, vocab_size, idx_to_token, B_bar, Phi_bar, B_bar_Phi_bar, G_bar, Q_bar)
        cosine_similarities['W_O_B_bar'].append(similarities['W_O_B_bar'])
        for layer_idx in range(actual_model.num_layers):
            cosine_similarities[f'layer_{layer_idx}_W_Q_bar'].append(similarities[f'layer_{layer_idx}_W_Q_bar'])
            cosine_similarities[f'layer_{layer_idx}_V_B_bar_Phi_bar'].append(similarities[f'layer_{layer_idx}_V_B_bar_Phi_bar'])
        cosine_similarities['epochs'].append(epoch + 1)

logs_file = os.path.join(output_folder, "training_logs.json")
with open(logs_file, 'w') as f:
    json.dump(training_logs, f, indent=2)
