from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from collections import Counter
import json
import os

# Load the dataset
dataset = load_dataset("roneneldan/TinyStories")

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Prepare training data for BPE
training_texts = []
for i, example in enumerate(dataset['train']):
    training_texts.append(example['text'])

# Train the BPE tokenizer
trainer = trainers.BpeTrainer(
    vocab_size=10000,
    special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
)

tokenizer.train_from_iterator(training_texts, trainer=trainer)

# Add post-processor for special tokens
start_token_id = tokenizer.token_to_id("<s>")
end_token_id = tokenizer.token_to_id("</s>")

tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[
        ("<s>", start_token_id),
        ("</s>", end_token_id),
    ]
)

def tokenize_text(text):
    encoding = tokenizer.encode(text)
    return encoding.tokens

# Get vocabulary from the trained tokenizer
vocab_size = tokenizer.get_vocab_size()
vocab = list(tokenizer.get_vocab().keys())
token_to_idx = tokenizer.get_vocab()
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

from datasets import Dataset

# Create output directory if it doesn't exist
os.makedirs('processed_data', exist_ok=True)

# Convert dataset to Dataset format and save
processed_dataset = Dataset.from_list(list(dataset['train']))
processed_dataset.save_to_disk('processed_data/tiny_stories')

# Save the BPE tokenizer
tokenizer.save('processed_data/bpe_tokenizer.json')

# Save vocabulary and mappings
vocab_data = {
    'vocab': vocab,
    'token_to_idx': token_to_idx,
    'idx_to_token': idx_to_token,
    'vocab_size': vocab_size,
    'tokenizer_type': 'BPE'
}

with open('processed_data/vocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)

