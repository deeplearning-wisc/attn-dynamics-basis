from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")

from collections import Counter
import re

def tokenize_text(text):
    """
    Tokenize text into words and individual characters.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Find all words and individual characters
    tokens = []
    current_word = ""
    
    for char in text:
        if char.isalpha():
            current_word += char
        else:
            # Add current word
            if current_word:
                tokens.append(current_word)
                current_word = ""
            # Add the non-letter character as a token
            tokens.append(char)
    
    # Last word
    if current_word:
        tokens.append(current_word)
    
    return tokens

# Collect frequencies from the train split
word_freq = Counter()

for i, example in enumerate(dataset['train']):
    text = example['text']
    tokens = tokenize_text(text)
    word_freq.update(tokens)

total_tokens = sum(word_freq.values())

# Create vocabulary (most common tokens)
vocab_size = 3002
vocab = [token for token, _ in word_freq.most_common(vocab_size)]
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
idx_to_token = {idx: token for idx, token in enumerate(vocab)}

# Filter dataset to only include examples with tokens in vocabulary
filtered_examples = []

for i, example in enumerate(dataset['train']):
    text = example['text']
    tokens = tokenize_text(text)
    
    # Check if all tokens are in vocabulary
    all_in_vocab = all(token in token_to_idx for token in tokens)
    
    if all_in_vocab:
        filtered_examples.append(example)

from datasets import Dataset
import json
import os

# Create output directory if it doesn't exist
os.makedirs('filtered_data', exist_ok=True)

# Convert filtered examples to Dataset format
filtered_dataset = Dataset.from_list(filtered_examples)

# Save the dataset
filtered_dataset.save_to_disk('filtered_data/filtered_tiny_stories')

# Save vocabulary and mappings
vocab_data = {
    'vocab': vocab,
    'token_to_idx': token_to_idx,
    'idx_to_token': idx_to_token,
    'vocab_size': len(vocab),
    'total_tokens': total_tokens,
    'vocabulary_coverage': sum(word_freq[token] for token in vocab) / total_tokens * 100
}

with open('filtered_data/vocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)

# Save frequency data for analysis
freq_data = {token: freq for token, freq in word_freq.most_common()}
with open('filtered_data/token_frequencies.json', 'w', encoding='utf-8') as f:
    json.dump(freq_data, f, ensure_ascii=False, indent=2)
