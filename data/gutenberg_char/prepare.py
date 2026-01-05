"""
Prepare character-level dataset from downloaded texts.

Reads all pg*.txt files, concatenates them, builds a character vocabulary,
encodes to integers, and saves as data.bin with meta.pkl for decoding.
"""

import os
import glob
import pickle
import numpy as np

# Get directory of this script
script_dir = os.path.dirname(__file__)

# Read all pg*.txt files into a single buffer
pattern = os.path.join(script_dir, 'pg*.txt')
files = sorted(glob.glob(pattern))
print(f"Found {len(files)} files: {[os.path.basename(f) for f in files]}")

data = ''
for filepath in files:
    with open(filepath, 'r') as f:
        data += f.read() + '\n'

print(f"Length of dataset in characters: {len(data):,}")

# Get all unique characters in the text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"All unique characters: {''.join(chars)}")
print(f"Vocab size: {vocab_size:,}")

# Create mappings from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    """Encode a string to a list of integers."""
    return [stoi[c] for c in s]


def decode(l):
    """Decode a list of integers to a string."""
    return ''.join([itos[i] for i in l])


# Encode to integers
data_ids = encode(data)
print(f"Dataset has {len(data_ids):,} tokens")

# Export to single bin file
data_ids = np.array(data_ids, dtype=np.uint16)
data_ids.tofile(os.path.join(script_dir, 'data.bin'))
print(f"Written to {os.path.join(script_dir, 'data.bin')}")

# Save meta information for encoding/decoding later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(script_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
print(f"Written to {os.path.join(script_dir, 'meta.pkl')}")
