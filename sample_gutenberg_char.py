"""
Sample from a trained GPT model on Gutenberg character-level dataset.

Usage:
    python sample_gutenberg_char.py "Hello world" 100
    python sample_gutenberg_char.py --input "Hello world" --max_new_tokens 100
    python sample_gutenberg_char.py --max_new_tokens 200  # Uses empty input
"""

import os
import argparse
import pickle

import torch

from GPT import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
out_dir = 'out_gutenberg_char'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
temperature = 1.0
top_k = None
seed = None  # Random seed each time by default

# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Sample from trained GPT model')
    parser.add_argument('input', nargs='?', default='', help='Input text to start generation')
    parser.add_argument('max_new_tokens', nargs='?', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--input', '-i', dest='input_flag', default=None, help='Input text (alternative)')
    parser.add_argument('--max_new_tokens', '-n', dest='max_tokens_flag', type=int, default=None, help='Max tokens (alternative)')
    parser.add_argument('--temperature', '-t', type=float, default=temperature, help='Sampling temperature')
    parser.add_argument('--top_k', '-k', type=int, default=top_k, help='Top-k sampling')
    parser.add_argument('--seed', '-s', type=int, default=seed, help='Random seed')
    parser.add_argument('--device', '-d', default=device, help='Device to use')
    args = parser.parse_args()
    
    # Handle both positional and flag arguments
    input_text = args.input_flag if args.input_flag is not None else args.input
    max_new_tokens = args.max_tokens_flag if args.max_tokens_flag is not None else args.max_new_tokens
    
    # Set seed (if specified)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.no_grad()
    
    # Load vocab from meta
    data_dir = os.path.join('data', 'gutenberg_char')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']  # string to int
    itos = meta['itos']  # int to string
    
    def encode(s):
        return [stoi[c] for c in s if c in stoi]
    
    def decode(tokens):
        return ''.join([itos[t] for t in tokens])
    
    # Load checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    
    # Create model
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.to(args.device)
    model.eval()
    
    # Encode input
    if input_text:
        input_ids = encode(input_text)
        if not input_ids:
            print(f"Warning: Input text '{input_text}' has no valid characters in vocab")
            input_ids = [0]  # Use first token as fallback
    else:
        # Start with a newline or first token
        input_ids = [stoi.get('\n', 0)]
    
    x = torch.tensor([input_ids], dtype=torch.long, device=args.device)  # (1, T)
    
    # Generate
    print(f"\nInput: '{input_text}'")
    print(f"Generating {max_new_tokens} tokens with temperature={args.temperature}, top_k={args.top_k}")
    print("-" * 50)
    
    with ctx:
        output = model.generate(x, max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    
    # Decode and print
    output_tokens = output[0].tolist()
    output_text = decode(output_tokens)
    
    print(output_text)
    print("-" * 50)
    print(f"Generated {len(output_tokens) - len(input_ids)} new tokens")


if __name__ == '__main__':
    main()

