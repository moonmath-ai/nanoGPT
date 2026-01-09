"""
Sample from a trained GPTM model on Gutenberg character-level dataset.

GPTM operates in the teacher's hidden state space, so we need:
1. Teacher model - to get initial hidden states from input text
2. GPTM model - to predict state dynamics and next token

The generation loop:
1. Run teacher on input to get initial hidden state s_x
2. For each step:
   - GPTM([s_x, embed(last_token)]) → [s_y, token_logits]
   - Sample next token from logits
   - Use s_y as input for next step

Usage:
    python sample_gutenberg_char_gptm.py --ckpt out_gutenberg_char/ckpt_GPTM_GPT2X111111_in512_q32.pt "Once upon" 200
"""

import os
import argparse
import pickle
import importlib

import torch
from torch.nn import functional as F

from GPTM import GPTConfig as GPTMConfig, GPT as GPTM

# -----------------------------------------------------------------------------
# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
temperature = 1.0
top_k = None
seed = None

# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Sample from trained GPTM model')
    parser.add_argument('--ckpt', '-c', required=True, help='Path to GPTM checkpoint')
    parser.add_argument('input', nargs='?', default='', help='Input text to start generation')
    parser.add_argument('max_new_tokens', nargs='?', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--input', '-i', dest='input_flag', default=None, help='Input text (alternative)')
    parser.add_argument('--max_new_tokens', '-n', dest='max_tokens_flag', type=int, default=None, help='Max tokens (alternative)')
    parser.add_argument('--temperature', '-t', type=float, default=temperature, help='Sampling temperature')
    parser.add_argument('--top_k', '-k', type=int, default=top_k, help='Top-k sampling')
    parser.add_argument('--seed', '-s', type=int, default=seed, help='Random seed')
    parser.add_argument('--device', '-d', default=device, help='Device to use')
    parser.add_argument('--reanchor', '-r', type=int, default=0, help='Re-anchor state via teacher every N tokens (0=disabled)')
    args = parser.parse_args()
    
    # Handle both positional and flag arguments
    input_text = args.input_flag if args.input_flag is not None else args.input
    max_new_tokens = args.max_tokens_flag if args.max_tokens_flag is not None else args.max_new_tokens
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    # Setup
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.no_grad()
    
    # Load vocab
    data_dir = os.path.join('data', 'gutenberg_char')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    
    def encode(s):
        return [stoi[c] for c in s if c in stoi]
    
    def decode(tokens):
        return ''.join([itos[t] for t in tokens])
    
    # -------------------------------------------------------------------------
    # Load GPTM checkpoint
    # -------------------------------------------------------------------------
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"GPTM checkpoint not found at {ckpt_path}")
    
    print(f"Loading GPTM from {ckpt_path}")
    gptm_ckpt = torch.load(ckpt_path, map_location=args.device)
    
    # Create GPTM model
    gptm_args = gptm_ckpt['model_args']
    gptm_config = GPTMConfig(**gptm_args)
    gptm = GPTM(gptm_config)
    
    # Load GPTM weights
    state_dict = gptm_ckpt['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    gptm.load_state_dict(state_dict)
    gptm.to(args.device)
    gptm.eval()
    
    print(f"GPTM parameters: {gptm.get_num_params()/1e6:.2f}M")
    
    # -------------------------------------------------------------------------
    # Load teacher model
    # -------------------------------------------------------------------------
    teacher_ckpt_path = gptm_ckpt.get('teacher_ckpt', None)
    if teacher_ckpt_path is None:
        raise ValueError("GPTM checkpoint doesn't contain teacher_ckpt path")
    
    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_ckpt_path}")
    
    print(f"Loading teacher from {teacher_ckpt_path}")
    teacher_ckpt = torch.load(teacher_ckpt_path, map_location=args.device)
    
    # Dynamically import teacher model
    gpt_variant = teacher_ckpt['config'].get('gpt_variant', 'GPT')
    print(f"Teacher model: {gpt_variant}")
    gpt_module = importlib.import_module(gpt_variant)
    TeacherConfig = gpt_module.GPTConfig
    Teacher = gpt_module.GPT
    
    # Create teacher
    teacher_args = teacher_ckpt['model_args']
    teacher_config = TeacherConfig(**teacher_args)
    teacher = Teacher(teacher_config)
    
    # Load teacher weights
    state_dict = teacher_ckpt['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    teacher.load_state_dict(state_dict)
    teacher.to(args.device)
    teacher.eval()
    
    print(f"Teacher parameters: {teacher.get_num_params()/1e6:.2f}M")
    
    # Get teacher's input_len and q_len
    input_len = teacher_ckpt['config'].get('input_len', 512)
    teacher_q_len = teacher_config.q_len if teacher_config.q_len is not None else input_len
    print(f"Teacher input_len: {input_len}, q_len: {teacher_q_len}")
    
    # Check if residual state mode was used during training
    use_residual_state = gptm_ckpt.get('config', {}).get('use_residual_state', False)
    if use_residual_state:
        print("Residual state prediction: ENABLED (model predicts Δs)")
    
    # -------------------------------------------------------------------------
    # Setup hook to capture hidden states
    # -------------------------------------------------------------------------
    captured_hidden = {}
    
    def get_hook(name):
        def hook(module, input, output):
            captured_hidden[name] = input[0].detach()
        return hook
    
    # Hook the last transformer
    if hasattr(teacher.model, 'dec'):
        hook_handle = teacher.model.dec.register_forward_hook(get_hook('last_input'))
    elif hasattr(teacher.model, 'blocks') and len(teacher.model.blocks) > 0:
        hook_handle = teacher.model.blocks[-1].register_forward_hook(get_hook('last_input'))
    else:
        raise ValueError("Could not find last transformer in teacher model")
    
    def get_teacher_hidden(tokens):
        """Get hidden states before last transformer."""
        captured_hidden.clear()
        with torch.no_grad():
            _ = teacher(tokens)
        return captured_hidden['last_input']
    
    def forward_gptm(s_current, last_token_emb):
        """Forward through GPTM, return predicted state and token logits.
        
        In normal mode: model outputs s_next directly
        In residual mode: model outputs Δs, so s_next = s_current + Δs
        """
        input_emb = torch.cat([s_current, last_token_emb], dim=1)  # (B, q_len+1, n_embd)
        
        x = gptm.model.drop(input_emb)
        for block in gptm.model.blocks:
            x = block(x, rope_start_idx=0)
        x = gptm.model.ln_o(x)
        
        # Predicted state delta or next state
        pred_state = x[:, :-1, :]  # (B, q_len, n_embd)
        logits = gptm.model.e2v(x[:, -1, :])  # (B, vocab)
        
        # Compute actual next state
        if use_residual_state:
            s_next = s_current + pred_state  # residual: s_next = s_current + Δs
        else:
            s_next = pred_state  # direct: s_next = output
        
        return s_next, logits
    
    # -------------------------------------------------------------------------
    # Encode input
    # -------------------------------------------------------------------------
    if input_text:
        input_ids = encode(input_text)
        if not input_ids:
            print(f"Warning: Input text '{input_text}' has no valid characters")
            input_ids = [0]
    else:
        input_ids = [stoi.get('\n', 0)]
    
    # Pad input to input_len if needed
    if len(input_ids) < input_len:
        # Pad with newlines at the beginning
        pad_token = stoi.get('\n', 0)
        input_ids = [pad_token] * (input_len - len(input_ids)) + input_ids
    elif len(input_ids) > input_len:
        # Truncate to last input_len tokens
        input_ids = input_ids[-input_len:]
    
    x = torch.tensor([input_ids], dtype=torch.long, device=args.device)  # (1, input_len)
    
    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------
    print(f"\nInput: '{input_text}'")
    print(f"Generating {max_new_tokens} tokens with temperature={args.temperature}, top_k={args.top_k}")
    if args.reanchor > 0:
        print(f"Re-anchoring via teacher every {args.reanchor} tokens")
    print("-" * 50)
    
    generated_tokens = list(input_ids)
    reanchor_count = 0
    
    with ctx:
        # Get initial hidden state from teacher
        s_current = get_teacher_hidden(x)  # (1, q_len, n_embd)
        last_token = x[:, -1:]  # (1, 1)
        
        for step in range(max_new_tokens):
            # Re-anchor: get fresh state from teacher every N steps
            if args.reanchor > 0 and step > 0 and step % args.reanchor == 0:
                # Build current sequence (last input_len tokens)
                current_seq = generated_tokens[-input_len:]
                x_reanchor = torch.tensor([current_seq], dtype=torch.long, device=args.device)
                s_current = get_teacher_hidden(x_reanchor)
                reanchor_count += 1
            
            # Get embedding for last token
            last_token_emb = teacher.model.v2e(last_token)  # (1, 1, n_embd)
            
            # Forward through GPTM
            s_next, logits = forward_gptm(s_current, last_token_emb)
            
            # Apply temperature
            logits = logits / args.temperature
            
            # Top-k sampling
            if args.top_k is not None:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            
            # Append to generated
            generated_tokens.append(next_token.item())
            
            # Update state and last token for next iteration
            s_current = s_next
            last_token = next_token
    
    # Cleanup
    hook_handle.remove()
    
    # Decode and print
    output_text = decode(generated_tokens)
    print(output_text)
    print("-" * 50)
    print(f"Generated {max_new_tokens} new tokens")
    if reanchor_count > 0:
        print(f"Re-anchored {reanchor_count} times")


if __name__ == '__main__':
    main()
