"""
Training script for GPTM using knowledge distillation from a teacher model.

The teacher model is loaded from a checkpoint and used to generate hidden state
representations. GPTM learns to model the dynamics in this compressed space.

Training pairs:
- Input:  [s_x, embed(x[-1])] where s_x = teacher hidden states before last transformer
- Target: [s_y, embed(y[-1])] where s_y = teacher hidden states for shifted sequence

Loss options for hidden states (state_loss_type):
- 'mse':         Mean squared error (default)
- 'cosine':      Cosine similarity loss (1 - cos_sim)
- 'contrastive': InfoNCE loss - treats each target as a pseudo-class (like CE for embeddings)
- 'kl_div':      KL divergence on softmax distributions over embedding dimension

Usage:
    python train_gutenberg_char_gptm.py --teacher_ckpt out_gutenberg_char/ckpt_GPT2X111111_in512_q32.pt
"""

import os
import time
import math
import pickle
import argparse
import importlib
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from GPTM import GPTConfig as GPTMConfig, GPT as GPTM

# -----------------------------------------------------------------------------
# Training parameters
dataset = 'gutenberg_char'
gradient_accumulation_steps = 1
batch_size = 64
input_len = 512  # Input sequence length (must match teacher's training input_len)

# GPTM parameters (will be derived from teacher model)
n_layer = 8  # Number of layers in GPTM
dropout = 0.2
has_bias = False
init_std = 0.02

# Loss configuration
state_loss_type = 'mse'  # 'mse', 'contrastive', 'cosine', 'kl_div'
state_loss_weight = 1.0  # Weight for state loss (static)
ce_weight_max = 0.1      # Maximum weight for CE loss (dynamic, ramps up)
ce_warmup_iters = 10000  # Iterations to ramp CE weight from 0 to ce_weight_max
contrastive_temp = 0.1   # Temperature for contrastive loss
kl_div_temp = 1.0        # Temperature for KL divergence (softmax temperature)

# Regularization
state_dither_std = 0.01  # Gaussian noise std added to s_x (0 = disabled)

# I/O
out_dir = 'out_gutenberg_char'
eval_interval = 250
eval_iters = 200
eval_only = False
log_interval = 10
always_save_checkpoint = False
init_from = 'scratch'  # 'scratch' or 'auto'
wandb_log = True
wandb_project = 'gutenberg-char'

# AdamW optimizer
learning_rate = 1e-3
max_iters = 100000
lr_decay_iters = max_iters
min_lr = 1e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
decay_lr = True
warmup_iters = 100

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False

# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train GPTM with teacher distillation')
    parser.add_argument('--teacher_ckpt', '-t', required=True, help='Path to teacher checkpoint')
    args = parser.parse_args()
    
    # Sanity checks
    assert max_iters >= ce_warmup_iters, f"max_iters ({max_iters}) must be >= ce_warmup_iters ({ce_warmup_iters})"
    
    # Check input_len matches teacher (after loading checkpoint)
    def check_input_len(teacher_ckpt):
        teacher_input_len = teacher_ckpt['config'].get('input_len', None)
        if teacher_input_len is not None and input_len != teacher_input_len:
            print(f"⚠️  Warning: input_len ({input_len}) != teacher's input_len ({teacher_input_len})")
            print(f"   Consider setting input_len = {teacher_input_len}")
    
    # Setup
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(int(time.time()))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # -------------------------------------------------------------------------
    # Load teacher model
    # -------------------------------------------------------------------------
    print(f"Loading teacher model from {args.teacher_ckpt}")
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device)
    check_input_len(teacher_ckpt)
    
    # Dynamically import correct GPT variant
    gpt_variant = teacher_ckpt['config'].get('gpt_variant', 'GPT')
    print(f"Teacher model variant: {gpt_variant}")
    gpt_module = importlib.import_module(gpt_variant)
    TeacherConfig = gpt_module.GPTConfig
    Teacher = gpt_module.GPT
    
    # Create teacher model
    teacher_args = teacher_ckpt['model_args']
    teacher_config = TeacherConfig(**teacher_args)
    teacher = Teacher(teacher_config)
    
    # Load teacher weights
    state_dict = teacher_ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    teacher.load_state_dict(state_dict)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    print(f"Teacher parameters: {teacher.get_num_params()/1e6:.2f}M")
    
    # Get teacher's q_len and n_embd
    teacher_q_len = teacher_config.q_len if teacher_config.q_len is not None else input_len
    teacher_n_embd = teacher_config.n_embd
    teacher_n_head = teacher_config.n_head
    vocab_cardinality = teacher_config.vocab_cardinality
    
    print(f"Teacher q_len: {teacher_q_len}, n_embd: {teacher_n_embd}")
    
    # -------------------------------------------------------------------------
    # Setup hook to capture hidden states before last transformer
    # -------------------------------------------------------------------------
    captured_hidden = {}
    
    def get_hook(name):
        def hook(module, input, output):
            # input is a tuple, first element is the hidden states
            captured_hidden[name] = input[0].detach()
        return hook
    
    # Find and hook the last transformer (dec for GPT2X, last block for others)
    if hasattr(teacher.model, 'dec'):
        # GPT2X style: has explicit dec layer
        hook_handle = teacher.model.dec.register_forward_hook(get_hook('last_input'))
        print("Hooked teacher.model.dec (last transformer)")
    elif hasattr(teacher.model, 'blocks') and len(teacher.model.blocks) > 0:
        # GPT style: hook the last block
        hook_handle = teacher.model.blocks[-1].register_forward_hook(get_hook('last_input'))
        print(f"Hooked teacher.model.blocks[-1] (last transformer)")
    else:
        raise ValueError("Could not find last transformer block in teacher model")
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    data_dir = os.path.join('data', dataset)
    data = np.memmap(os.path.join(data_dir, 'data.bin'), dtype=np.uint16, mode='r')
    data_len = len(data)
    print(f"Dataset has {data_len:,} tokens")
    
    # Load vocab
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    assert vocab_cardinality == meta['vocab_size'], "Vocab size mismatch"
    
    def get_batch():
        max_start = data_len - input_len - 1
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + input_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + input_len].astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y
    
    # -------------------------------------------------------------------------
    # Create GPTM model
    # -------------------------------------------------------------------------
    # GPTM input length: q_len (hidden states) + 1 (last token embedding)
    gptm_args = dict(
        vocab_cardinality=vocab_cardinality,
        n_embd=teacher_n_embd,
        n_head=teacher_n_head,
        n_layer=n_layer,
        has_bias=has_bias,
        init_std=init_std,
        dropout=dropout,
    )
    
    # Checkpoint path for GPTM
    teacher_name = os.path.basename(args.teacher_ckpt).replace('.pt', '').replace('ckpt_', '')
    gptm_ckpt_path = os.path.join(out_dir, f'ckpt_GPTM_{teacher_name}.pt')
    
    wandb_run_name = f'GPTM-{teacher_name}'
    wandb_run_id = None
    first_eval = True
    iter_num = 0
    best_val_loss = 1e9
    
    # Local copy of init_from to avoid scoping issues
    init_mode = init_from
    if init_mode == 'auto' and os.path.exists(gptm_ckpt_path):
        init_mode = 'resume'
        print(f"Auto-resuming from {gptm_ckpt_path}")
    
    if init_mode == 'scratch':
        print("Initializing GPTM from scratch")
        gptm_config = GPTMConfig(**gptm_args)
        gptm = GPTM(gptm_config)
    else:
        print(f"Resuming GPTM from {gptm_ckpt_path}")
        ckpt = torch.load(gptm_ckpt_path, map_location=device)
        gptm_config = GPTMConfig(**ckpt['model_args'])
        gptm = GPTM(gptm_config)
        state_dict = ckpt['model']
        for k in list(state_dict.keys()):
            if k.startswith('_orig_mod.'):
                state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
        gptm.load_state_dict(state_dict)
        iter_num = ckpt['iter_num']
        best_val_loss = ckpt['best_val_loss']
        wandb_run_id = ckpt.get('wandb_run_id', None)
    
    gptm.to(device)
    print(f"GPTM parameters: {gptm.get_num_params()/1e6:.2f}M")
    
    # Optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = gptm.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_mode == 'resume':
        optimizer.load_state_dict(ckpt['optimizer'])
    
    if compile_model:
        print("Compiling GPTM...")
        gptm = torch.compile(gptm)
    
    # -------------------------------------------------------------------------
    # Loss functions for hidden states
    # -------------------------------------------------------------------------
    
    def compute_mse_loss(pred, target):
        """Mean squared error loss on embeddings."""
        return F.mse_loss(pred, target)
    
    def compute_cosine_loss(pred, target):
        """Cosine similarity loss (1 - cosine_sim)."""
        # Flatten to (B*T, D) and compute mean cosine similarity
        pred_flat = pred.reshape(-1, pred.size(-1))
        target_flat = target.reshape(-1, target.size(-1))
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)
        return 1.0 - cos_sim.mean()
    
    def compute_contrastive_loss(pred, target, temperature=0.1):
        """
        InfoNCE / Contrastive loss - treats each target as a pseudo-class.
        Like cross-entropy but on continuous embedding space.
        """
        B, T, D = pred.shape
        
        # Flatten to (B*T, D)
        pred_flat = pred.reshape(-1, D)
        target_flat = target.reshape(-1, D)
        
        # L2 normalize
        pred_norm = F.normalize(pred_flat, dim=-1)
        target_norm = F.normalize(target_flat, dim=-1)
        
        # Similarity matrix: (B*T, B*T)
        logits = pred_norm @ target_norm.T / temperature
        
        # Labels: diagonal is correct match
        labels = torch.arange(B * T, device=pred.device)
        
        return F.cross_entropy(logits, labels)
    
    def compute_kl_div_loss(pred, target, temperature=1.0):
        """
        KL divergence on softmax distributions over embedding dimension.
        Treats each embedding as unnormalized log-probabilities.
        """
        # Apply softmax along embedding dimension
        pred_probs = F.softmax(pred / temperature, dim=-1)
        target_probs = F.softmax(target / temperature, dim=-1)
        
        # KL divergence: sum over embedding dim, mean over batch and sequence
        kl_div = F.kl_div(
            pred_probs.log(),
            target_probs,
            reduction='batchmean'
        )
        return kl_div
    
    def compute_state_loss(pred, target):
        """Compute state loss based on selected loss type."""
        if state_loss_type == 'mse':
            return compute_mse_loss(pred, target)
        elif state_loss_type == 'cosine':
            return compute_cosine_loss(pred, target)
        elif state_loss_type == 'contrastive':
            return compute_contrastive_loss(pred, target, temperature=contrastive_temp)
        elif state_loss_type == 'kl_div':
            return compute_kl_div_loss(pred, target, temperature=kl_div_temp)
        else:
            raise ValueError(f"Unknown state_loss_type: {state_loss_type}")
    
    # -------------------------------------------------------------------------
    # Training functions
    # -------------------------------------------------------------------------
    
    def get_teacher_hidden(tokens):
        """Run teacher on tokens and return hidden states before last transformer."""
        captured_hidden.clear()
        with torch.no_grad():
            _ = teacher(tokens)
        return captured_hidden['last_input']  # (B, q_len, n_embd)
    
    def get_ce_weight(it):
        """Dynamic CE weight: ramps from 0 to ce_weight_max over ce_warmup_iters."""
        if it >= ce_warmup_iters:
            return ce_weight_max
        return ce_weight_max * (it / ce_warmup_iters)
    
    def forward_gptm(input_emb, target_emb, target_tokens, ce_weight):
        """
        Forward pass through GPTM with embedding input.
        
        Args:
            input_emb: (B, q_len+1, n_embd) - [s_x, embed(x[-1])]
            target_emb: (B, q_len+1, n_embd) - [s_y, embed(y[-1])]
            target_tokens: (B,) - y[-1] token indices for cross-entropy
            ce_weight: current CE weight (dynamic)
            
        Returns:
            loss: combined state loss + CE loss
            state_loss_val: state loss value for logging
            ce_loss_val: CE loss value for logging
        """
        B, T, n_embd = input_emb.size()
        
        # Skip v2e, apply dropout directly to embeddings
        x = gptm.model.drop(input_emb)
        
        # Transformer blocks
        for block in gptm.model.blocks:
            x = block(x, rope_start_idx=0)
        
        # Output: (B, T, n_embd)
        x = gptm.model.ln_o(x)
        
        # State loss on hidden states (positions 0..q_len-1)
        state_loss = compute_state_loss(x[:, :-1, :], target_emb[:, :-1, :])
        
        # Cross-entropy loss on last position (predicting next token)
        logits_last = gptm.model.e2v(x[:, -1, :])  # (B, vocab)
        ce_loss = F.cross_entropy(logits_last, target_tokens)
        
        # Combined loss (state_loss_weight is static, ce_weight is dynamic)
        loss = state_loss_weight * state_loss + ce_weight * ce_loss
        
        return loss, state_loss.item(), ce_loss.item()
    
    @torch.no_grad()
    def estimate_loss():
        """Estimate validation loss (CE only on predicted character)."""
        gptm.eval()
        ce_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch()
            
            # Get teacher hidden states
            s_x = get_teacher_hidden(x)  # (B, q_len, n_embd)
            
            # Get embedding for last token
            embed_x_last = teacher.model.v2e(x[:, -1:])  # (B, 1, n_embd)
            
            # Input: [s_x, embed(x[-1])]
            input_emb = torch.cat([s_x, embed_x_last], dim=1)  # (B, q_len+1, n_embd)
            target_tokens = y[:, -1]  # (B,)
            
            with ctx:
                # Forward through GPTM
                x_emb = gptm.model.drop(input_emb)
                for block in gptm.model.blocks:
                    x_emb = block(x_emb, rope_start_idx=0)
                x_emb = gptm.model.ln_o(x_emb)
                
                # CE loss on last position only
                logits_last = gptm.model.e2v(x_emb[:, -1, :])  # (B, vocab)
                ce_loss = F.cross_entropy(logits_last, target_tokens)
            
            ce_losses[k] = ce_loss.item()
        
        gptm.train()
        return {'val': ce_losses.mean()}
    
    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    
    # -------------------------------------------------------------------------
    # Wandb logging
    # -------------------------------------------------------------------------
    config = {
        'teacher_ckpt': args.teacher_ckpt,
        'gpt_variant': gpt_variant,
        'teacher_q_len': teacher_q_len,
        'n_layer': n_layer,
        'n_embd': teacher_n_embd,
        'batch_size': batch_size,
        'input_len': input_len,
        'state_loss_type': state_loss_type,
        'state_loss_weight': state_loss_weight,
        'ce_weight_max': ce_weight_max,
        'ce_warmup_iters': ce_warmup_iters,
        'contrastive_temp': contrastive_temp,
        'kl_div_temp': kl_div_temp,
        'state_dither_std': state_dither_std,
        'learning_rate': learning_rate,
        'max_iters': max_iters,
    }
    
    print(f"State loss: {state_loss_type} (weight={state_loss_weight})")
    print(f"CE weight: 0 → {ce_weight_max} over {ce_warmup_iters} iters (dynamic)")
    if state_dither_std > 0:
        print(f"State dither: std={state_dither_std}")
    
    if wandb_log:
        import wandb
        if wandb_run_id:
            try:
                wandb.init(project=wandb_project, id=wandb_run_id, resume="must", config=config)
            except wandb.errors.UsageError:
                wandb.init(project=wandb_project, name=wandb_run_name, config=config)
                wandb_run_id = wandb.run.id
        else:
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)
            wandb_run_id = wandb.run.id
    
    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print("\nStarting training...")
    t0 = time.time()
    local_iter_num = 0
    
    while iter_num <= max_iters:
        # Learning rate
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: val loss {losses['val']:.4f}")
            if wandb_log and not first_eval:
                wandb.log({
                    "iter": iter_num,
                    "val/loss": losses['val'],
                    "lr": lr,
                })
            first_eval = False
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': gptm.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': gptm_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                        'wandb_run_id': wandb_run_id,
                        'teacher_ckpt': args.teacher_ckpt,
                    }
                    print(f"Saving checkpoint to {gptm_ckpt_path}")
                    torch.save(checkpoint, gptm_ckpt_path)
        
        if iter_num == 0 and eval_only:
            break
        
        # Get batch
        x, y = get_batch()
        
        # Get teacher hidden states
        s_x = get_teacher_hidden(x)  # (B, q_len, n_embd)
        s_y = get_teacher_hidden(y)  # (B, q_len, n_embd)
        
        # Add dither to input state (regularization, training only)
        if state_dither_std > 0:
            s_x = s_x + torch.randn_like(s_x) * state_dither_std
        
        # Get embeddings for last tokens (use teacher's embedding layer)
        with torch.no_grad():
            embed_x_last = teacher.model.v2e(x[:, -1:])  # (B, 1, n_embd)
            embed_y_last = teacher.model.v2e(y[:, -1:])  # (B, 1, n_embd)
        
        # Concatenate inputs
        input_emb = torch.cat([s_x, embed_x_last], dim=1)   # (B, q_len+1, n_embd)
        target_emb = torch.cat([s_y, embed_y_last], dim=1)  # (B, q_len+1, n_embd)
        target_tokens = y[:, -1]  # (B,)
        
        # Get current CE weight (dynamic)
        ce_weight = get_ce_weight(iter_num)
        
        # Forward/backward
        with ctx:
            loss, state_loss_val, ce_loss_val = forward_gptm(input_emb, target_emb, target_tokens, ce_weight)
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gptm.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f} ({state_loss_type}={state_loss_val:.4f}, ce={ce_loss_val:.4f} w={ce_weight:.3f}), time {dt*1000:.2f}ms, lr {lr:.2e}")
        
        iter_num += 1
        local_iter_num += 1
    
    # Cleanup
    hook_handle.remove()
    print("Training complete!")


if __name__ == '__main__':
    main()
