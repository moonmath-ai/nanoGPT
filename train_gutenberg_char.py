"""
Training script for GPT on Gutenberg character-level dataset.

Can be run on a single GPU or with distributed data parallel (DDP).

To run on a single GPU:
$ python train_gutenberg_char.py --batch_size=32 --compile=False

To run with DDP on 4 GPUs on 1 node:
$ torchrun --standalone --nproc_per_node=4 train_gutenberg_char.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# gpt_variant = 'GPT'  # Must match the import below
# from GPT import GPTConfig, GPT  # encoder -> self-attention
# gpt_variant = 'GPT1'
# from GPT1 import GPTConfig, GPT  # encoder -> cross-attention with encoder's output
# gpt_variant = 'GPT2'
# from GPT2 import GPTConfig, GPT  # encoder -> cross-attention with encoder's input
gpt_variant = 'GPT2X'
from GPT2X import GPTConfig, GPT  # encoder -> cross-attention with encoder's input or self-attention -> decoder

# -----------------------------------------------------------------------------
# Training parameters
dataset = 'gutenberg_char'
gradient_accumulation_steps = 1
batch_size = 64
input_len = 512  # Input sequence length
loss_last_only = False  # If True, compute loss only on the last token position

# Model parameters
n_embd = 384
n_head = 6
n_layer = 8
dropout = 0.2
has_bias = False
init_std = 0.02
q_len = 32  # If None, uses input_len (no compression)
assert q_len is None or q_len <= input_len, f"q_len ({q_len}) must be <= input_len ({input_len})"
block_bmp = 63

# I/O
out_dir = 'out_gutenberg_char'
eval_interval = 250
eval_iters = 200
eval_only = False
log_interval = 10
always_save_checkpoint = False
init_from = 'scratch'  # 'scratch' or 'auto' (find latest matching checkpoint)
wandb_log = True
wandb_project = 'gutenberg-char'
wandb_run_name = f'{gpt_variant}-gutenberg-char-in{input_len}-q{q_len}'

# AdamW optimizer
learning_rate = 1e-3
max_iters = 1000000
lr_decay_iters = max_iters
min_lr = 1e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
# # Adjusted params
# learning_rate = 1e-3
# min_lr = 1e-5  # 100x lower for finer tuning
# warmup_iters = max(100, int(0.03 * max_iters))  # 3% warmup
# lr_decay_iters = int(0.9 * max_iters)  # start decay at 10%, end at 90%

# DDP settings
backend = 'nccl'

# System
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Disabled due to torch.compile issues with causal_trunc_self_attn mask

# Collect all config variables into a dict (for checkpoints and wandb logging)
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(int(time.time()) + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------

# Data loader
data_dir = os.path.join('data', dataset)
data = np.memmap(os.path.join(data_dir, 'data.bin'), dtype=np.uint16, mode='r')
data_len = len(data)
print(f"Dataset has {data_len:,} tokens")


def get_batch():
    """
    Get a batch of training data for standard next-token prediction.
    
    - Randomizes starting positions in the dataset
    - Input: sequence of length input_len
    - Target: input shifted by 1 (next token for each position)
    """
    # Ensure we have room for input_len + 1 (need one extra for target shift)
    max_start = data_len - input_len - 1
    if max_start <= 0:
        raise ValueError(f"Dataset too small for input_len={input_len}")
    
    # Randomize starting positions
    ix = torch.randint(0, max_start, (batch_size,))
    
    # Get input sequences: (batch_size, input_len)
    x = torch.stack([torch.from_numpy(data[i:i + input_len].astype(np.int64)) for i in ix])
    
    # Get target sequences: input shifted by 1 (next token prediction)
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + input_len].astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    
    return x, y


# -----------------------------------------------------------------------------

# Initialize training state
iter_num = 0
best_val_loss = 1e9
val_loss_no_improve_count = 0
early_stop_patience = 3

# Load vocab size from meta
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_cardinality = meta['vocab_size']
    print(f"Found vocab_size = {vocab_cardinality} (inside {meta_path})")
else:
    raise FileNotFoundError(f"meta.pkl not found at {meta_path}")

# -----------------------------------------------------------------------------

# Model init
model_args = dict(
    vocab_cardinality=vocab_cardinality,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    has_bias=has_bias,
    init_std=init_std,
    dropout=dropout,
    q_len=q_len,
    block_bmp=block_bmp,
)

def get_checkpoint_path():
    """Get checkpoint path for current config (gpt_variant, input_len, q_len)."""
    return os.path.join(out_dir, f'ckpt_{gpt_variant}_in{input_len}_q{q_len}.pt')

# Model initialization
wandb_run_id = None  # For resuming wandb runs
first_eval = True  # Skip first wandb log (only print to screen)

if init_from == 'auto':
    # Check if checkpoint exists for current config
    ckpt_path = get_checkpoint_path()
    if os.path.exists(ckpt_path):
        init_from = 'resume'
        print(f"Auto-resuming from {ckpt_path}")
    else:
        init_from = 'scratch'
        print("No matching checkpoint found, starting from scratch")

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # Only reached via 'auto' which sets ckpt_path
    print(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Force config attributes to match checkpoint
    for k in ['vocab_cardinality', 'n_embd', 'n_head', 'n_layer', 'has_bias', 'q_len']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # Fix state dict keys if needed
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # Get wandb run ID for resuming
    wandb_run_id = checkpoint.get('wandb_run_id', None)

model.to(device)

# GradScaler for float16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Compile model
if compile:
    print("Compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss():
    """
    Estimate validation loss over multiple batches.
    
    Samples random batches from training data - serves as validation since
    the dataset is large enough that sampled batches are effectively unseen.
    Always computes loss on last token only.
    """
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch()
        with ctx:
            logits, loss = model(X, Y, loss_last_only=True)
        losses[k] = loss.item()
    out['val'] = losses.mean()
    model.train()
    return out


def get_lr(it):
    """Learning rate decay scheduler (cosine with warmup)."""
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # After decay, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# Logging
if wandb_log and master_process:
    import wandb
    if wandb_run_id:
        # Try to resume existing run, fall back to new run if it doesn't exist
        print(f"Trying to resume wandb run {wandb_run_id}")
        try:
            wandb.init(project=wandb_project, id=wandb_run_id, resume="must", config=config)
        except wandb.errors.UsageError:
            print(f"Wandb run {wandb_run_id} not found, starting new run")
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)
            wandb_run_id = wandb.run.id
    else:
        # Start new run
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        wandb_run_id = wandb.run.id

# -----------------------------------------------------------------------------

# Training loop
X, Y = get_batch()
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and checkpoint
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: val loss {losses['val']:.4f}")
        if wandb_log and not first_eval:
            wandb.log({
                "iter": iter_num,
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            })
        first_eval = False
        # Check for improvement
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            val_loss_no_improve_count = 0
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'wandb_run_id': wandb_run_id,
                }
                ckpt_path = get_checkpoint_path()
                print(f"Saving checkpoint to {ckpt_path} (iter {iter_num})")
                torch.save(checkpoint, ckpt_path)
        else:
            val_loss_no_improve_count += 1
            # if val_loss_no_improve_count >= early_stop_patience:
            #     print(f"Early stopping: loss hasn't improved for {early_stop_patience} evaluations")
            #     print(f"Best loss: {best_val_loss:.4f}")
            #     break
        # Always save if requested
        if always_save_checkpoint and iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
                'wandb_run_id': wandb_run_id,
            }
            ckpt_path = get_checkpoint_path()
            print(f"Saving checkpoint to {ckpt_path} (iter {iter_num})")
            torch.save(checkpoint, ckpt_path)
    
    if iter_num == 0 and eval_only:
        break

    # Forward/backward with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y, loss_last_only=loss_last_only)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch()
        scaler.scale(loss).backward()
    
    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt, input_len)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    
    iter_num += 1
    local_iter_num += 1

    # Termination
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

