import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from .LayerNorm import LayerNorm
    from .Transformer import Transformer
except ImportError:
    from LayerNorm import LayerNorm
    from Transformer import Transformer


@dataclass
class GPTConfig:
    vocab_cardinality: int = 99
    max_input_len: int = 1024  # Maximum sequence length
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    has_bias: bool = False
    init_std: float = 0.02
    dropout: float = 0.2
    use_rope: bool = False  # Use RoPE instead of learned positional embeddings
    q_len: int = None  # If set, first block uses causal_trunc_self_attn to compress to q_len tokens


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build transformer blocks
        if config.q_len is not None:
            # First block compresses to q_len, rest use causal self-attention
            blocks = [Transformer(config, attn_type='causal_trunc_self_attn')]
            blocks += [Transformer(config, attn_type='causal_self_attn') for _ in range(config.n_layer - 1)]
        else:
            # All blocks use causal self-attention
            blocks = [Transformer(config, attn_type='causal_self_attn') for _ in range(config.n_layer)]
        
        self.model = nn.ModuleDict(dict(
            v2e = nn.Embedding(config.vocab_cardinality, config.n_embd),
            drop = nn.Dropout(config.dropout),
            blocks = nn.ModuleList(blocks),
            ln_o = LayerNorm(config.n_embd, has_bias=config.has_bias),
            e2v = nn.Linear(config.n_embd, config.vocab_cardinality, bias=False),
        ))
        # Positional embeddings (only if not using RoPE)
        if not config.use_rope:
            self.model['p2e'] = nn.Embedding(config.max_input_len, config.n_embd)
        # Weight tying (https://paperswithcode.com/method/weight-tying)
        self.model.v2e.weight = self.model.e2v.weight

        # Initialize all weights
        self._init_weights()

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self):
        """
        Initialize weights for all Linear and Embedding modules in the model.
        
        Iterates over all modules and applies normal initialization with configurable
        standard deviation. Special handling for residual projection layers (c_proj)
        with scaled initialization per GPT-2 paper for training stability in deep networks.
        """
        for name, module in self.named_modules():
            # Only initialize Linear and Embedding modules
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                continue
            
            # Default initialization standard deviation
            init_std = self.config.init_std
            
            # Special scaled initialization for residual projections (c_proj)
            # Scales down by 1/sqrt(2*n_layer) to prevent activation explosion in deep networks
            if isinstance(module, nn.Linear) and name.endswith('c_proj'):
                init_std /= math.sqrt(2 * self.config.n_layer)
            
            # Initialize weights with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)

    def get_num_params(self):
        """
        Return the total number of trainable parameters in the model.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def forward(self, input, output_target=None, loss_last_only=False):
        """
        Forward pass through the GPT model.
        
        Args:
            input: Input token indices of shape (B, T)
            output_target: Target token indices of shape (B, T) for loss computation.
                          If None, no loss is computed.
            loss_last_only: If True, compute loss only on the last token position.
                           If False (default), compute loss on all output positions.
        
        Returns:
            logits: Output logits of shape (B, T, vocab_cardinality) or (B, q_len, vocab_cardinality) if q_len is set
            loss: Cross-entropy loss if output_target is provided, else None
        """
        B, T = input.size()
        assert T <= self.config.max_input_len, f"Sequence length {T} exceeds max_input_len {self.config.max_input_len}"
        
        # Token embeddings
        x = self.model.v2e(input)  # (B, T, n_embd)
        
        # Add positional embeddings (if not using RoPE)
        if not self.config.use_rope:
            pos = torch.arange(0, T, dtype=torch.long, device=input.device)  # (T,)
            x = x + self.model.p2e(pos)  # (B, T, n_embd)
        
        x = self.model.drop(x)  # (B, T, n_embd)
        
        # Transformer blocks (pass rope_start_idx=0 if using RoPE)
        # If q_len is set, first block compresses to (B, q_len, n_embd)
        rope_start_idx = 0 if self.config.use_rope else None
        for block in self.model.blocks:
            x = block(x, rope_start_idx=rope_start_idx)  # (B, T or q_len, n_embd)
        
        # Output sequence length (q_len if set, otherwise T)
        output_len = self.config.q_len if self.config.q_len is not None else T
        
        # Output layer
        x = self.model.ln_o(x)  # (B, output_len, n_embd)
        logits = self.model.e2v(x)  # (B, output_len, vocab_cardinality)
        
        # Compute loss
        if output_target is not None:
            # Use last output_len targets to match output shape
            target = output_target[:, -output_len:]  # (B, output_len)
            if loss_last_only:
                # Loss on last token only: (B, vocab_cardinality) vs (B,)
                loss = F.cross_entropy(logits[:, -1, :], target[:, -1])
            else:
                # Loss on all output tokens: (B*output_len, vocab_cardinality) vs (B*output_len,)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.reshape(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure AdamW optimizer with weight decay.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters (tuple)
            device_type: Device type ('cuda' or 'cpu')
            
        Returns:
            optimizer: Configured AdamW optimizer
        """
        # Start with all candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optimizer groups: 2D parameters (weight tensors in matmuls + embeddings) get weight decay,
        # 1D parameters (biases and layernorms) do not
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt, T):
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.
        
        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time per iteration in seconds
            T: Sequence length
            
        Returns:
            mfu: Model flops utilization as fraction of A100 peak
        """
        # See PaLM paper Appendix B as reference: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Express flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # Per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, input, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively from the model.
        
        Args:
            input: Input token indices of shape (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If specified, only sample from top-k logits
            
        Returns:
            Generated token sequence of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop input to max_input_len if needed
            input_cond = input if input.size(1) <= self.config.max_input_len else input[:, -self.config.max_input_len:]
            
            # Forward pass
            logits, _ = self(input_cond)  # (B, T, vocab_cardinality)
            
            # Get logits for last position and apply temperature
            logits = logits[:, -1, :] / temperature  # (B, vocab_cardinality)
            
            # Optionally crop to top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # (B, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')  # (B, vocab_cardinality)
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)  # (B, vocab_cardinality)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            input = torch.cat((input, next_token), dim=1)  # (B, T+1)

        return input

