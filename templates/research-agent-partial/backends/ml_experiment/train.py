"""
train.py — Minimal GPT training for AutoResearch ratchet experiments.

AGENT CONTRACT:
- This is the ONLY file the agent modifies.
- Final stdout line must be:  VAL_BPB: <float>
- Do not modify the evaluation block marked ## FIXED: EVALUATION BLOCK.
- Defaults are tuned for 12GB VRAM (4070 Super).
  5060 Ti (16GB): DEPTH=6 and batch_size=16 are safe starting points.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# ── Hyperparameters (agent modifies this section) ────────────────────────────

DEPTH = 4              # transformer layers — keep <=4 for 12GB, <=6 for 16GB
N_HEADS = 8
D_MODEL = 512          # embedding dimension
D_FF = D_MODEL * 4
CONTEXT_LEN = 256
BATCH_SIZE = 8         # safe for 12GB; try 16 on 5060 Ti
DROPOUT = 0.1
LEARNING_RATE = 3e-4
LR_SCHEDULE = "cosine" # "flat" | "cosine" | "warmup_cosine"
WARMUP_STEPS = 100
OPTIMIZER = "adamw"    # "adamw" | "muon"  (muon requires triton)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# ── Fixed constants (do not modify) ──────────────────────────────────────────

VOCAB_SIZE = 256       # byte-level — no tokenizer needed, keeps setup simple
TRAIN_MINUTES = None   # set at runtime from env; fallback below
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert D_MODEL % N_HEADS == 0
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).split(D_MODEL, dim=2)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(reshape, qkv)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=DROPOUT if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FF, bias=False)
        self.fc2 = nn.Linear(D_FF, D_MODEL, bias=False)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[Block() for _ in range(DEPTH)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss


# ── Dataset ───────────────────────────────────────────────────────────────────

class ByteDataset(Dataset):
    def __init__(self, data: bytes, context_len: int):
        self.data = torch.frombuffer(data, dtype=torch.uint8).long()
        self.context_len = context_len

    def __len__(self):
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.context_len + 1]
        return chunk[:-1], chunk[1:]


def get_data():
    """Download TinyShakespeare if not present. Returns (train_bytes, val_bytes)."""
    cache = Path(__file__).parent / "data" / "tinyshakespeare.txt"
    if not cache.exists():
        import urllib.request
        cache.parent.mkdir(exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading TinyShakespeare...")
        urllib.request.urlretrieve(url, cache)
    raw = cache.read_bytes()
    split = int(len(raw) * 0.9)
    return raw[:split], raw[split:]


# ── LR Schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, total_steps: int) -> float:
    if LR_SCHEDULE == "flat":
        return LEARNING_RATE
    elif LR_SCHEDULE == "cosine":
        return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * step / total_steps))
    elif LR_SCHEDULE == "warmup_cosine":
        if step < WARMUP_STEPS:
            return LEARNING_RATE * step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))
    return LEARNING_RATE


# ── FIXED: EVALUATION BLOCK — do not modify ───────────────────────────────────

def evaluate_val_bpb(model: NanoGPT, val_data: bytes) -> float:
    """Compute validation bits-per-byte. Vocabulary-size independent."""
    model.eval()
    dataset = ByteDataset(val_data, CONTEXT_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    avg_nll = total_loss / total_tokens          # nats per token (byte)
    val_bpb = avg_nll / math.log(2)             # convert nats → bits
    return val_bpb

# ─────────────────────────────────────────────────────────────────────────────


def build_optimizer(model: NanoGPT):
    if OPTIMIZER == "muon":
        try:
            from muon import Muon
            return Muon(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        except ImportError:
            print("[warn] muon not installed, falling back to AdamW")
    return torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )


def main():
    import os
    train_minutes = int(os.getenv("TRAIN_MINUTES", "5"))
    deadline = time.time() + train_minutes * 60

    train_data, val_data = get_data()
    train_dataset = ByteDataset(train_data, CONTEXT_LEN)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = NanoGPT().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params | device={DEVICE} | DEPTH={DEPTH} | D_MODEL={D_MODEL}")

    optimizer = build_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    step = 0
    epoch = 0
    total_steps = int(train_minutes * 60 * 200)  # rough estimate for schedule

    model.train()
    while time.time() < deadline:
        for x, y in loader:
            if time.time() >= deadline:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            # Apply LR schedule
            lr = get_lr(step, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if step % 100 == 0:
                elapsed = time.time() - (deadline - train_minutes * 60)
                print(f"step={step} loss={loss.item():.4f} lr={lr:.2e} elapsed={elapsed:.0f}s")
            step += 1
        epoch += 1

    # FIXED: always last line — do not modify
    val_bpb = evaluate_val_bpb(model, val_data)
    print(f"VAL_BPB: {val_bpb:.4f}")


if __name__ == "__main__":
    main()
