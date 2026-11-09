"""wsp.py: shared code for the Weight-Space Programming notebooks.

Nothing here is new. Each piece is built and explained step by step in a notebook:
  - Catch game (reset, observe, step, play), hand-written policy  -> 01_write_and_read_a_policy
  - AttnOnly transformer, repeated-sequence task, head ablation    -> 02_reading_qk_ov_circuits
  - Loading the Qwen edit, n-gram repetition metric                -> 03_dissecting_the_qwen_edit
Later notebooks import it so they don't copy code you already know.
"""
import json
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 01: the Catch game
# A ball falls one row per step on a W x W grid. A 1-cell paddle sits on the
# bottom row and can move left, stay, or move right. Reward: caught or not.
# The observation has 2 channels (ball, paddle), like MinAtar in Experiment 5.
# ---------------------------------------------------------------------------
W = 10
ACTIONS = ["left", "stay", "right"]


def reset(batch, generator=None):
    return {
        "bx": torch.randint(0, W, (batch,), generator=generator),
        "by": torch.zeros(batch, dtype=torch.long),
        "px": torch.randint(0, W, (batch,), generator=generator),
    }


def observe(s):
    """State -> flat vector of 2*W*W numbers (channel 0 = ball, channel 1 = paddle)."""
    B = s["bx"].shape[0]
    o = torch.zeros(B, 2, W, W)
    o[torch.arange(B), 0, s["by"], s["bx"]] = 1.0
    o[torch.arange(B), 1, W - 1, s["px"]] = 1.0
    return o.view(B, -1)


def step(s, action):
    """action: 0 = left, 1 = stay, 2 = right. Moves the paddle, then the ball falls one row."""
    s["px"] = (s["px"] + action - 1).clamp(0, W - 1)
    s["by"] = s["by"] + 1


@torch.no_grad()
def play(policy, batch=2000, greedy=True, seed=0):
    """Catch rate of a policy (a function: observation -> 3 logits) over `batch` balls."""
    g = torch.Generator().manual_seed(seed)
    s = reset(batch, g)
    for _ in range(W - 1):
        logits = policy(observe(s))
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        step(s, a)
    return (s["px"] == s["bx"]).float().mean().item()


def hand_written_policy():
    """The 200 -> 2 -> 3 network written by hand in notebook 01 (no training)."""
    net = nn.Sequential(nn.Linear(2 * W * W, 2), nn.ReLU(), nn.Linear(2, 3))
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        col = torch.arange(W).float()
        ball = col.repeat(W)                      # each ball cell "votes" its column
        paddle = torch.zeros(W, W)
        paddle[W - 1] = col
        paddle = paddle.view(-1)                  # each paddle cell "votes" its column
        net[0].weight[0] = torch.cat([ball, -paddle])   # h0 = relu(bx - px)
        net[0].weight[1] = torch.cat([-ball, paddle])   # h1 = relu(px - bx)
        net[2].weight[:] = torch.tensor([[0.0, 10.0], [0.0, 0.0], [10.0, 0.0]])
        net[2].bias[:] = torch.tensor([0.0, 5.0, 0.0])
    return net


# ---------------------------------------------------------------------------
# 02: an attention-only transformer, with the head weights easy to reach
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head, self.hs = n_head, d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def head_weights(self, h):
        """W_Q, W_K, W_V (each hs x d) and W_O (d x hs) of head h."""
        s = slice(h * self.hs, (h + 1) * self.hs)
        return self.q.weight[s], self.k.weight[s], self.v.weight[s], self.o.weight[:, s]

    def forward(self, x, ablate_head=None):
        B, T, D = x.shape
        H, hs = self.n_head, self.hs
        q = self.q(x).view(B, T, H, hs).transpose(1, 2)
        k = self.k(x).view(B, T, H, hs).transpose(1, 2)
        v = self.v(x).view(B, T, H, hs).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / hs ** 0.5
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        pattern = scores.masked_fill(future, float("-inf")).softmax(-1)   # (B, H, T, T)
        z = pattern @ v                                                   # (B, H, T, hs)
        if ablate_head is not None:
            z = z.clone()
            z[:, ablate_head] = 0.0
        return self.o(z.transpose(1, 2).reshape(B, T, D)), pattern


class AttnOnly(nn.Module):
    """Embeddings + n_layer attention layers + unembedding. No MLP, no LayerNorm:
    every computation is a head, so every computation can be read from Q/K/V/O."""

    def __init__(self, vocab=48, d_model=64, n_head=4, n_layer=2, ctx=48):
        super().__init__()
        self.vocab, self.ctx = vocab, ctx
        self.E = nn.Embedding(vocab, d_model)
        self.P = nn.Embedding(ctx, d_model)
        self.layers = nn.ModuleList([Attention(d_model, n_head) for _ in range(n_layer)])
        self.U = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx, ablate=None):
        x = self.E(idx) + self.P(torch.arange(idx.shape[1], device=idx.device))
        patterns = []
        for layer_i, layer in enumerate(self.layers):
            head = ablate[1] if ablate is not None and ablate[0] == layer_i else None
            out, pat = layer(x, ablate_head=head)
            patterns.append(pat)
            x = x + out
        return self.U(x), patterns


def repeated_batch(batch, vocab=48, ctx=48, generator=None):
    """Random tokens with ONE random segment (6 to 16 tokens) repeated at a random place.
    Returns x, y (next tokens), mask (positions predictable by copying), src (where the
    first copy of each repeated token sits, -1 elsewhere)."""
    x = torch.randint(0, vocab, (batch, ctx + 1), generator=generator)
    mask = torch.zeros(batch, ctx + 1, dtype=torch.bool)
    src = torch.full((batch, ctx + 1), -1)
    for b in range(batch):
        n = int(torch.randint(6, 17, (1,), generator=generator))
        p = int(torch.randint(0, ctx + 1 - 2 * n, (1,), generator=generator))
        x[b, p + n:p + 2 * n] = x[b, p:p + n]
        mask[b, p + n + 1:p + 2 * n] = True
        src[b, p + n:p + 2 * n] = torch.arange(p, p + n)
    return x[:, :-1], x[:, 1:], mask[:, 1:], src[:, :-1]


@torch.no_grad()
def repeated_loss(model, x, y, mask, ablate=None):
    logits, _ = model(x, ablate=ablate)
    return F.cross_entropy(logits[mask], y[mask]).item()


# ---------------------------------------------------------------------------
# 03: the Qwen3.5-0.8B edit published by the authors (Experiment 8, run p50c)
# ---------------------------------------------------------------------------
LOGS = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math"
QWEN = "Qwen/Qwen3.5-0.8B"
QWEN_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"   # the weights the edit was made on


def download(url, dest):
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_p50c_edit(cache=HERE / "data" / "p50c"):
    """The diff file of Astra's final 50-pass edit: changed entries as bfloat16 bit patterns."""
    import numpy as np
    man = json.loads(download(f"{LOGS}/runs/p50c/edit/manifest.json", cache / "manifest.json").read_text())
    npz = np.load(download(f"{LOGS}/runs/p50c/edit/edit.npz", cache / "edit.npz"))
    return man, npz


def apply_sparse_edit(weight, idx, val_bits):
    """Copy of a bf16 weight with entries idx replaced by the bf16 bit patterns val_bits."""
    import numpy as np
    bits = weight.detach().cpu().contiguous().view(torch.int16).numpy().copy().reshape(-1)
    bits[idx] = val_bits.view(np.int16)
    return torch.from_numpy(bits.reshape(weight.shape)).view(torch.bfloat16)


def practice_problems(n=16, cache=HERE / "data"):
    """The first n GSM8K-train practice problems Astra could use (none are from a test set)."""
    path = download(f"{LOGS}/practice/gsm8k_train.jsonl", cache / "gsm8k_train.jsonl")
    with open(path) as f:
        return [json.loads(line) for _, line in zip(range(n), f)]


def tail_repetition(text, n=8, window=400):
    """Share of repeated word n-grams in the last `window` words (0 = no repetition, ~1 = a loop).
    The same idea as Astra's 'duplicate tail phrases' measure."""
    words = text.split()[-window:]
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n)]
    return 1 - len(set(grams)) / max(1, len(grams))
