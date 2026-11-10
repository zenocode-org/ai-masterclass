"""minigpt.py — code shared by the chapter 2 notebooks.

Everything here is built and explained step by step in the notebooks:
  - corpus loading, CharTokenizer, prepare_data                    -> 01_datasets
  - BigramLanguageModel                                            -> 02 and 03
  - get_batch                                                      -> 03_batch_learning
  - Head, MultiHeadAttention                                       -> 05_attention
  - FeedForward                                                    -> 06_feedforward
  - Block                                                          -> 07_stacking_layers_and_backprop
  - GPTLanguageModel                                               -> 08_gpt_language_model
  - estimate_loss                                                  -> 09_training_and_evaluation
Later notebooks import it so they do not copy code you have already seen.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# On laptops with many cores, too many threads slow PyTorch down.
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

HERE = Path(__file__).resolve().parent
CORPUS = {
    "shakespeare": "shakespeare.txt",
    "kaamelott": "kaamelott.txt",
    "plato": "plato_train.txt",
}
# Files are looked up next to the notebook, then in the repository's corpus/ folder,
# then (for Plato) in philo-masterclass/data, next to the repository.
FOLDERS = [HERE, HERE / "data", *(p / "corpus" for p in HERE.parents),
           HERE.parents[2] / "philo-masterclass" / "data"]


def corpus_path(name):
    filename = CORPUS[name]
    for folder in FOLDERS:
        if (folder / filename).exists():
            return folder / filename
    raise FileNotFoundError(f"{filename} not found. Put it next to the notebook.")


def load_text(name):
    return corpus_path(name).read_text(encoding="utf-8")


class CharTokenizer:
    """The simplest tokenizer: one character = one integer."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def prepare_data(name, train_frac=0.9):
    """Text -> tokenizer -> tensor of indices -> (train, val), split in text order."""
    text = load_text(name)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    return text, tok, data[:n], data[n:]


def get_batch(data, batch_size, block_size, device="cpu"):
    """Draws batch_size windows of block_size characters; y = x shifted by one."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    """The bigram: a (V, V) table that directly holds the logits."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)          # (B, T, V)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                     # last position
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)    # "what I contain"
        self.query = nn.Linear(n_embd, head_size, bias=False)  # "what I am looking for"
        self.value = nn.Linear(n_embd, head_size, bias=False)  # "what I pass on"
        # The mask is NOT a parameter: register_buffer stores it with the model
        # (and moves it to the GPU with it) without the optimizer touching it.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                          # (B, T, hs)
        q = self.query(x)                                        # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # (B, T, T) scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # forbid the future
        wei = F.softmax(wei, dim=-1)                             # each row sums to 1
        wei = self.dropout(wei)
        v = self.value(x)                                        # (B, T, hs)
        return wei @ v                                           # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated then projected back to n_embd channels."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)    # (B, T, n_head*hs)
        return self.dropout(self.proj(out))                     # (B, T, n_embd)


class FeedForward(nn.Module):
    """Small network applied independently at each position: C -> 4C -> ReLU -> C."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block (pre-norm): communication (attention) then computation (FeedForward)."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # residual connection #1
        x = x + self.ffwd(self.ln2(x))   # residual connection #2
        return x


class GPTLanguageModel(nn.Module):
    """Character-level, decoder-only mini-GPT."""

    def __init__(self, vocab_size, block_size=64, n_embd=64, n_head=4, n_layer=4, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                    # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb                    # (B, T, C)  broadcast over B
        x = self.blocks(x)                       # (B, T, C)
        x = self.ln_f(x)                         # (B, T, C)
        logits = self.lm_head(x)                 # (B, T, V)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]          # at most block_size characters
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # last position only
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters=20, device="cpu"):
    """Mean loss over eval_iters batches, for train AND val, with dropout disabled."""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
