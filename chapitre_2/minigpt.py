"""minigpt.py — code partagé par les notebooks du chapitre 2.

Tout ce qui est ici est construit et expliqué pas à pas dans les notebooks :
  - chargement des corpus, TokeniseurCaracteres, preparer_donnees  -> 01_datasets
  - BigramLanguageModel                                            -> 02 et 03
  - get_batch                                                      -> 03_batch_learning
  - Head, MultiHeadAttention                                       -> 05_attention
  - FeedForward                                                    -> 06_feedforward
  - Block                                                          -> 07_enchainement_backprop
  - GPTLanguageModel                                               -> 08_gpt_language_model
  - estimate_loss                                                  -> 09_entrainement_evaluation
Les notebooks suivants l'importent pour ne pas recopier le code déjà vu.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# Sur les portables à beaucoup de cœurs, trop de threads ralentit PyTorch.
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

ICI = Path(__file__).resolve().parent
CORPUS = {
    "shakespeare": "shakespeare.txt",
    "kaamelott": "script-kaamelott.txt",
    "platon": "plato_train.txt",
}
# On cherche les fichiers à côté du notebook, puis dans le dossier parent,
# puis (pour Platon) dans philo-masterclass/data.
DOSSIERS = [ICI, ICI / "data", ICI.parent, ICI.parent.parent / "philo-masterclass" / "data"]


def chemin_corpus(nom):
    fichier = CORPUS[nom]
    for dossier in DOSSIERS:
        if (dossier / fichier).exists():
            return dossier / fichier
    raise FileNotFoundError(f"{fichier} introuvable. Placez-le à côté du notebook.")


def charger_texte(nom):
    return chemin_corpus(nom).read_text(encoding="utf-8")


class TokeniseurCaracteres:
    """Le tokeniseur le plus simple : un caractère = un entier."""

    def __init__(self, texte):
        self.chars = sorted(set(texte))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def preparer_donnees(nom, part_train=0.9):
    """Texte -> tokeniseur -> tenseur d'indices -> (train, val), découpe dans l'ordre du texte."""
    texte = charger_texte(nom)
    tok = TokeniseurCaracteres(texte)
    data = torch.tensor(tok.encode(texte), dtype=torch.long)
    n = int(part_train * len(data))
    return texte, tok, data[:n], data[n:]


def get_batch(data, batch_size, block_size, device="cpu"):
    """Tire batch_size fenêtres de block_size caractères ; y = x décalé d'un cran."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    """Le bigramme : une table (V, V) qui contient directement les logits."""

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
            logits = logits[:, -1, :]                     # dernière position
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class Head(nn.Module):
    """Une tête d'auto-attention causale."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)    # « ce que je contiens »
        self.query = nn.Linear(n_embd, head_size, bias=False)  # « ce que je cherche »
        self.value = nn.Linear(n_embd, head_size, bias=False)  # « ce que je transmets »
        # Le masque n'est PAS un paramètre : register_buffer le range avec le modèle
        # (et le déplace sur le GPU avec lui) sans que l'optimiseur y touche.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                          # (B, T, hs)
        q = self.query(x)                                        # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # (B, T, T) scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # interdire le futur
        wei = F.softmax(wei, dim=-1)                             # chaque ligne somme à 1
        wei = self.dropout(wei)
        v = self.value(x)                                        # (B, T, hs)
        return wei @ v                                           # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Plusieurs têtes en parallèle, concaténées puis reprojetées sur n_embd canaux."""

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
    """Petit réseau appliqué indépendamment à chaque position : C -> 4C -> ReLU -> C."""

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
    """Bloc Transformer (pré-norm) : communication (attention) puis calcul (FeedForward)."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # connexion résiduelle n°1
        x = x + self.ffwd(self.ln2(x))   # connexion résiduelle n°2
        return x


class GPTLanguageModel(nn.Module):
    """Mini-GPT decoder-only, au niveau caractère."""

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
        x = tok_emb + pos_emb                    # (B, T, C)  broadcasting sur B
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
            idx_cond = idx[:, -self.block_size:]          # au plus block_size caractères
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # dernière position seulement
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters=20, device="cpu"):
    """Moyenne de la loss sur eval_iters batchs, pour train ET val, dropout désactivé."""
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


def compter_parametres(module):
    return sum(p.numel() for p in module.parameters())
