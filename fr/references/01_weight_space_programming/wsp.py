"""wsp.py : code partagé des notebooks Weight-Space Programming.

Rien de nouveau ici. Chaque morceau est construit et expliqué pas à pas dans un notebook :
  - jeu Catch (reset, observe, step, play), politique écrite à la main  -> 01_write_and_read_a_policy
  - transformer AttnOnly, tâche de séquence répétée, ablation de tête  -> 02_reading_qk_ov_circuits
  - chargement de la modification de Qwen, mesure de répétition n-gram -> 03_dissecting_the_qwen_edit
Les notebooks suivants l'importent pour ne pas recopier du code que vous connaissez déjà.
"""
import json
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 01 : le jeu Catch
# Une balle tombe d'une ligne par pas sur une grille W x W. Une raquette d'une
# case est posée sur la ligne du bas et peut aller à gauche, rester ou aller à
# droite. Récompense : balle attrapée ou non.
# L'observation a 2 canaux (balle, raquette), comme MinAtar dans l'expérience 5.
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
    """État -> vecteur plat de 2*W*W nombres (canal 0 = balle, canal 1 = raquette)."""
    B = s["bx"].shape[0]
    o = torch.zeros(B, 2, W, W)
    o[torch.arange(B), 0, s["by"], s["bx"]] = 1.0
    o[torch.arange(B), 1, W - 1, s["px"]] = 1.0
    return o.view(B, -1)


def step(s, action):
    """action : 0 = gauche, 1 = rester, 2 = droite. Déplace la raquette, puis la balle descend d'une ligne."""
    s["px"] = (s["px"] + action - 1).clamp(0, W - 1)
    s["by"] = s["by"] + 1


@torch.no_grad()
def play(policy, batch=2000, greedy=True, seed=0):
    """Taux de balles attrapées par une politique (une fonction : observation -> 3 logits) sur `batch` balles."""
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
    """Le réseau 200 -> 2 -> 3 écrit à la main dans le notebook 01 (sans entraînement)."""
    net = nn.Sequential(nn.Linear(2 * W * W, 2), nn.ReLU(), nn.Linear(2, 3))
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        col = torch.arange(W).float()
        ball = col.repeat(W)                      # chaque case balle « vote » pour sa colonne
        paddle = torch.zeros(W, W)
        paddle[W - 1] = col
        paddle = paddle.view(-1)                  # chaque case raquette « vote » pour sa colonne
        net[0].weight[0] = torch.cat([ball, -paddle])   # h0 = relu(bx - px)
        net[0].weight[1] = torch.cat([-ball, paddle])   # h1 = relu(px - bx)
        net[2].weight[:] = torch.tensor([[0.0, 10.0], [0.0, 0.0], [10.0, 0.0]])
        net[2].bias[:] = torch.tensor([0.0, 5.0, 0.0])
    return net


# ---------------------------------------------------------------------------
# 02 : un transformer fait uniquement d'attention, aux poids de têtes faciles d'accès
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
        """W_Q, W_K, W_V (chacune hs x d) et W_O (d x hs) de la tête h."""
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
    """Embeddings + n_layer couches d'attention + unembedding. Ni MLP, ni LayerNorm :
    tout calcul est une tête, donc tout calcul peut se lire dans Q/K/V/O."""

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
    """Tokens aléatoires avec UN segment aléatoire (6 à 16 tokens) répété à un endroit aléatoire.
    Renvoie x, y (tokens suivants), mask (positions prévisibles par copie), src (position de
    la première copie de chaque token répété, -1 ailleurs)."""
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
# 03 : la modification de Qwen3.5-0.8B publiée par les auteurs (expérience 8, run p50c)
# ---------------------------------------------------------------------------
LOGS = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math"
QWEN = "Qwen/Qwen3.5-0.8B"
QWEN_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"   # les poids sur lesquels la modification a été faite


def download(url, dest):
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_p50c_edit(cache=HERE / "data" / "p50c"):
    """Le fichier diff de la modification finale d'Astra (50 passes) : entrées modifiées, sous forme de motifs de bits bfloat16."""
    import numpy as np
    man = json.loads(download(f"{LOGS}/runs/p50c/edit/manifest.json", cache / "manifest.json").read_text())
    npz = np.load(download(f"{LOGS}/runs/p50c/edit/edit.npz", cache / "edit.npz"))
    return man, npz


def apply_sparse_edit(weight, idx, val_bits):
    """Copie d'un poids bf16 dont les entrées idx sont remplacées par les motifs de bits bf16 val_bits."""
    import numpy as np
    bits = weight.detach().cpu().contiguous().view(torch.int16).numpy().copy().reshape(-1)
    bits[idx] = val_bits.view(np.int16)
    return torch.from_numpy(bits.reshape(weight.shape)).view(torch.bfloat16)


def practice_problems(n=16, cache=HERE / "data"):
    """Les n premiers problèmes d'entraînement GSM8K-train qu'Astra pouvait utiliser (aucun ne vient d'un jeu de test)."""
    path = download(f"{LOGS}/practice/gsm8k_train.jsonl", cache / "gsm8k_train.jsonl")
    with open(path) as f:
        return [json.loads(line) for _, line in zip(range(n), f)]


def tail_repetition(text, n=8, window=400):
    """Part de n-grammes de mots répétés dans les `window` derniers mots (0 = aucune répétition, ~1 = une boucle).
    Même idée que la mesure « duplicate tail phrases » d'Astra."""
    words = text.split()[-window:]
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n)]
    return 1 - len(set(grams)) / max(1, len(grams))
