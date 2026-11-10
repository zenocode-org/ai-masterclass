"""bigram.py — small tools shared by the chapter 1 notebooks.

Everything here is built and explained step by step in the notebooks:
  - load_names, build_vocabulary, count_bigrams                    -> 01_counting_bigrams
  - generate_names                                                 -> 01_counting_bigrams
  - build_examples                                                 -> 03_one_hot_and_weights
  - forward_from_indices                                           -> 04_softmax_forward_pass
  - nll_from_probs                                                 -> 02 and 05
  - plot_char_matrix: a simple plotting helper
Later notebooks import it so they do not copy code you have already seen.
"""
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 2147483647
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
HERE = Path(__file__).resolve().parent


def load_names():
    """Reads names.txt next to the notebook, or in the repository's corpus/ folder, otherwise downloads it."""
    for folder in [HERE, HERE / "data", *(p / "corpus" for p in HERE.parents)]:
        if (folder / "names.txt").exists():
            path = folder / "names.txt"
            break
    else:
        path = HERE / "names.txt"
        try:
            urllib.request.urlretrieve(DATA_URL, path)
        except Exception as exc:
            raise FileNotFoundError(
                "names.txt not found and download failed: get it from "
                "https://github.com/karpathy/makemore and put it next to the notebook."
            ) from exc
    words = [w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()]
    assert words and all(set(w) <= set("abcdefghijklmnopqrstuvwxyz") for w in words)
    return words


def build_vocabulary(words):
    """'.' -> 0, 'a' -> 1, ..., 'z' -> 26."""
    chars = sorted(set("".join(words)))
    stoi = {".": 0, **{c: i + 1 for i, c in enumerate(chars)}}
    itos = {i: c for c, i in stoi.items()}
    return chars, stoi, itos


def count_bigrams(words, stoi):
    """N[i, j] = number of times character j immediately follows character i."""
    V = len(stoi)
    N = torch.zeros((V, V), dtype=torch.int32)
    for w in words:
        sequence = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(sequence, sequence[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def build_examples(words, stoi):
    """Every bigram of the corpus: xs = current character, ys = next character."""
    xs, ys = [], []
    for w in words:
        sequence = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(sequence, sequence[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def generate_names(P, itos, n=8, seed=SEED, max_len=40):
    """Samples n names: start from '.', draw the next character from the current row, stop on '.'."""
    g = torch.Generator().manual_seed(seed)
    results = []
    for _ in range(n):
        out, ix = [], 0
        for _ in range(max_len):
            ix = torch.multinomial(P[ix], 1, replacement=True, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        results.append("".join(out))
    return results


def forward_from_indices(x_indices, weights):
    """Forward pass of the neural bigram: indices -> one-hot -> logits -> probabilities."""
    encoded = F.one_hot(x_indices, num_classes=weights.shape[0]).float()  # (n, 27)
    logits = encoded @ weights                                            # (n, 27)
    counts = logits.exp()                                                 # (n, 27), all > 0
    return counts / counts.sum(dim=1, keepdim=True)                       # (n, 27), rows sum to 1


def nll_from_probs(probs, targets):
    """Mean NLL: -log of the probability given to the correct answer, averaged."""
    n = len(targets)
    correct_probs = probs[torch.arange(n), targets]   # (n,): one probability per example
    return -correct_probs.log().mean()                # scalar


def plot_char_matrix(matrix, itos, *, title="", cbar_label="Value", cmap="Blues",
                     xlabel="Next character (column)", ylabel="Current character (row)",
                     ax=None, figsize=(8, 6.5), annotate=False, **imshow_kwargs):
    """Heatmap of a (27, 27) matrix: one row and one column per character."""
    arr = matrix.detach().cpu().numpy() if hasattr(matrix, "detach") else np.asarray(matrix)
    labels = [itos[i] for i in range(arr.shape[0])]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap=cmap, **imshow_kwargs)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if arr[i, j] > arr.max() / 2 else "#333")
    ax.figure.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    return ax
