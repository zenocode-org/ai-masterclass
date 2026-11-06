"""bigramme.py — petits outils partagés par les notebooks du chapitre 1.

Tout ce qui est ici est construit et expliqué pas à pas dans les notebooks :
  - charger_prenoms, construire_vocabulaire, compter_bigrammes      -> 01_compter_les_bigrammes
  - generer_prenoms                                                  -> 01_compter_les_bigrammes
  - construire_exemples                                              -> 03_one_hot_et_poids
  - forward_from_indices                                             -> 04_softmax_forward_pass
  - nll_from_probs                                                   -> 02 et 05
  - plot_matrice_caracteres : un simple outil d'affichage
Les notebooks suivants l'importent pour ne pas recopier le code déjà vu.
"""
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 2147483647
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
ICI = Path(__file__).resolve().parent


def charger_prenoms():
    """Lit names.txt à côté du notebook, ou dans le dossier parent, sinon le télécharge."""
    for dossier in [ICI, ICI / "data", ICI.parent]:
        if (dossier / "names.txt").exists():
            chemin = dossier / "names.txt"
            break
    else:
        chemin = ICI / "names.txt"
        try:
            urllib.request.urlretrieve(DATA_URL, chemin)
        except Exception as exc:
            raise FileNotFoundError(
                "names.txt introuvable et téléchargement impossible : récupérez-le sur "
                "https://github.com/karpathy/makemore et placez-le à côté du notebook."
            ) from exc
    words = [w.strip().lower() for w in chemin.read_text(encoding="utf-8").splitlines() if w.strip()]
    assert words and all(set(w) <= set("abcdefghijklmnopqrstuvwxyz") for w in words)
    return words


def construire_vocabulaire(words):
    """'.' -> 0, 'a' -> 1, ..., 'z' -> 26."""
    chars = sorted(set("".join(words)))
    stoi = {".": 0, **{c: i + 1 for i, c in enumerate(chars)}}
    itos = {i: c for c, i in stoi.items()}
    return chars, stoi, itos


def compter_bigrammes(words, stoi):
    """N[i, j] = nombre de fois où le caractère j suit immédiatement le caractère i."""
    V = len(stoi)
    N = torch.zeros((V, V), dtype=torch.int32)
    for w in words:
        sequence = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(sequence, sequence[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def construire_exemples(words, stoi):
    """Tous les bigrammes du corpus : xs = caractère courant, ys = caractère suivant."""
    xs, ys = [], []
    for w in words:
        sequence = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(sequence, sequence[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def generer_prenoms(P, itos, n=8, seed=SEED, max_len=40):
    """Échantillonne n prénoms : partir de '.', tirer le suivant dans la ligne courante, s'arrêter sur '.'."""
    g = torch.Generator().manual_seed(seed)
    resultats = []
    for _ in range(n):
        out, ix = [], 0
        for _ in range(max_len):
            ix = torch.multinomial(P[ix], 1, replacement=True, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        resultats.append("".join(out))
    return resultats


def forward_from_indices(x_indices, weights):
    """Forward pass du bigramme neuronal : indices -> one-hot -> logits -> probabilités."""
    encoded = F.one_hot(x_indices, num_classes=weights.shape[0]).float()  # (n, 27)
    logits = encoded @ weights                                            # (n, 27)
    counts = logits.exp()                                                 # (n, 27), tous > 0
    return counts / counts.sum(dim=1, keepdim=True)                       # (n, 27), lignes de somme 1


def nll_from_probs(probs, targets):
    """NLL moyenne : -log de la probabilité donnée à la bonne réponse, moyennée."""
    n = len(targets)
    bonnes_probas = probs[torch.arange(n), targets]   # (n,) : une probabilité par exemple
    return -bonnes_probas.log().mean()                # scalaire


def plot_matrice_caracteres(matrice, itos, *, title="", cbar_label="Valeur", cmap="Blues",
                            xlabel="Caractère suivant (colonne)", ylabel="Caractère courant (ligne)",
                            ax=None, figsize=(8, 6.5), annoter=False, **imshow_kwargs):
    """Heatmap d'une matrice (27, 27) : une ligne et une colonne par caractère."""
    arr = matrice.detach().cpu().numpy() if hasattr(matrice, "detach") else np.asarray(matrice)
    labels = [itos[i] for i in range(arr.shape[0])]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap=cmap, **imshow_kwargs)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if annoter:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if arr[i, j] > arr.max() / 2 else "#333")
    ax.figure.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    return ax
