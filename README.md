# AI Masterclass — du bigramme au fine-tuning d'un LLM

Un cours en notebooks Jupyter, en français, pour **comprendre les modèles de langage en les construisant**.

Le cours avance en trois étapes. Vous commencez par compter des lettres dans des prénoms. Vous écrivez ensuite un mini-GPT de zéro. Vous finissez par adapter un vrai modèle de 7 milliards de paramètres (Mistral 7B) à Kaamelott, Platon ou Shakespeare.

> **Le fil rouge des trois chapitres.** Un modèle de langage fait toujours la même chose : il **prédit le token suivant**. Il produit des **logits**, le **softmax** les transforme en probabilités, la **cross-entropy** (NLL) mesure l'erreur, et la **descente de gradient** corrige les poids. Ces quatre étapes ne changent jamais. D'un chapitre à l'autre, seule change **la façon de calculer les logits**.

---

## 1. Par où commencer ?

| Si vous… | Commencez par |
|---|---|
| découvrez le sujet | `chapitre_1/01_compter_les_bigrammes.ipynb`, puis suivez l'ordre |
| connaissez déjà softmax, cross-entropy et descente de gradient | `chapitre_2/01_datasets.ipynb` (un récapitulatif du chapitre 1 est dans `chapitre_2/02`) |
| voulez seulement le fine-tuning | `chapitre_3/README.md` (lisez au moins `chapitre_2/05` sur l'attention avant) |

**Prérequis :** Python, un peu de NumPy ou PyTorch, les matrices (produit matriciel) et les dérivées. Aucune connaissance préalable en deep learning n'est nécessaire.

**Matériel :** un portable ordinaire suffit pour les chapitres 1 et 2 et pour les notebooks 01 à 04 du chapitre 3. Tout tourne **sur CPU**. Seuls les notebooks 05 à 07 du chapitre 3 demandent un GPU, celui gratuit de Google Colab.

---

## 2. Plan du dépôt

```text
.
├── README.md                  ← vous êtes ici
├── pyproject.toml, uv.lock    ← l'environnement Python (uv sync)
├── names.txt                  ← 32 000 prénoms (chapitre 1)
├── shakespeare.txt            ← Tiny Shakespeare, 1,1 M caractères (chapitres 2 et 3)
├── kaamelott.txt              ← la série Kaamelott complète, 2,4 M caractères (chapitre 2)
├── script-kaamelott.txt       ← un court extrait de Kaamelott (chapitre 3)
├── plato_train.txt            ← dialogues de Platon, trad. Victor Cousin, 3,9 M caractères (chapitres 2 et 3)
├── preparer_kaamelott.py      ← reconstruit kaamelott.txt depuis le dataset Kaggle
│
├── chapitre_1/                ← des bigrammes au premier réseau de neurones (6 notebooks)
│   ├── 01_… 06_*.ipynb
│   ├── bigramme.py            ← le code du chapitre, réutilisé d'un notebook à l'autre
│   ├── figures/               ← schémas (SVG)
│   └── GUIDE_ENSEIGNANT.md
│
├── chapitre_2/                ← du bigramme au mini-GPT (9 notebooks)
│   ├── 01_… 09_*.ipynb
│   ├── minigpt.py             ← tokeniseur, attention, blocs, GPTLanguageModel…
│   ├── slides_encodeur_decodeur.md   ← diapos (Marp) entre les notebooks 03 et 04
│   ├── figures/
│   └── GUIDE_ENSEIGNANT.md
│
├── chapitre_3/                ← fine-tuning de Mistral 7B avec LoRA / QLoRA (7 notebooks)
│   ├── 01_… 07_*.ipynb
│   ├── finetune.py            ← données, LoRA, quantification, évaluation
│   ├── data/                  ← exemples (prompt, réponse) en JSONL, produits par le notebook 02
│   ├── checkpoints/           ← mini-GPT pré-entraîné sur Platon (notebook 03)
│   ├── figures/
│   ├── README.md              ← liens « Open in Colab »
│   └── GUIDE_ENSEIGNANT.md
│
└── CHAPITRE_1.ipynb           ← ancienne version du chapitre 1 en un seul notebook (remplacée par chapitre_1/)
```

**Trois règles pour s'y retrouver :**
1. **Les notebooks sont numérotés** : suivez l'ordre `01, 02, 03…`. Chaque notebook commence par ce que vous saurez faire à la fin, et finit par un bilan, des questions de vérification et des références.
2. **Les fichiers `.py` ne contiennent rien de nouveau.** `bigramme.py`, `minigpt.py` et `finetune.py` rangent le code **déjà expliqué** dans les notebooks, pour que les notebooks suivants l'importent au lieu de le recopier. Si vous voulez savoir d'où vient une fonction, l'en-tête du fichier dit dans quel notebook elle est construite.
3. **Les corpus sont à la racine.** Les modules les cherchent à côté du notebook, puis dans le dossier parent : vous n'avez rien à déplacer.

Les fichiers `GUIDE_ENSEIGNANT.md` s'adressent aux enseignants (déroulé, durées, chiffres mesurés). Vous pouvez les lire : ils contiennent les réponses aux questions de vérification.

---

## 3. Exécuter les notebooks

Deux possibilités : **Google Colab** (rien à installer) ou **votre machine** avec `uv`.

Tous les notebooks sont livrés **avec leurs sorties**, sauf 05 à 07 du chapitre 3 qui demandent un GPU. Vous pouvez donc les lire comme un livre avant de les exécuter.

### Option A — Google Colab (recommandé pour démarrer)

Cliquez sur un lien ci-dessous, ou sur le bouton « Open in Colab » en haut de chaque notebook. Lancez ensuite *Exécution → Tout exécuter*.

| Chapitre | Notebooks |
|---|---|
| 1 — Bigrammes | [01](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_1/01_compter_les_bigrammes.ipynb) · [02](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_1/02_mesurer_la_loss.ipynb) · [03](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_1/03_one_hot_et_poids.ipynb) · [04](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_1/04_softmax_forward_pass.ipynb) · [05](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_1/05_gradients_et_mise_a_jour.ipynb) · [06](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_1/06_entrainement_et_comparaison.ipynb) |
| 2 — Mini-GPT | [01](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_2/01_datasets.ipynb) · [02](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_2/02_bigramme_nn_module.ipynb) · [03](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_2/03_batch_learning.ipynb) · [04](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_2/04_architecture_transformer.ipynb) · [05](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_2/05_attention.ipynb) · [06](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_2/06_feedforward.ipynb) · [07](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_2/07_enchainement_backprop.ipynb) · [08](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_2/08_gpt_language_model.ipynb) · [09](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_2/09_entrainement_evaluation.ipynb) |
| 3 — Fine-tuning | [01](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/01_pourquoi_finetuner.ipynb) · [02](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/02_donnees_dialogues.ipynb) · [03](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/03_lora_a_la_main.ipynb) · [04](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/04_memoire_quantification.ipynb) · [05](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/05_qlora_mistral_colab.ipynb) · [06](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/06_evaluation_colab.ipynb) · [07](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/07_annexe_fusion_gguf.ipynb) |

**Comment ça marche.** Colab n'ouvre que le fichier `.ipynb`, sur une machine vide. La **première cellule de code** de chaque notebook détecte Colab et clone ce dépôt. Elle récupère ainsi les corpus et les modules (`bigramme.py`, `minigpt.py`, `finetune.py`), puis se place dans le dossier du chapitre. En local, cette cellule ne fait rien. Il n'y a **rien à copier sur Google Drive**.

- PyTorch, NumPy, Matplotlib et Transformers sont déjà installés sur Colab. Les notebooks 05 à 07 du chapitre 3 installent eux-mêmes leurs bibliothèques en plus (TRL, bitsandbytes…).
- Pour les notebooks 05 à 07 du chapitre 3 : *Exécution → Modifier le type d'exécution → GPU (T4)*. Les autres tournent sur le CPU gratuit.
- Vos modifications sont perdues à la fermeture de la session. Pour les garder : *Fichier → Enregistrer une copie dans Drive*.
- Après un redémarrage de la session, relancez le notebook depuis la première cellule.

### Option B — Sur votre machine, avec uv

[uv](https://docs.astral.sh/uv/) crée l'environnement Python et installe les bonnes versions en une commande. Le fichier `pyproject.toml` liste les dépendances et `uv.lock` fige leurs versions.

```bash
# 1. installer uv (une seule fois) : https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh        # Windows : voir le lien ci-dessus

# 2. récupérer le cours et créer l'environnement (.venv/)
git clone https://github.com/zenocode-org/ai-masterclass.git
cd ai-masterclass
uv sync

# 3. lancer Jupyter
uv run jupyter lab
```

L'environnement contient PyTorch, NumPy, Matplotlib, Transformers, PEFT et JupyterLab : de quoi exécuter les chapitres 1 et 2 et les notebooks 01 à 04 du chapitre 3. Pour exécuter aussi les notebooks 05 à 07 hors de Colab, il faut un GPU NVIDIA et `uv sync --group gpu`.

Dans VS Code, choisissez l'interpréteur `.venv` du dossier comme noyau des notebooks.

**Sans uv :** `python -m venv .venv`, activez-le, puis `pip install torch numpy matplotlib transformers peft accelerate jupyterlab`. Sous Linux sans GPU NVIDIA, installez d'abord la version CPU de PyTorch, bien plus légère (la version par défaut embarque CUDA, plusieurs Go) : `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

---

## 4. Chapitre 1 — Des bigrammes au premier réseau de neurones

**Question centrale :** peut-on prédire la lettre suivante d'un prénom ? On le fait d'abord **en comptant**, puis en **apprenant** une matrice de poids. Les deux chemins mènent à la même table.

**Corpus :** `names.txt`. **Durée :** environ 2 h 30. **Tout tourne sur CPU** (environ 2 minutes de calcul pour le notebook 06).

| # | Notebook | Ce que vous apprenez |
|---|---|---|
| 01 | `01_compter_les_bigrammes` | Compter les transitions dans une table `N` (27 × 27), normaliser chaque ligne en probabilités, tirer au sort avec `torch.multinomial`, générer des prénoms. Le piège du *broadcasting* (`keepdim=True`). |
| 02 | `02_mesurer_la_loss` | Noter un modèle : likelihood → log-likelihood → **NLL moyenne**. Pourquoi le logarithme, la perplexité, et pourquoi une probabilité nulle rend la loss infinie (lissage). |
| 03 | `03_one_hot_et_poids` | Encoder un caractère en **one-hot**. `one_hot @ W` revient à lire une ligne de `W`. On obtient 27 scores libres, les **logits**. |
| 04 | `04_softmax_forward_pass` | Le **softmax** (exponentielle puis normalisation) et ses propriétés. `softmax(log N) = P`. Le *forward pass* complet. |
| 05 | `05_gradients_et_mise_a_jour` | `loss.backward()`, `W.grad`, la vérification par différence finie, le gradient calculé à la main (`probs − one_hot(y)`), le *learning rate*. |
| 06 | `06_entrainement_et_comparaison` | La boucle d'entraînement complète. Le réseau retrouve la table de comptage (loss 2,458 contre 2,454). La régularisation, cousine du lissage. |

**À retenir :** logits → softmax → NLL → gradient. Ces quatre étapes seront les mêmes jusqu'à Mistral 7B.

---

## 5. Chapitre 2 — Du bigramme au mini-GPT

**Question centrale :** un bigramme ne voit qu'**un** caractère. Comment donner à chaque position accès à **tout le passé**, sans compter toutes les combinaisons ? La réponse est l'**attention**, puis le Transformer complet. Le chapitre suit la vidéo *Let's build GPT* d'Andrej Karpathy.

**Corpus :** Shakespeare, Kaamelott, Platon (au choix). **Durée :** environ 4 h 45, en deux séances. **CPU :** environ 2 minutes d'entraînement au notebook 09.

| # | Notebook | Ce que vous apprenez |
|---|---|---|
| 01 | `01_datasets` | Les trois corpus. Le tokeniseur caractère (`encode` / `decode`). Le texte devient un tenseur. La séparation train / validation. La loss de référence `log V`. |
| 02 | `02_bigramme_nn_module` | Le bigramme du chapitre 1 réécrit en PyTorch : `nn.Module`, `nn.Embedding` (= `one_hot @ W`), `F.cross_entropy`, **SGD contre AdamW**. |
| 03 | `03_batch_learning` | Les mini-batchs `(B, T)`. Pourquoi on transforme `(B, T, V)` en `(B·T, V)` avant la loss. |
| — | `slides_encodeur_decodeur.md` | 10 diapos : l'architecture encodeur-décodeur de 2017, et pourquoi GPT n'en garde que le décodeur. |
| 04 | `04_architecture_transformer` | La carte du Transformer *decoder-only* : ce que nous allons construire, pièce par pièce. |
| 05 | `05_attention` | **Le cœur du cours.** La moyenne du passé, le masque causal, le softmax masqué, **Q / K / V**, le facteur `√d`, la multi-head attention. |
| 06 | `06_feedforward` | Le réseau appliqué à chaque position, ReLU / GELU, le **dropout** (mode `train` contre mode `eval`). |
| 07 | `07_enchainement_backprop` | Comment les couches s'enchaînent et s'influencent pendant la rétropropagation. Les **connexions résiduelles** et la **LayerNorm**, démontrées par l'expérience. |
| 08 | `08_gpt_language_model` | Le modèle complet : embedding de position, blocs, `lm_head`, test de causalité, `generate`. |
| 09 | `09_entrainement_evaluation` | Entraîner (loss de validation 1,91 contre 2,48 pour le bigramme), la perplexité, la loss par position, les cartes d'attention apprises, la température, puis des expériences à faire vous-même. |

**À retenir :** **l'attention, c'est la communication entre positions ; le FeedForward, c'est le calcul à chaque position.** Les résidus et la LayerNorm permettent d'empiler les blocs.

---

## 6. Chapitre 3 — Fine-tuner Mistral 7B avec QLoRA

**Question centrale :** on ne réentraîne pas un modèle de 7 milliards de paramètres sur un portable. Comment l'**adapter** à un style (les répliques de Kaamelott, les dialogues de Platon) avec peu de mémoire ? La réponse tient en deux idées : **LoRA** (n'entraîner que de petites matrices ajoutées au modèle) et la **quantification 4 bits** (QLoRA).

**Matériel :** CPU pour les notebooks 01 à 04, GPU Colab (T4 gratuit) pour 05 à 07. Les liens Colab et le détail sont dans [`chapitre_3/README.md`](chapitre_3/README.md).

| # | Notebook | Ce que vous apprenez |
|---|---|---|
| 01 | `01_pourquoi_finetuner` | Mistral 7B, c'est le GPT du chapitre 2 en beaucoup plus grand. Les tokens BPE. Prompting, RAG ou fine-tuning ? La loss restreinte à la réponse (masque `-100`). |
| 02 | `02_donnees_dialogues` | Transformer des répliques en exemples `(prompt, réponse)`, séparer par scène, exporter en JSONL. |
| 03 | `03_lora_a_la_main` | LoRA codé à la main sur le mini-GPT du chapitre 2 : entraînement de zéro, fine-tuning complet, LoRA. Ce qu'on apprend et ce qu'on oublie. |
| 04 | `04_memoire_quantification` | Où part la mémoire pendant un entraînement. La quantification NF4. L'accumulation de gradient. |
| 05 | `05_qlora_mistral_colab` | QLoRA de Mistral-7B-Instruct avec TRL. Comparaison avant / après. |
| 06 | `06_evaluation_colab` | Perplexité, ROUGE-L, mémorisation, évaluation humaine en aveugle. |
| 07 | `07_annexe_fusion_gguf` | Fusionner l'adapter, convertir en GGUF, lancer le modèle avec llama.cpp. |

---

## 7. Ce qui change d'un chapitre à l'autre

| | Chapitre 1 | Chapitre 2 | Chapitre 3 |
|---|---|---|---|
| Token | un caractère (27) | un caractère (65 à 116) | un sous-mot BPE (32 768) |
| Contexte | 1 caractère | 64 caractères | des milliers de tokens |
| Calcul des logits | une ligne de `W` | embeddings + 4 blocs d'attention | 32 blocs pré-entraînés + petits adaptateurs LoRA |
| Paramètres entraînés | 729 | ≈ 210 000 | ≈ 42 millions, soit 0,6 % des 7 milliards |
| Loss | NLL | `F.cross_entropy` | `F.cross_entropy`, sur la réponse seulement |
| Mise à jour | `W -= lr * W.grad` | AdamW | AdamW 8 bits paginé (`paged_adamw_8bit`) |

---

## 8. Conseils pour travailler

- **Prédisez avant d'exécuter.** Beaucoup de cellules posent une question (« quelle forme ? », « quel optimiseur gagne ? »). Répondez-y avant de lancer : c'est là que l'on apprend.
- **Surveillez les formes.** Presque tous les bugs sont des formes de tenseurs inattendues. Affichez `x.shape` sans hésiter.
- **Changez un seul paramètre à la fois**, et comparez la **loss de validation** plutôt que la qualité d'un texte généré « à l'œil ». Le notebook 09 du chapitre 2 propose un tableau d'expériences.
- **Relancez le noyau** (*Kernel → Restart*) si un notebook se comporte bizarrement : les notebooks supposent qu'on les exécute dans l'ordre, depuis le début.
- **Les schémas** (`figures/`) résument chaque notion : ce sont de bonnes fiches de révision.

### Mini-glossaire

| Terme | En une phrase |
|---|---|
| **token** | l'unité que le modèle lit et prédit (ici un caractère, puis un sous-mot) |
| **logits** | des scores libres, un par token possible, avant le softmax |
| **softmax** | exponentielle puis division par la somme : les scores deviennent des probabilités |
| **NLL / cross-entropy** | −log de la probabilité donnée à la bonne réponse, en moyenne : la loss |
| **perplexité** | `exp(loss)` : le nombre de choix entre lesquels le modèle « hésite » en moyenne |
| **gradient** | la pente de la loss par rapport à chaque poids ; `backward()` le calcule |
| **embedding** | une table où chaque token lit son vecteur (`nn.Embedding`) |
| **attention** | chaque position choisit (Q·K) quelles positions passées consulter, et récupère leur contenu (V) |
| **connexion résiduelle** | `x + f(x)` : le bloc ajoute une correction au lieu de tout remplacer |
| **LoRA** | entraîner seulement deux petites matrices `A·B` ajoutées à des poids gelés |
| **quantification** | stocker les poids sur moins de bits (4 au lieu de 16) pour économiser la mémoire |

---

## 9. Pour aller plus loin

- Andrej Karpathy, série [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html), en particulier *makemore* (chapitre 1) et *Let's build GPT* (chapitre 2).
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
- J. Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/).
- 3Blue1Brown, [série sur les réseaux de neurones et les Transformers](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).
- Hu et al., [*LoRA*](https://arxiv.org/abs/2106.09685), 2021 · Dettmers et al., [*QLoRA*](https://arxiv.org/abs/2305.14314), 2023.
- D. Jurafsky et J. H. Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) (livre gratuit en ligne).

Chaque notebook se termine par ses propres références, plus précises.

---

## 10. Données et droits

| Fichier | Contenu | Source | Statut |
|---|---|---|---|
| `names.txt` | 32 033 prénoms | [makemore](https://github.com/karpathy/makemore) (A. Karpathy) | licence MIT |
| `shakespeare.txt` | *Tiny Shakespeare* | [char-rnn](https://github.com/karpathy/char-rnn) (A. Karpathy) | domaine public |
| `plato_train.txt` | dialogues de Platon, traduction de Victor Cousin | [Wikisource](https://fr.wikisource.org/wiki/Auteur:Platon) | domaine public |
| `kaamelott.txt` | les 455 épisodes de *Kaamelott*, une réplique par ligne `PERSONNAGE: réplique` | [dataset Kaggle](https://www.kaggle.com/datasets/jeanmidev/kaamelott-tv-show-transcripts-fr), mis en forme par `preparer_kaamelott.py` | œuvre protégée : usage pédagogique uniquement |
| `script-kaamelott.txt` | les premières répliques du premier épisode | même source | œuvre protégée : usage pédagogique uniquement |

*Kaamelott* est une œuvre d'Alexandre Astier, protégée par le droit d'auteur. Les transcriptions servent ici uniquement à l'enseignement. Ne redistribuez pas de modèle entraîné dessus, en particulier s'il est capable de réciter des répliques (voir le test de mémorisation du chapitre 3, notebook 06).
