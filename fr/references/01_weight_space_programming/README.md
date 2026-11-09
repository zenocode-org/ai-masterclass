# 01 · Weight-Space Programming

> **Source :** Rikiya Takehi\*, Jayden Teoh\*, Alex Zhang, Omar Khattab (MIT CSAIL), *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*, billet de blog, 2 novembre 2026.
> <https://rikiyat.github.io/blog/weight-space-programming/>
> **Journaux complets** (chaque prompt, chaque réponse, chaque ligne de code et chaque modification de poids) : <https://github.com/RikiyaT/lm_understand-logs>
> **Modèle modifié :** <https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming>

Il s'agit d'un billet de blog accompagné de journaux publics, pas d'un article relu par des pairs. Les auteurs qualifient eux-mêmes les résultats de « très préliminaires ».

---

## Sommaire

1. [Le résumé en un paragraphe](#1-le-résumé-en-un-paragraphe)
2. [Les huit expériences en bref](#2-les-huit-expériences-en-bref)
3. [Pourquoi on peut *lire* des poids](#3-pourquoi-on-peut-lire-des-poids)
4. [La modification de Qwen, ligne par ligne](#4-la-modification-de-qwen-ligne-par-ligne)
5. [Lecture critique](#5-lecture-critique)
6. [Les notebooks](#6-les-notebooks)
7. [Glossaire](#7-glossaire)
8. [Références](#8-références)

---

## 1. Le résumé en un paragraphe

Les poids d'un réseau de neurones sont un programme, mais en général personne ne l'écrit. C'est un optimiseur comme la descente de gradient qui s'en charge (le « Software 2.0 » de Karpathy). Le billet se demande si un LLM de pointe peut jouer le rôle du programmeur : **lire** des poids pour comprendre ce que fait un réseau, et **écrire** des poids pour le modifier, **sans descente de gradient**. Ce LLM, `gpt-6-astra` d'OpenAI (appelé « Astra »), reçoit des poids sous forme de nombres, parfois un bac à sable Python, et aucun gradient. Il raisonne sur ce que calcule chaque partie. Il identifie à quel jeu joue un réseau de politique, et quelle forme dessine un modèle de diffusion. Il trouve les têtes de copie d'un petit transformer. Il répare une politique de conduite automobile défaillante en modifiant 3,5 % de ses poids, et il écrit, à partir de poids tous nuls, des réseaux de jeu qui battent PPO à Breakout. Le résultat phare : il diagnostique pourquoi Qwen3.5-0.8B échoue en mathématiques (son raisonnement **tourne en boucle** jusqu'à la limite de tokens), remonte la boucle jusqu'à une seule direction écrite par deux couches MLP, et la supprime en changeant **131 072 poids sur 0,8 milliard (0,016 %)**. MATH-500 passe de 7,9 % à 48,1 %, GSM8K de 13,8 % à 67,1 %.

Les auteurs distinguent trois capacités :

| Capacité | Ce que cela signifie | Expériences |
|---|---|---|
| **Lire** | déduire ce que fait un réseau à partir de ses poids | 1, 2, 3, 7 |
| **Réparer** | changer certains poids pour corriger un comportement | 4, 8 |
| **Écrire** | produire de zéro les poids d'un réseau qui fonctionne | 5, 6 |

---

## 2. Les huit expériences en bref

| # | Tâche | Réseau | Ce dont Astra disposait | Résultat |
|---|---|---|---|---|
| 1 | Sur quel jeu cette politique a-t-elle été entraînée ? | MLP 500→32→32→5, 17 k poids, PPO sur Breakout / Pong / Pac-Man / Freeway | les poids seulement | juste pour les 4 |
| 2 | Que dessine ce modèle de diffusion ? | MLP 160→64→64→144, 24 k poids (DDPM, images 12×12) | les poids seulement, sans forward pass | bonne forme pour les 3, mais dessinée en contour au lieu d'une forme pleine |
| 3 | Quelle langue ce transformer a-t-il apprise, et quelles têtes copient ? | 4 couches × 6 têtes, 7,2 M poids, seuls les 4 blocs du milieu sont fournis | bac à sable Python, sans forward pass | bonne langue pour les 3 (Python, romaji, anglais). Les têtes de copie correspondent aux scores mesurés |
| 4 | Réparer une voiture qui reste bloquée sur une rampe (Kinetix) | acteur PPO de 171 k poids, 0/256 réussites | poids, scène, un épisode, un seul essai | 1 024/1 024 réussites, 3,5 % des poids modifiés |
| 5 | Écrire une politique MinAtar à partir de zéros | MLP 600→32→32→6 | règles du jeu et code source, sans outils, une seule réponse | Breakout **517** contre **25** pour PPO-500M ; Seaquest 53 contre 89 ; Asterix 53 contre 55 |
| 6 | Écrire de zéro un réseau de Go (9×9) de 1,5 M poids | 243→512×6→82 | bac à sable Python, de 0 à 50 forward passes | avec 0 forward pass, gagne plus de la moitié des parties contre deux petits réseaux PPO |
| 7 | Laquelle de deux copies de Gemma-3-270M est complaisante (*sycophantic*) ? | 2 copies affinées | les poids seulement : **échec 4/4**. Avec les motifs d'attention : 3/3 | c'est la tête 10.2 qui la trahit |
| 8 | Rendre Qwen3.5-0.8B meilleur en maths | 0,8 Md, 24 couches | Python sur les poids, 10/25/50 forward passes | MATH-500 7,9 → 48,1, GSM8K 13,8 → 67,1 (50 passes) |
| + | Idem sur Gemma-3-270M (qui ne boucle pas) | 270 M | idem | **aucun gain réel** (6,8 → 7,0 à 7,7) |

C'est la dernière ligne qui compte le plus pour l'interprétation. Voir le [§5](#5-lecture-critique).

---

## 3. Pourquoi on peut *lire* des poids

Qu'on puisse (humain ou LLM) regarder une matrice de nombres et dire « ceci joue à Breakout » semble magique. Ça l'est beaucoup moins quand on voit que chaque matrice de poids a un **rôle** précis. Trois idées du cours couvrent presque tout le billet.

### 3.1 Une couche linéaire = un ensemble de détecteurs (lignes) et un ensemble d'écrivains (colonnes)

Pour `y = W x`, avec `W` de forme `(out, in)` :

- **La ligne `i`** de `W` est un *gabarit*. La sortie `i` est grande quand `x` ressemble à cette ligne (c'est un produit scalaire). Remettez une ligne de la première couche à la forme de l'entrée (une grille 10×10, une image) et vous **voyez** ce qu'elle détecte. L'expérience 1 fonctionne ainsi : des poids concentrés sur la ligne du bas de la grille signifient « cette unité surveille la raquette », donc le jeu est Breakout.
- **La colonne `j`** de `W` est ce que l'entrée `j` *écrit* dans la sortie. Dans un transformer, le `down_proj` du MLP (de forme `d_model × d_ff`) a une colonne par neurone caché. **La colonne `j` est le vecteur que le neurone `j` ajoute au flux résiduel quand il s'active.** L'expérience 8 modifie exactement ces colonnes.

> **Lien avec le cours :** au chapitre 1, la ligne `i` de la matrice de poids `W` du bigramme (27×27) contient les logits de « ce qui suit la lettre `i` ». C'était déjà une table lisible.

### 3.2 Le flux résiduel : tout le monde lit et écrit dans le même vecteur

Dans un transformer pre-norm (chapitre 2, notebook 07), `x = x + attn(ln(x))` puis `x = x + mlp(ln(x))`. Chaque couche **lit** le vecteur courant `x` et lui **ajoute** quelque chose. Le flux résiduel est donc un espace de travail partagé de dimension `d_model` (1 024 pour Qwen3.5-0.8B), et le sens vit dans des **directions** de cet espace. Si une direction `u` signifie « je suis sur le point de me répéter », alors :

- un neurone *favorise* la boucle si sa colonne de sortie a une grande composante selon `u`, c'est-à-dire un grand `uᵀ W[:, j]` ;
- on peut *supprimer* cette influence en retirant la composante : `W[:, j] ← W[:, j] − u (uᵀ W[:, j])`.

Cette seule ligne est toute la modification de Qwen. Elle est détaillée au [§4](#4-la-modification-de-qwen-ligne-par-ligne).

### 3.3 Une tête d'attention = deux petits programmes : QK (où regarder) et OV (quoi déplacer)

C'est la partie la plus utile pour la question du chapitre 2 : *« qu'apprennent vraiment Q, K et V ? »*.

Pour une position de requête `i` et une position de clé `j`, une tête calcule :

```
score(i, j) = (W_Q x_i) · (W_K x_j) = x_iᵀ (W_Qᵀ W_K) x_j        ← le circuit QK
out_i      = Σ_j softmax(score)_ij · W_O W_V x_j                ← le circuit OV
```

`W_Q` et `W_K` ne servent jamais séparément. Seul leur **produit** `W_QK = W_Qᵀ W_K` (d×d) compte, et c'est lui qui décide *quelles positions regardent lesquelles*. De même, seul `W_OV = W_O W_V` compte pour *l'information copiée* une fois l'attention décidée. Donc :

| Matrice | Question à laquelle elle répond | Comment la lire dans les poids |
|---|---|---|
| `W_QK = W_Qᵀ W_K` | « pour une requête qui ressemble à *ceci*, quelles clés semblent pertinentes ? » | la prendre en sandwich entre des embeddings : `E W_QK Eᵀ` (token→token) ou `P W_QK Pᵀ` (position→position). Une diagonale forte dans la version token signifie « regarder le même token ». Une sous-diagonale forte dans la version position signifie « regarder la position précédente ». |
| `W_OV = W_O W_V` | « si je regarde le token *t*, qu'est-ce que j'écris ? » | `U W_OV Eᵀ` est une table vocab×vocab : « regarder le token *t* augmente le logit du token *s* ». Si elle est dominée par sa diagonale, la tête **copie**. Un résumé en un nombre : le **score de copie** `Σλ / Σ|λ|` sur ses valeurs propres λ (+1 = copie pure). |

L'expérience 3 repose là-dessus. Astra a composé les matrices de poids de cette façon pour trouver les « têtes de copie » sans exécuter le modèle. La méthode vient d'Elhage et al. (2021), que personne n'avait montré à Astra. **Le notebook 02 la refait à la main.**

### 3.4 Écrire des poids = compiler un algorithme

Si vous savez ce que fait chaque matrice, vous pouvez en écrire une vous-même. Pour attraper une balle qui tombe, une politique n'a besoin que du signe de `colonne_balle − colonne_raquette`. Deux neurones ReLU le calculent, un par signe, et la couche de sortie le transforme en « gauche / rester / droite ». **Le notebook 01 écrit ce réseau à la main en 10 lignes, et il attrape 100 % des balles.** L'expérience 5 fait la même chose à plus grande échelle : « la première couche calcule où la balle va tomber, la seconde y déplace la raquette ». La version recherche de cette idée s'appelle Tracr / RASP, qui compilent des programmes en poids de transformer.

---

## 4. La modification de Qwen, ligne par ligne

D'après le rapport final d'Astra (`exp8_math/runs/p50c/work_notes/final_report.md` dans les journaux) :

```
W = W0 − 6 · outer(u, mask64 * (u @ W0))
```

appliquée à `model.language_model.layers.{15,19}.mlp.down_proj.weight` (forme 1024 × 3584).

| Symbole | Forme | Signification |
|---|---|---|
| `W0` | 1024 × 3584 | le `down_proj` d'origine : colonne `j` = ce que le neurone `j` du MLP écrit dans le flux résiduel |
| `u` | 1024 | une **« direction de répétition » unitaire** : la différence moyenne entre les vecteurs résiduels dans des contextes de raisonnement *répétitifs* et *productifs* (mesurée en sortie des couches 16 et 20) |
| `u @ W0` | 3584 | pour chaque neurone, à quel point il écrit selon `u` |
| `mask64` | 3584 | 1 pour les 64 neurones qui écrivent le plus selon `u`, 0 ailleurs |
| `outer(u, ·)` | 1024 × 3584 | une matrice de **rang 1** |

**Ce que cela fait à chacun des 64 neurones sélectionnés :** sa composante selon `u` valait `c = uᵀ W0[:, j]`. Après la modification, elle vaut `c − 6c = −5c`. Le neurone poussait le modèle *vers* la boucle. Désormais, il le pousse cinq fois plus fort *dans l'autre sens*. Tout ce qui est orthogonal à `u` est intact, de même que les 3 520 autres neurones.

- Avec `α = 1` au lieu de 6, ce serait une **projection** exacte : les neurones n'écriraient plus du tout selon `u`. C'est la même idée que l'« ablation directionnelle » utilisée pour supprimer les refus (Arditi et al., 2024).
- Avec `α = 6`, elle **dépasse** volontairement la cible et devient un petit vecteur de pilotage (*steering vector*) intégré, contre la répétition.
- Astra a choisi α, les couches et les 64 neurones par des tests dose-réponse (α = 1, 2, 4 réduisaient la log-probabilité des continuations répétitives de 0,024 / 0,049 / 0,104 nat/token, alors que les productives bougeaient à peine). Il a aussi fait un **contrôle par direction aléatoire** de même norme, qui ne gagnait que +1,6 point ± 3,5.
- Le run à 10 passes utilisait une version dense (`W' = W − 1.5 u uᵀ W` sur 30 matrices). Le run à 25 passes mettait à zéro la tête d'attention 7 de la couche 15.

**Le notebook 03 reconstruit cette modification à partir du diff publié, sur le GPU d'un ordinateur portable** (ou sur CPU). Il retrouve `u` par SVD, vérifie que le changement est exactement de rang 1, et mesure la boucle sur de vrais problèmes.

---

## 5. Lecture critique

**Points forts**

- **Tout est publié.** Chaque prompt, chaque réponse et chaque exécution de code est public, et les modifications sont fournies sous forme de diffs exacts avec leurs empreintes (*hashes*). Le notebook 03 vérifie que la modification de Qwen est bien celle décrite dans le billet.
- **Il y a des contrôles** là où c'est important (exp. 8 : contrôle par direction aléatoire, tests d'isolement des couches, problèmes d'entraînement mis de côté). Les auteurs rapportent aussi leurs échecs, ce qui compte : l'exp. 7 a échoué avec les poids seuls, Gemma n'a montré aucun gain, et l'exp. 2 a dessiné des contours au lieu de formes pleines.
- Les méthodes d'interprétabilité qu'Astra redécouvre (composition QK/OV, ablation de direction) sont **bien établies**. Cela rend les résultats plausibles.

**Points de vigilance**

- **Corriger une pathologie ≠ ajouter une capacité.** Avec le mode réflexion activé, Qwen3.5-0.8B passe 93 % de ses réponses à MATH-500 à boucler jusqu'à la limite de 32 768 tokens. Supprimer la boucle *débloque* des maths que le modèle savait déjà faire. Sur Gemma-3-270M, qui ne boucle pas, la modification n'a apporté aucun gain réel. Les auteurs écrivent que la modification « ne pouvait pas ajouter une capacité absente du modèle ». « 7,9 → 48,1 » mesure donc surtout la correction d'un bug.
- **Échantillons minuscules par endroits :** un seul essai dans l'exp. 4, trois paires de modèles dans l'exp. 7, nombre de tentatives non précisé dans les exp. 1 et 2.
- **Aucun test d'oubli.** « Seuls 0,016 % des poids ont changé, donc peu de choses sont oubliées » est présenté comme une conviction. MMLU-Pro et GPQA *montent* bien après la modification, mais aucune batterie de tests de régression large n'a été lancée.
- **Accès au modèle :** `gpt-6-astra` à l'effort de raisonnement maximal, jusqu'à 20 heures par run. Vous ne pouvez pas reproduire l'*agent* à moindre coût. Vous *pouvez* en revanche reproduire et étudier chaque *artefact* qu'il a produit, et c'est ce que font ces notebooks.
- **Petites incohérences dans le billet :** l'exp. 6 décrit l'adversaire PPO comme « de même taille » alors qu'il fait 243→32→32→82. La valeur de référence GPQA d'une figure était une valeur intermédiaire (24,5 ; la valeur finale est 22,7).
- **Est-ce vraiment « sans optimisation » ?** Aucun gradient n'a été utilisé, mais les exp. 6 et 8 conservent les changements qui gagnent sur les données d'entraînement. C'est une forme de recherche guidée par le raisonnement, ce que les auteurs reconnaissent.

**La grande idée à retenir :** l'interprétabilité et l'optimisation deviennent la même activité. Si vous savez dire *pourquoi* un réseau échoue, vous pouvez souvent le réparer par une petite modification chirurgicale au lieu de l'entraîner davantage.

---

## 6. Les notebooks

Tous les notebooks sont en français, s'exécutent de haut en bas et sont livrés avec leurs sorties : vous pouvez d'abord les lire comme un livre.

| Notebook | Lien avec le billet | Ce que vous faites | Matériel | Durée |
|---|---|---|---|---|
| [`01_write_and_read_a_policy.ipynb`](01_write_and_read_a_policy.ipynb) | exp. 1, 4, 5 | **écrire** à la main un réseau qui joue (2 neurones, 100 % de balles attrapées) ; en entraîner un par RL ; **lire** ses poids comme des images ; **réparer** un réseau cassé en modifiant ses poids | CPU | ~2 min |
| [`02_reading_qk_ov_circuits.ipynb`](02_reading_qk_ov_circuits.ipynb) | exp. 3, 7 | entraîner un minuscule transformer fait uniquement d'attention à copier des séquences répétées, puis trouver **à partir des poids seuls** quelles têtes copient et comment le circuit d'induction est câblé (QK / OV / composition). Confirmer par des ablations | CPU | ~3 min |
| [`03_dissecting_the_qwen_edit.ipynb`](03_dissecting_the_qwen_edit.ipynb) | exp. 8 | charger le vrai Qwen3.5-0.8B, appliquer la modification des auteurs, prouver qu'elle est de rang 1, retrouver la « direction de boucle », voir le modèle de base boucler et le modèle modifié s'en échapper, faire un test dose-réponse et un contrôle aléatoire | GPU ≥ 4 Go (T4 de Colab OK) ; CPU possible mais lent | ~25 min sur GPU |
| [`04_let_an_llm_read_weights.ipynb`](04_let_an_llm_read_weights.ipynb) | l'idée centrale | sérialiser un petit réseau en texte et demander à un LLM (Claude via l'API, ou copier-coller le prompt dans n'importe quel chat) sur quel jeu il a été entraîné | CPU + clé d'API facultative | ~2 min |

Le code partagé se trouve dans `wsp.py`. Comme dans le cours, il ne contient rien qui ne soit expliqué dans un notebook.

---

## 7. Glossaire

| Terme | Signification |
|---|---|
| **Weight-space programming (WSP)** | lire et écrire directement les poids d'un réseau en raisonnant, au lieu de passer par la descente de gradient |
| **Flux résiduel** (*residual stream*) | le vecteur `x` qui traverse un transformer. Chaque couche le lit et lui ajoute quelque chose |
| **Direction** | un vecteur unitaire `u` du flux résiduel qui porte un « sens ». Sa valeur à une position est `uᵀx` |
| **Modification de rang 1** | `W ← W + a bᵀ` : change toute une matrice avec une seule paire de vecteurs. Retirer une direction d'une matrice en est un exemple |
| **Ablation directionnelle** | `W ← W − u uᵀ W` : la couche ne peut plus écrire selon `u` |
| **Circuit QK** | `W_Qᵀ W_K` : décide où une tête regarde |
| **Circuit OV** | `W_O W_V` : décide ce qu'une tête déplace une fois qu'elle regarde |
| **Tête de copie** | une tête dont le circuit OV envoie chaque token sur lui-même (valeurs propres positives) |
| **Tête d'induction** | une tête qui, en voyant le token A, cherche un A antérieur et copie le token qui le *suivait*. Elle a besoin d'une tête auxiliaire dans une couche précédente (composition) |
| **Logit lens** | multiplier un vecteur du flux résiduel par l'unembedding pour voir quels tokens il favorise |
| **Dose-réponse** | appliquer une modification avec une intensité α = 0, 1, 2, … et vérifier que l'effet croît avec α. C'est un indice que l'effet est causal |

---

## 8. Références

Cités par le billet :
- A. Karpathy, *Software 2.0* (2017). <https://karpathy.medium.com/software-2-0-a64152b37c35>
- I. Schlag, K. Irie, J. Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* (2021). <https://arxiv.org/abs/2102.11174>
- D. Lindner et al., *Tracr: Compiled Transformers as a Laboratory for Interpretability* (2023). <https://arxiv.org/abs/2301.05062>
- N. Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021). <https://transformer-circuits.pub/2021/framework/index.html>

Pour approfondir les notebooks :
- C. Olsson et al., *In-context Learning and Induction Heads* (2022). <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>
- A. Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024). <https://arxiv.org/abs/2406.11717>
- K. Meng et al., *Locating and Editing Factual Associations in GPT* (ROME, 2022). <https://arxiv.org/abs/2202.05262>
- M. Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories* (2021). <https://arxiv.org/abs/2012.14913>
- G. Weiss, Y. Goldberg, E. Yahav, *Thinking Like Transformers* (RASP, 2021). <https://arxiv.org/abs/2106.06981>
