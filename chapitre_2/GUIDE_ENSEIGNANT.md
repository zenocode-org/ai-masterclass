# Guide enseignant — Chapitre 2 : du bigramme au mini-GPT

Ce document sert à **conduire** la séance : ordre, durées, messages clés, transitions, pièges et réponses aux questions. Les notebooks contiennent le cours complet. Ici, seulement ce dont vous avez besoin en tant qu'enseignant.

## Vue d'ensemble

| # | Notebook | Durée | Message en une phrase | Calcul CPU |
|---|---|---|---|---|
| 01 | `01_datasets` | 20 min | Un modèle imite **un** texte ; le texte devient une suite d'entiers. | < 5 s |
| 02 | `02_bigramme_nn_module` | 35 min | Le bigramme du chap. 1 réécrit en PyTorch ; AdamW > SGD car il normalise la taille des pas. | < 5 s |
| 03 | `03_batch_learning` | 25 min | `(B, T)` = B·T exercices ; reshape `(B·T, V)` sans changer une valeur. | < 5 s |
| — | `slides_encodeur_decodeur.md` | 10 min | Ce qu'on n'implémente pas : encodeur et cross-attention. | — |
| 04 | `04_architecture_transformer` | 20 min | La carte : entrée et sortie = bigramme, seul le milieu change. | < 5 s |
| 05 | `05_attention` | 50 min | Softmax masqué + Q·K pour choisir où regarder, V pour quoi récupérer. | < 5 s |
| 06 | `06_feedforward` | 25 min | Attention = communication, FeedForward = calcul ; ReLU indispensable ; dropout. | < 5 s |
| 07 | `07_enchainement_backprop` | 40 min | Les couches apprennent ensemble ; les résidus font passer le gradient. | ≈ 35 s |
| 08 | `08_gpt_language_model` | 25 min | Assemblage + embedding de position ; le modèle complet est causal. | < 5 s |
| 09 | `09_entrainement_evaluation` | 35 min | Courbes train/val, loss par position, têtes, génération, surapprentissage. | ≈ 3–4 min |

**Total : ≈ 4 h 45.** Découpage conseillé en deux séances :
- **Séance A (≈ 2 h 15)** : 01 → 02 → 03 → diapos → 04. Fin sur la carte de l'architecture : « la prochaine fois, on construit le milieu ».
- **Séance B (≈ 2 h 30)** : 05 → 06 → 07 → 08 → 09. Lancer l'entraînement du 09 **au début de l'explication du 09** (il tourne pendant que vous commentez la boucle).

Si vous n'avez que 2 × 2 h : raccourcissez 01 (10 min), passez vite sur la vérification « AdamW à la main » du 02, et sautez le §5 du 07 (« une couche influence les autres ») en le donnant en lecture.

## Mise en place

- Dossier : `from-bigram-to-transformer/chapitre_2/`. Les notebooks importent `minigpt.py` (même dossier). Le code de ce module est exactement celui construit dans les notebooks : il évite de recopier les classes d'un notebook à l'autre.
- Données : `shakespeare.txt` et `script-kaamelott.txt` sont dans le dossier parent. **`plato_train.txt` est cherché dans `../philo-masterclass/data/`** : pour distribuer le cours aux étudiants, copiez-le à côté des notebooks (ou dans `chapitre_2/data/`).
- Dépendances : `torch`, `matplotlib`, `numpy`. Tout tourne sur CPU ; `minigpt` limite PyTorch à 8 threads (au-delà, c'est plus lent sur les portables hybrides).
- Les schémas sont dans `figures/` (SVG) : ils s'affichent dans Jupyter et VS Code. Deux images viennent de Wikimedia (connexion requise, crédits sous les images).
- Tous les notebooks ont été exécutés et sauvegardés **avec leurs sorties** : vous pouvez les projeter sans rien relancer.

## Le fil rouge à répéter

> **Le problème ne change jamais :** prédire le caractère suivant. La loss (cross-entropy) et la boucle (forward → zero_grad → backward → step) sont les mêmes du bigramme au GPT. **Seule la façon de produire les logits change.**

Et la phrase de Karpathy pour le bloc : **l'attention, c'est la communication ; le FeedForward, c'est le calcul.**

---

## 01 — Datasets (20 min)

**Déroulé.** Aperçus des trois corpus → tableau comparatif → fréquences de caractères → tokeniseur → `data` → split train/val → `log V`.

**Messages clés.**
- Espaces et `\n` sont des caractères : le modèle apprend la mise en page.
- Un indice est une **étiquette**, pas une quantité (préparer l'embedding).
- Kaamelott fait 4 237 caractères : il servira à **montrer le surapprentissage** au 09, pas à générer de belles répliques.

**Moment à exploiter.** La cellule « Arthur, c'est pas faux ! » : Kaamelott ne connaît **pas** l'apostrophe `'` (le script utilise `’`). Excellent exemple de « le modèle ne connaît que les symboles de son corpus ».

**Questions fréquentes.**
- *Pourquoi pas des mots ?* Vocabulaire énorme (25 000 mots distincts dans Shakespeare contre 65 caractères), mots inconnus impossibles. Les vrais GPT utilisent des sous-mots (BPE), avec le même modèle derrière.
- *Pourquoi couper dans l'ordre ?* Couper au hasard laisserait les voisins de chaque caractère de validation dans le train : évaluation trop optimiste.

**Transition.** « On a des entiers. Le modèle le plus simple qui prédit l'entier suivant, vous le connaissez : c'est le bigramme. »

## 02 — Bigramme en `nn.Module` (35 min)

**Déroulé.** `nn.Module` → `nn.Embedding == one_hot @ W` → classe `BigramLanguageModel` → `F.cross_entropy` (à la main vs fonction, puis stabilité) → 120 pas avec AdamW lr=1e-3 (**rien ne bouge**) → explication SGD / Adam / AdamW → vérification d'un pas d'AdamW à la main → échelle des gradients → expérience à 6 optimiseurs → génération de 120 caractères.

**Messages clés.**
- `nn.Embedding(V, V)` **est** la matrice `W` du chapitre 1 (assert à l'écran).
- `F.cross_entropy` prend des **logits** et des **indices** ; c'est softmax + NLL, en stable.
- **AdamW** : le pas fait ≈ `lr` quelle que soit la taille du gradient (démonstration : gradient = 3, pas = 0,1).

**Chiffres mesurés (Shakespeare, 120 pas, 256 paires/pas, même graine).**

| Optimiseur | loss val |
|---|---|
| SGD lr=1e-3 | 4,72 (aucun apprentissage) |
| SGD lr=1 | 4,05 |
| SGD lr=10 | 2,89 |
| AdamW lr=1e-3 | 4,58 |
| AdamW lr=1e-2 | 3,57 |
| AdamW lr=1e-1 | **2,54** |
| meilleur bigramme (comptage) | 2,48 |

**Moment à exploiter.** Faire **prédire** avant d'exécuter l'expérience : « quel optimiseur gagne ? faut-il le même lr ? ». La plupart pensent que SGD lr=1e-3 « marchera un peu ». Puis montrer la cellule « échelle des gradients » : |gradient| moyen ≈ 4·10⁻⁴, 27 lignes sur 65 sans gradient dans un batch → SGD a besoin de lr=10.

**Piège.** La loss initiale (4,72) est **supérieure** à `log V` (4,17) : l'init `N(0,1)` donne des logits bruités, donc un modèle confiant à tort. Ce n'est pas un bug.

**Transition.** « Le bigramme plafonne à 2,48 parce qu'il ne voit qu'un caractère. Pour lui donner du contexte, il faut lui donner des **morceaux de texte**, pas des paires. »

## 03 — Batch learning (25 min)

**Déroulé.** Pourquoi des mini-batchs (histogrammes du bruit) → fenêtres x/y (figure BONJOUR) → `get_batch` → **erreur** de `F.cross_entropy` sur `(B, T, V)` → reshape (image `03_reshape.png` + figure numérique) → vérification « moyenne des B·T pertes » → bigramme version séquences.

**Messages clés.**
- Une fenêtre de T caractères = **T exercices** (contexte de 1 à T).
- Les B lignes d'un batch ne communiquent **jamais**.
- Le reshape ne fait **ni somme ni permutation** : `(b, t)` → ligne `b·T + t`, pour les logits **et** les cibles.

**Moment à exploiter.** L'erreur volontaire : `Expected target size [4, 65], got [4, 8]`. Demander : « pourquoi PyTorch parle-t-il de 65 ? » → il prend la dimension 1 pour les classes.

**Réponse à la question du notebook.** `torch.randint(len(data) - block_size)` : si on commençait à moins de T+1 caractères de la fin, `y` serait trop court.

**Transition.** Projeter les diapos encodeur-décodeur, puis 04.

## Diapos encodeur-décodeur (10 min) puis 04 — Architecture (20 min)

Les diapos (`slides_encodeur_decodeur.md`, format Marp, notes de présentation incluses) répondent à « est-ce qu'il faut un deck juste pour ça ? ». Oui, mais court : 10 diapos. La diapo centrale est le **tableau des trois attentions** (d'où viennent Q, K, V ; masque ou non).

**04, messages clés.**
- GPT = decoder-only : pas de source → pas d'encodeur → pas de cross-attention.
- Entrée, sortie et loss = bigramme ; seul **le milieu** change.
- Répartition des paramètres (≈ 211 000) : le FeedForward pèse **2 fois** l'attention dans chaque bloc (15,6 % contre 7,8 %).
- Le tableau « 2017 vs mini-GPT » : pré-norm, positions apprises, caractères. Le principe ne change pas.

**Renvoi.** Vidéo de Karpathy, chapitre *encoder vs. decoder vs. both (?) Transformers* (vers la fin de la vidéo).

## 05 — Attention (50 min, cœur du chapitre)

**Déroulé.**
1. Moyenne cumulative en boucles (5 min).
2. Astuce matricielle `A @ B` puis **softmax masqué** (10 min). Insister : pourquoi ce détour ? Parce que les scores ne resteront pas nuls.
3. Masque causal (figure 3 panneaux) : le futur = la réponse.
4. **Exemple `chat` calculé à la main** (10 min). Faire calculer la ligne `a` au tableau avant d'exécuter : e¹, e³, e¹ → 0,11 / 0,79 / 0,11. La valeur (5, 5) de `t` ne fuit nulle part.
5. Q, K, V avec l'analogie de la bibliothèque, puis la **tête « caractère précédent » construite à la main** (10 min). Message : Q/K peuvent exprimer une règle ; l'entraînement trouve ses propres règles.
6. Vrais tenseurs : formes, transposition de K, `poids @ V`, facteur √hs (variance 18,4 → 1,15).
7. Classe `Head` + **test de causalité** (positions 0–6 inchangées).
8. Multi-head : figure, code, nombre de paramètres (16 448).

**Si vous manquez de temps.** Le §5 (vrais tenseurs) peut être survolé : l'exemple à la main et la tête construite portent l'essentiel.

**Pièges et questions fréquentes.**
- *Les scores, c'est déjà des probabilités ?* Non : après softmax seulement, et ligne par ligne (`dim=-1`).
- *Pourquoi K et V séparés ?* Ce qui sert à **trouver** ≠ ce qu'on veut **récupérer** (exemple de la voyelle qui cherche la consonne précédente).
- *Le masque, c'est un paramètre ?* Non : `register_buffer`. Il n'apparaît pas dans le nombre de paramètres.
- *Pourquoi le dropout sur les poids d'attention ?* Renvoyer au 06.
- *Comment l'attention sait-elle l'ordre ?* Elle ne le sait pas : renvoyer au 08 (embedding de position).

**Note.** Dans l'ancien notebook (`Chapitre_2_GPT_pre-final`, cellule « Même idée encore, mais avec un masque + softmax »), la ligne `masked_fill` était **commentée** : la comparaison affichait donc une matrice uniforme non masquée. C'est corrigé dans le nouveau 05.

## 06 — FeedForward (25 min)

**Déroulé.** Formule (éq. 2 du papier) → deux Linear = une Linear (assert) → expérience `sin(3x)` → ReLU / GELU / tanh et leurs dérivées → test « seule la position 5 change » vs attention (5 à 9) → poids (33 088 contre 16 448) → dropout (train/eval, facteur 1/(1−p)).

**Messages clés.**
- **Attention = mélange sur T ; FeedForward = mélange sur C.**
- Sans non-linéarité, la profondeur ne sert à rien.
- Dropout : actif seulement en `train()`. D'où `model.eval()` avant de mesurer et de générer.

**Moment à exploiter.** Le test des positions modifiées : `FeedForward → [5]`, `Attention → [5, 6, 7, 8, 9]`. Cela résume à la fois le masque causal et le rôle de chaque couche.

## 07 — Enchaînement et rétropropagation (40 min)

Ce notebook répond à la demande « comment les layers s'enchaînent et s'influencent pendant l'entraînement ».

**Déroulé.**
1. **Réseau à deux nombres** (10 min) : faire le tableau de rétropropagation **au tableau** avant d'exécuter. Résultat : dL/dw1 = 15, dL/dw2 = −5. Puis `w2 = 0` (la couche 1 ne reçoit plus rien) et ReLU éteinte (plus personne n'apprend).
2. LayerNorm (moyenne 0, écart-type 1 **par position**) et `Block`.
3. **Trace des hooks** : l'ordre exact des couches et les formes. Montrer que tout le cœur reste `(B, T, C)`.
4. **Expérience centrale** (≈ 30 s) : 8 blocs, 4 variantes.
5. Influence vers l'avant (perturber le bloc 0 → tout change au-dessus) et vers l'arrière (perturber le bloc 3 → le gradient du bloc 0 change de 37 %).
6. Taille des mises à jour par groupe.

**Chiffres mesurés (8 blocs, C=64, à l'initialisation puis 200 pas d'AdamW).**

| Variante | gradient bloc 0 | gradient bloc 7 | loss après 200 pas |
|---|---|---|---|
| résidus + LayerNorm | 4·10⁻¹ | 4·10⁻¹ | 2,48 |
| résidus seuls | 6·10⁻¹ | 8·10⁻¹ | 2,48 |
| LayerNorm seule | 1,2 | 6,0 | 3,32 |
| ni l'un ni l'autre | **2·10⁻⁷** | 5,0 | 3,32 |

**Message à faire passer.** Le gradient d'une couche est un **produit** des dérivées des couches au-dessus (donc il dépend de **leurs poids**) ; la résiduelle transforme ce produit en **somme** grâce au « +1 ». Les couches **co-évoluent** : on les entraîne ensemble, avec une seule loss.

**Nuance honnête.** Sur ce petit réseau et 200 pas, « résidus seuls » fait aussi bien que « résidus + LayerNorm ». La LayerNorm compte davantage pour des réseaux plus profonds et des entraînements plus longs. Ne pas sur-vendre la LayerNorm avec cette expérience.

## 08 — GPTLanguageModel (25 min)

**Déroulé.** Embedding de position (le même token en positions 1 et 3 → vecteurs différents) → code complet en une cellule → tableau `forward` ligne par ligne → instanciation → loss initiale (4,35, proche de `log V` = 4,17) → **test de causalité du modèle complet** → `generate` (erreur volontaire sans recadrage → `IndexError`) → texte non entraîné.

**Questions fréquentes.**
- *Pourquoi additionner plutôt que concaténer token et position ?* Les deux fonctionnent ; additionner garde `C` canaux et le réseau apprend à séparer les deux informations. C'est le choix de GPT-2.
- *Pourquoi la loss initiale est-elle plus proche de log V que pour le bigramme ?* Les logits passent par LayerNorm + une Linear à petite initialisation : ils sont proches de 0.
- *KV cache ?* Mentionné en remarque : en production, on ne recalcule pas tout le contexte.

## 09 — Entraînement et évaluation (35 min)

**Lancez la cellule d'entraînement dès le début** et commentez la boucle pendant qu'elle tourne.

**Déroulé.** Hyperparamètres → boucle (figure) → courbes train/val avec repères uniforme et bigramme → perplexité → **loss par position** → **cartes d'attention apprises** → génération → température → **Kaamelott : surapprentissage** + mesure de copie → expériences → synthèse.

**Résultats de référence (CPU, 8 threads).** Voir le tableau en fin de guide.

**Messages clés.**
- Tout ce qui est sous la ligne du bigramme est dû au **contexte**.
- La **loss par position** est la preuve chiffrée : à t = 0 le GPT fait comme un bigramme, avec 10 caractères de contexte il fait nettement mieux.
- Juger un modèle sur la **validation**, pas sur un texte généré.
- Kaamelott : train → 0, val ↑. Le modèle **recopie** de longs passages ; sur Shakespeare, il compose.

**Têtes d'attention.** Les motifs varient d'un entraînement à l'autre. Cherchez avec la classe : diagonale décalée (caractère précédent), colonnes sur les espaces / retours à la ligne (début de mot ou de ligne), têtes diffuses.

**Expériences de fin.** Répartir les lignes du tableau du §9 entre groupes d'étudiants ; chacun rapporte **une** loss de validation. Comparer au tableau.

---

## Références principales (pour vous)

- A. Karpathy, *Let's build GPT: from scratch, in code, spelled out* — https://www.youtube.com/watch?v=kCc8FmEb1nY ; code : https://github.com/karpathy/ng-video-lecture
- Vaswani et al., *Attention Is All You Need*, 2017 — https://arxiv.org/abs/1706.03762
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW) — https://arxiv.org/abs/1711.05101
- He et al., *Deep Residual Learning* — https://arxiv.org/abs/1512.03385 ; Ba et al., *Layer Normalization* — https://arxiv.org/abs/1607.06450
- Srivastava et al., *Dropout* — https://jmlr.org/papers/v15/srivastava14a.html
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (flux résiduel) — https://transformer-circuits.pub/2021/framework/index.html
- J. Alammar, *The Illustrated Transformer* — https://jalammar.github.io/illustrated-transformer/
- 3Blue1Brown, *Attention in transformers, visually explained* — https://www.youtube.com/watch?v=eMlx5fFNoYc

## Résultats de référence du notebook 09

Mesuré sur CPU (8 threads), configuration du notebook : B=32, T=64, C=64, 4 têtes, 4 blocs, dropout 0,1, AdamW lr=1e-3, 2 000 pas, 211 777 paramètres.

| Mesure | Valeur |
|---|---|
| Durée de l'entraînement | 112 s (≈ 4 min pour tout le notebook) |
| Loss val : uniforme / meilleur bigramme / mini-GPT | 4,17 / 2,48 / **1,91** |
| Perplexité : uniforme / bigramme / mini-GPT | 65 / 12,0 / **6,8** caractères |
| Loss par position (val) : t=0 / t=10 / t=63 | 2,51 / 1,85 / 1,91 |
| Kaamelott, 800 pas sans dropout : train / val finale / meilleure val | 0,27 / 3,87 / 2,33 |
| Plus long passage recopié du corpus (300 caractères générés) | Kaamelott 26 car. · Shakespeare 12 car. |

Cartes d'attention observées lors de cette exécution : en couche 0, trois têtes sur quatre regardent surtout le **caractère courant** et une est diffuse. En couche 3, les motifs sont plus variés, et une tête se concentre sur le **dernier retour à la ligne** (le début de la réplique). Les motifs changent d'un entraînement à l'autre.

Référence Karpathy (GPU, ≈ 10 M de paramètres, 5 000 pas) : val ≈ 1,48.
