# Guide enseignant — Chapitre 1 : des bigrammes au premier réseau de neurones

Ce document sert à **conduire** la séance : ordre, durées, messages clés, transitions, pièges et réponses aux questions. Le cours complet est dans les notebooks. Ce guide ne garde que ce dont vous avez besoin en tant qu'enseignant.

## Vue d'ensemble

| # | Notebook | Durée | Message en une phrase | Calcul CPU |
|---|---|---|---|---|
| 01 | `01_compter_les_bigrammes` | 30 min | Compter les transitions, normaliser chaque ligne, tirer au sort : on a déjà un modèle de langage. | < 5 s |
| 02 | `02_mesurer_la_loss` | 25 min | La NLL moyenne note un modèle : −log de la probabilité de la bonne réponse, en moyenne. | < 5 s |
| 03 | `03_one_hot_et_poids` | 20 min | `one_hot(x) @ W` = lire la ligne `x` de `W` : 27 scores libres. | < 5 s |
| 04 | `04_softmax_forward_pass` | 20 min | Softmax = exp puis division par la somme ; `softmax(log N) = P`. | < 5 s |
| 05 | `05_gradients_et_mise_a_jour` | 25 min | `backward()` calcule les pentes, nous faisons le pas `W -= lr * W.grad`. | < 5 s |
| 06 | `06_entrainement_et_comparaison` | 25 min | 100 à 400 pas suffisent pour que `W` retrouve la table du comptage. | ≈ 2 min |

**Total : ≈ 2 h 25.** Pour tenir en **2 h** :
- dans le 01, passez vite sur « Est-ce mieux que le hasard ? » ;
- dans le 02, montrez la perplexité et le lissage en 3 minutes (ou en lecture) ;
- dans le 04, faites la section 3 (propriétés du softmax) à l'oral, sans les cellules ;
- dans le 06, sautez la section 5 (régularisation).

Pour une séance de 2 × 1 h 15 : coupez après le 03. La séance 2 commence alors sur la question « comment transformer des scores en probabilités ? ».

## Mise en place

- Dossier : `fr/chapitre_1/`. Les notebooks importent `bigramme.py` (même dossier). Ce module contient exactement le code construit dans les notebooks (comptage, génération, forward, NLL) et une fonction d'affichage des heatmaps.
- Données : `names.txt` est cherché à côté des notebooks, puis dans le dossier `corpus/` du dépôt (où il se trouve déjà), sinon il est téléchargé depuis le dépôt *makemore*. Pour les étudiants, copiez `names.txt` dans `chapitre_1/` pour ne dépendre d'aucune connexion.
- Dépendances : `torch`, `matplotlib`, `numpy`. Pas de GPU.
- Les schémas sont dans `figures/` (SVG, tous locaux).
- Les six notebooks ont été exécutés et sauvegardés **avec leurs sorties** : vous pouvez les projeter sans rien relancer. Le 06 prend environ 2 minutes, lancez-le pendant que vous commentez la boucle.
- Le notebook d'origine `CHAPITRE_1.ipynb` n'a pas été modifié.

## Le fil rouge à répéter

> **Deux chemins vers la même table.** Le chemin 1 est de **compter**. Le chemin 2 est d'**apprendre** `W` en minimisant la NLL. Le chemin 2 est plus long ici, mais lui seul passe à l'échelle.

Les **quatre invariants** à faire répéter à la fin : logits → softmax → NLL moyenne → descente de gradient. Ils resteront identiques pour le Transformer du chapitre 2. Seule changera la manière de calculer les logits.

---

## 01 — Compter les bigrammes (30 min)

**Déroulé.**
1. Le défi.
2. Le corpus.
3. `stoi` / `itos`.
4. La figure « emma → 5 bigrammes → 5 +1 ».
5. Le comptage `N` et le contrôle `somme = lettres + prénoms`.
6. La heatmap annotée et le top 10.
7. La normalisation de la ligne `.`.
8. Le **piège du broadcasting**.
9. `multinomial` vs `argmax`.
10. La génération, puis la comparaison avec l'uniforme.

**Messages clés.**
- Ligne = caractère courant, colonne = caractère suivant. `N` n'est pas symétrique : `N['a','m'] = 1634`, `N['m','a'] = 2590`.
- On normalise **par ligne** car on conditionne sur le caractère courant.
- `multinomial` échantillonne, `argmax` choisit toujours la même chose.

**Moments à exploiter.**
- **Le piège `keepdim`.** Faites deviner la forme de `N.sum(dim=1)` avant d'exécuter. La version fausse ne plante pas, mais ses lignes somment à 7,0 ou 0,2. C'est le bug le plus courant de ce chapitre. Message : « testez toujours que vos distributions somment à 1 ».
- **`argmax` donne `a`, `a`, `a`.** Après `.`, la lettre la plus fréquente est `a`. Après `a`, le plus fréquent est `.`. Faites-le prédire à la classe avant d'exécuter.
- **Modèle uniforme :** des chaînes de 40 caractères imprononçables, contre une longueur moyenne de 6,3 pour le bigramme (6,1 dans le corpus).

**Chiffres mesurés.**
- 32 033 prénoms, 228 146 transitions, 102 cases de `N` à zéro sur 729.
- Bigrammes les plus fréquents : `n→.` (6 763), `a→.` (6 640), `a→n` (5 438).
- Sur 1 000 prénoms générés, 102 existent déjà dans `names.txt`.

**Questions fréquentes.**
- *Pourquoi un seul symbole `.` pour le début et la fin ?* Il suffit : la ligne `.` donne les débuts, la colonne `.` les fins. Un symbole de moins à gérer.
- *Les prénoms générés sont mauvais, c'est normal ?* Oui. Le modèle n'a qu'un caractère de mémoire. C'est toute la motivation des chapitres suivants.

**Transition.** « Ces prénoms ont l'air mieux que le hasard. Mais "avoir l'air", ce n'est pas un nombre. Comment noter un modèle ? »

## 02 — Mesurer la loss (25 min)

**Déroulé.**
1. `P(e | .)`.
2. La likelihood d'`emma` (figure 02).
3. L'**underflow** du produit.
4. Le log, puis −log : la NLL.
5. La courbe −log p avec les points d'emma.
6. La NLL moyenne sur le corpus.
7. La perplexité.
8. Le piège du zéro et le lissage.

**Messages clés.**
- On ne note que la probabilité donnée à la **bonne réponse**.
- Pourquoi le log ? (1) Le produit tombe à **0 dès 100 transitions** en float32, à montrer à l'écran. (2) Le classement des modèles ne change pas. (3) Une somme se dérive facilement.
- Repères : **uniforme 3,30 · comptage 2,454**. Perplexité 27 contre 11,6.

**Moment à exploiter.** `andrejq` donne une NLL **infinie** : la transition `j→q` n'existe pas dans le corpus. Le lissage `N+1` coûte presque rien (2,4546 contre 2,4540). Avec `+100 000`, on retombe à l'uniforme (3,28). Cela prépare la régularisation du 06 et le *weight decay* d'AdamW au chapitre 2.

**Piège.** La loss est mesurée **sur les données qui ont servi à compter**. Dites-le explicitement : la séparation train/validation arrive au chapitre 2.

**Questions fréquentes.**
- *Pourquoi la moyenne et pas la somme ?* Pour comparer des corpus de tailles différentes, et pour que le learning rate ne dépende pas de la taille des données.
- *Le cours d'origine disait « on utilise le log parce que le papier Transformer le fait ».* Ce n'est pas la vraie raison. Les vraies raisons sont l'underflow, la monotonie et la dérivation (corrigé dans ce notebook).

**Transition.** « Nous avons un modèle (`P`) et une note (la NLL). Et si les probabilités venaient d'une matrice qu'on peut corriger ? »

## 03 — One-hot et poids (20 min)

**Déroulé.**
1. Le tableau comptage / réseau.
2. Pourquoi un réseau (l'explosion `27³ × 27`).
3. Les 5 exercices `xs` / `ys` d'emma.
4. Le one-hot.
5. La figure 03 et l'exemple 3 × 3.
6. 27 × 27 sur emma.
7. `W` aléatoire (heatmap et histogramme).

**Messages clés.**
- Un indice est une **étiquette** : `m = 13` n'est pas « 13 fois `a` ».
- `one_hot(x) @ W == W[x]` : c'est une **sélection de ligne** (assert à l'écran). Au chapitre 2, c'est `nn.Embedding`.
- Les deux `m` d'emma reçoivent **les mêmes scores** pour deux bonnes réponses différentes : c'est la limite du bigramme.

**Moment à exploiter.** Faites calculer `[0, 1, 0] @ W_mini` au tableau **avant** d'exécuter. Ensuite, demandez : « et `[1, 0, 0]` ? ».

**Questions fréquentes.**
- *Pourquoi `.float()` ?* `F.one_hot` renvoie des entiers, et `@` exige le même type des deux côtés.
- *Pourquoi `W` est-elle carrée ?* 27 entrées possibles × 27 sorties possibles. Au chapitre 2, l'embedding aura une autre largeur (`n_embd`).

**Transition.** « 27 scores, parfois négatifs, qui ne somment pas à 1. Comment en faire des probabilités ? »

## 04 — Softmax et forward pass (20 min)

**Déroulé.**
1. `exp` sur [−1, 0, 1].
2. La normalisation, puis le softmax.
3. Les 3 propriétés : décalage, échelle (température), stabilité numérique (la version naïve donne `nan` sur [1000, 1001, 1002]).
4. **`softmax(log N) = P`**.
5. Le forward pass complet (figure 04).
6. Comparaison `m → a` : 0,39 pour `P` contre 0,07 pour `W` aléatoire.
7. La loss d'emma : 3,77 pour le réseau aléatoire, contre 2,51 pour la table.

**Messages clés.**
- On garde des **logits libres** dans `W` et on les transforme : l'optimiseur peut bouger les poids sans casser les contraintes.
- **Les logits sont des « log-comptes ».** La table de comptage est **un point particulier** de l'espace des `W`. L'entraînement doit le retrouver sans voir `N`. C'est la phrase qui donne son sens à tout le chapitre.
- Seules les **différences** de logits comptent.

**Piège.** La variable `counts = logits.exp()` n'est **pas** `N`. Le nom vient de l'analogie avec les log-comptes. Dites-le une fois clairement.

**Transition.** « La loss est calculable. Mais la calculer ne change pas `W`. Dans quel sens bouger chaque poids ? »

## 05 — Gradients et mise à jour (25 min)

**Déroulé.**
1. La case à lire par ligne.
2. `torch.arange(n)`.
3. Le piège `[:, ys]`, qui donne une forme `(5, 5)`.
4. La dérivée partielle.
5. `requires_grad` / `backward` / `W.grad` (figure 05).
6. La vérification par différence finie.
7. **Le gradient à la main**, `probs − one_hot(y)`.
8. La comparaison des learning rates sur un pas.
9. Un vrai pas.

**Messages clés.**
- `backward()` **ne modifie pas** `W` : il remplit `W.grad`. La mise à jour, c'est nous.
- Interprétation de `probs − one_hot(y)` : la bonne réponse monte de `1 − p` et chaque mauvaise réponse baisse de `p`. On punit surtout les erreurs **confiantes**.
- Seules les lignes `.`, `a`, `e`, `m` reçoivent un gradient : ce sont les seules **entrées** d'emma.

**Chiffres mesurés.**
- ∂L/∂W['.', 'e'] = −0,1975 (backward) contre −0,1976 (différence finie).
- Un pas sur emma depuis une loss de 3,77 : lr = 1 → 3,57 ; lr = 10 → 1,93 ; lr = 100 → 0,31 ; lr = 1000 → `nan`.

**Moment à exploiter.** Le public connaît les dérivées : faites dériver `−z_y + log Σ e^{z_k}` au tableau (2 lignes). Ensuite, l'assert « gradient à la main == W.grad » est très satisfaisant.

**Piège.** lr = 100 donne 0,31 sur emma : sur **5 exemples**, on apprend par cœur. Ne laissez pas croire qu'un grand lr est toujours meilleur. Le 06 montre l'effet sur le vrai corpus.

**Questions fréquentes.**
- *Pourquoi `torch.no_grad()` pour la mise à jour ?* La mise à jour n'est pas une opération du modèle : on ne veut pas l'enregistrer dans le graphe (et PyTorch refuse de modifier en place une feuille qui exige un gradient).
- *Pourquoi `W.grad = None` ?* PyTorch **additionne** les gradients à chaque `backward()`. Au chapitre 2, c'est `optimizer.zero_grad()`.

**Transition.** « Un pas sur un prénom. Maintenant : 100 pas sur 228 146 bigrammes. »

## 06 — Entraînement et comparaison (25 min)

**Déroulé.**
1. Tous les `xs` / `ys` (M = 228 146).
2. La boucle annotée (figure 06_boucle).
3. 4 learning rates.
4. Les courbes.
5. 300 pas de plus.
6. La comparaison des barres après `a` et des 3 heatmaps (comptage, réseau, différence).
7. La génération avec la même graine.
8. La régularisation.
9. La synthèse, la question finale et l'annexe.

**Chiffres mesurés (100 pas, même `W` initial, NLL sur tout le corpus).**

| learning rate | loss après 100 pas |
|---|---|
| 0,1 | 3,673 |
| 1 | 3,189 |
| 10 | 2,576 |
| 50 | 2,473 |
| 50, puis 300 pas de plus | **2,458** |
| comptage | 2,454 |
| uniforme | 3,296 |

- Écart absolu moyen entre les deux tables : **0,0017**. Les lignes les plus différentes sont `q`, `x` et `w`, les moins observées (`q` : 272 occurrences).
- Avec la **même graine**, les 10 prénoms générés sont **identiques** pour les deux modèles. C'est un effet garanti en classe.
- Régularisation, après 100 pas avec lr = 50 :
  - λ = 0 : NLL 2,473, max |W| ≈ 4 ;
  - λ = 0,1 : NLL 2,506 ;
  - λ = 1 : NLL 2,721, max |W| ≈ 2,4.

**Messages clés.**
- Le réseau retrouve la table **sans jamais voir `N`**, seulement grâce à la loss.
- La loss initiale (3,76) est **pire que l'uniforme** : un `W` aléatoire a des préférences arbitraires. Ce n'est pas un bug (même remarque au chapitre 2).
- La régularisation est le lissage du réseau. Elle devient le *weight decay* d'AdamW au chapitre 2.

**Moment à exploiter.** Avant d'exécuter les courbes, faites voter : « quel lr arrive le plus près du comptage en 100 pas ? ». Puis la question d'ouverture du 01 : *pourquoi apprendre une table qu'on sait calculer ?* La réponse est **27⁸ ≈ 2,8 × 10¹¹** lignes pour 8 caractères de contexte.

**Transition vers le chapitre 2.** « On garde les quatre invariants. Au chapitre 2, `one_hot @ W` devient `nn.Embedding`, la NLL devient `F.cross_entropy`, la mise à jour devient `optimizer.step()`. Surtout, on donne au modèle **plus d'un caractère de contexte**. »

**Réponses de l'annexe.**
1. `P[stoi['a']]` contient 27 valeurs de somme 1 : c'est la distribution des caractères qui suivent `a`, normalisée par ligne.
2. Pour 200 bigrammes, la forme est `(200, 27)`.
3. La NLL explose parce que −log p tend vers ∞ quand p tend vers 0.
4. On suit `−W.grad` parce que le gradient indique la direction où la loss **monte**.
5. Le gradient de la ligne `q` est nul parce que `q` n'est jamais une **entrée** dans emma (la ligne n'est pas sélectionnée par le one-hot).
6. La limite levée au chapitre 2 est le contexte d'un seul caractère.

---

## Différences avec `CHAPITRE_1.ipynb`, à connaître

**Contenu conservé :** toute la progression d'origine (sections 1 à 7 et l'annexe), réorganisée en 6 notebooks.

**Ajouts :**
- le piège `keepdim` ;
- `argmax` contre `multinomial` ;
- la comparaison avec le modèle uniforme ;
- la démonstration de l'underflow ;
- la perplexité ;
- le lissage et `andrejq` ;
- les propriétés du softmax ;
- `softmax(log N) = P` ;
- le gradient analytique ;
- la continuation à 400 pas ;
- la carte des différences ;
- la régularisation ;
- 6 schémas.

**Corrections de l'ancien notebook :**
- `plot_matrice_caracteres` utilisait `np` sans `import numpy`. La fonction plantait dès qu'on lui passait autre chose qu'un tenseur.
- La cellule `P[0]` était placée **avant** la définition de `P` : elle plantait si on exécutait le notebook dans l'ordre.
- Une cellule markdown était vide (après « requires_grad… »).
- Il y avait une coquille, « Moyennner ».
- La justification « on utilise le log parce que le Transformer le fait » a été remplacée par les vraies raisons.
- La variable `device` était définie mais jamais utilisée : elle a été supprimée.

## Références principales (pour vous)

- A. Karpathy, [*The spelled-out intro to language modeling: building makemore*](https://www.youtube.com/watch?v=PaCmpygFfXo). Ce chapitre suit sa progression : comptage, NLL, one-hot, softmax, gradient, régularisation.
- A. Karpathy, [*building micrograd*](https://www.youtube.com/watch?v=VMj-3S1tku0), pour la rétropropagation en profondeur.
- D. Jurafsky et J. H. Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), chapitre 3 : n-grammes, perplexité, lissage.
- I. Goodfellow, Y. Bengio et A. Courville, [*Deep Learning*](https://www.deeplearningbook.org/), sections 4.1 (softmax stable), 5.5 (maximum de vraisemblance), 6.2.2.3 (softmax), 6.5 (rétropropagation), 7.1 (pénalité L2).
- Y. Bengio et al., [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), 2003. Le pont historique vers le chapitre 2.
- C. E. Shannon, *A Mathematical Theory of Communication*, 1948. Les approximations de l'anglais lettre par lettre, une anecdote d'ouverture qui marche bien.
