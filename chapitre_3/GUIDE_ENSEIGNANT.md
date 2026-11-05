# Guide enseignant — Chapitre 3 : fine-tuner Mistral 7B

Ce document sert à **conduire** la séance. Les notebooks contiennent le cours complet. Le document de conception (`deep-research-report(1).md`, conservé en local et non publié) donne les justifications et les sources.

## Vue d'ensemble

| # | Notebook | Durée | Message en une phrase | Calcul |
|---|---|---|---|---|
| 01 | `01_pourquoi_finetuner` | 20 min | Mistral 7B **est** le GPT du chapitre 2, 34 000 fois plus grand ; le SFT, c'est la même cross-entropy, masquée sur la réponse. | CPU < 10 s |
| 02 | `02_donnees_dialogues` | 30 min | La tâche est définie par la construction des exemples ; on split **par scène**. | CPU ≈ 10 s |
| 03 | `03_lora_a_la_main` | 35 min | LoRA = `W₀ + (α/r)·BA`, B = 0 au départ ; il apprend un style avec 5 % des paramètres, mais ce n'est pas magique. | CPU ≈ 2 min (+ 3 min sans checkpoint) |
| 04 | `04_memoire_quantification` | 25 min | 81 Go → 5 Go : LoRA supprime gradients et AdamW, NF4 divise les poids par 4. | CPU < 10 s |
| 05 | `05_qlora_mistral_colab` | 40 min | QLoRA complet sur un T4 : avant / après, masque vérifié, coût mesuré. | GPU, voir ci-dessous |
| 06 | `06_evaluation_colab` | 25 min | Une loss qui baisse n'est pas un meilleur modèle : mémorisation, oubli, évaluation humaine. | GPU ≈ 5 min |
| 07 | `07_annexe_fusion_gguf` | devoir | Fusion, GGUF, llama.cpp, et la question des droits. | GPU / RAM élevée |

**Total : ≈ 3 h** sans l'annexe.

### Version en 2 heures (scénario du document de conception)

| Temps | Contenu |
|---|---|
| **0–5 min** | Les étudiants ouvrent le **notebook 05 sur Colab** (GPU T4) et exécutent les §0 à §3 (installation + téléchargement de 14,5 Go ≈ 5 min). Ça tourne pendant la théorie. |
| 5–20 min | 01 : §1 (même architecture), §3 (prompting / RAG / FT), §4 (masque). Passer vite sur le BPE. |
| 20–40 min | 02 : §1, §3 (la fuite à 96–100 % est **le** moment fort), §4, §5. |
| 40–60 min | 03 : §1–§2 au tableau, puis l'expérience §4 (avec le checkpoint fourni, 2 min de calcul) et le graphique de compromis. |
| 60–70 min | 04 : §1 (graphique 81 / 14 / 5 Go) et §4 (NF4, histogrammes). Le reste en lecture. |
| 70–105 min | 05 : §4 à §9 sur Kaamelott (entraînement ≈ quelques minutes). Les plus rapides relancent sur Platon. |
| 105–120 min | 06 : §1, §3, §5 + discussion des limites, présentation du devoir. |

## Mise en place

- Dépôt public : `https://github.com/zenocode-org/ai-masterclass` (dossier `chapitre_3/`). Les notebooks Colab clonent ce dépôt : **toute modification doit être poussée** avant le cours.
- Local (01–04) : `torch`, `transformers` (tokeniseur Mistral, ~2 Mo téléchargés), `matplotlib`, `numpy`, `peft` (facultatif, §6 du 03). Tout tourne sur CPU.
- Colab (05–07) : la première cellule installe des **versions figées** testées ensemble : `transformers==5.22.1 trl==1.17.1 peft==0.21.2 bitsandbytes==0.50.2 accelerate==1.15.0 datasets==5.1.0`. Si Colab change sa version de PyTorch et que l'installation échoue, retirer les numéros de version est la première chose à essayer.
- Modèle : `mistralai/Mistral-7B-Instruct-v0.3`, Apache 2.0, **non soumis à autorisation** sur Hugging Face (pas besoin de jeton). Variante plus rapide à télécharger (4 Go, déjà quantifiée) : `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` ; dans ce cas, garder `quantization_config` n'a pas d'effet, le modèle est déjà en 4 bits.

### ⚠️ À faire avant le premier cours : un run complet sur Colab

**Ce qui a été testé :**
- les notebooks 01 à 04 ont été exécutés intégralement (CPU), sorties sauvegardées ;
- les notebooks 05, 06 et 07 (§1) ont été exécutés de bout en bout sur GPU avec un **mini-Mistral aléatoire** (même architecture, même tokeniseur, 8 M de paramètres) : chargement 4 bits bitsandbytes, `get_peft_model`, `SFTTrainer`, masque de la loss, évaluation, sauvegarde et rechargement de l'adapter, `disable_adapter`, fusion. Les deux chemins de précision ont été testés : BF16 et **FP16 (celui du T4)**.

**Ce qui n'a pas été testé :** le run réel sur Mistral 7B (pas de GPU assez gros sur la machine de préparation) et les commandes llama.cpp de l'annexe 07 (§2–§3). Les durées GPU ci-dessous sont des **estimations**. Avant le cours :
1. lancer le 05 sur Kaamelott **et** sur Platon sur un T4 Colab ; noter la durée, la mémoire maximale, le débit (tokens/s) et remplacer les estimations de ce guide ;
2. vérifier que les commentaires « ce que l'on s'attend à voir » (§7–§8 du 05, §3 du 06) correspondent aux sorties ; les ajuster sinon ;
3. sauvegarder les adapters obtenus sur un Drive partagé : **plan B** si Colab refuse un GPU le jour J (quotas). Le 06 peut alors tourner à partir de cet adapter.

### Estimations GPU (à remplacer par vos mesures)

| Run | T4 (15 Go) | Mémoire attendue |
|---|---|---|
| chargement 4 bits | 2–5 min (téléchargement) | ≈ 4 Go |
| Kaamelott, 38 exemples × 3 époques (15 pas) | quelques minutes | 6–9 Go |
| Platon ou Shakespeare, 800 exemples × 1 époque (50 pas) | 15–25 min | 6–10 Go |
| 06 complet (40 exemples, 12 × 2 générations, hors domaine) | ≈ 5 min | ≈ 5 Go |

## Le fil rouge à répéter

> **Rien de nouveau dans l'architecture ni dans la loss.** Le chapitre 3 change trois choses : **ce qu'on met dans les exemples** (02), **quels paramètres bougent** (03), **comment ils sont stockés** (04).

---

## 01 — Pourquoi fine-tuner ? (20 min)

**Messages clés.**
- Le décompte des paramètres « à la main » tombe pile sur celui de PyTorch (7 248 023 552) ; 96 % sont dans les linéaires des blocs.
- Tableau de correspondance chapitre 2 → Mistral : faire retrouver aux étudiants `query`→`q_proj`, `FeedForward`→`gate/up/down_proj`.
- Masque `-100` : `F.cross_entropy` l'ignore par défaut. Dans l'exemple Kaamelott, 19 tokens sur 59 comptent.

**Chiffres mesurés.** Caractères par token BPE : Kaamelott 2,74, Platon 2,90, Shakespeare 3,11.

**Questions fréquentes.**
- *Pourquoi 8 têtes K/V pour 32 têtes Q ?* Grouped-query attention : moins de mémoire pour le cache KV à l'inférence, perte de qualité négligeable.
- *Le modèle connaît-il déjà Kaamelott ?* Peut-être un peu (le web en parle). C'est pour ça qu'on compare avant/après sur des scènes de validation.

## 02 — Données (30 min)

**Moment à exploiter.** Le taux de fuite d'un split au hasard : **100 %** (Kaamelott), **96 %** (Platon), **95 %** (Shakespeare), contre 0 %, 8 %, 0,2 % pour le split par groupe. Faire **prédire** avant d'exécuter.

**Chiffres mesurés.**

| Corpus | Groupes | Répliques | Exemples train / val | Tokens p50 / p99 | Tokens par époque |
|---|---|---|---|---|---|
| Kaamelott | 7 scènes | 55 | 38 / 10 | 131 / 181 | 4 433 |
| Platon | 45 dialogues | 11 179 | 8 335 / 1 013 | 185 / 530 | 1,75 M |
| Shakespeare | 178 blocs de 40 | 7 097 | 6 013 / 665 | 162 / 430 | 1,08 M |

Les 8 % de Platon sont des formules socratiques (« Sans doute. », « Tu as raison. ») : bon sujet de discussion sur les doublons légitimes.

**Kaamelott est minuscule** (extrait de 4 237 caractères, le même qu'au chapitre 2). Le dataset Kaggle complet (> 30 000 répliques, transcriptions Hypnoweb) est l'extension naturelle pour le devoir ; voir la remarque sur les droits plus bas.

## 03 — LoRA à la main (35 min)

**Déroulé.** Rang de `BA` → classe `LoRALinear` (sortie identique au départ) → mini-GPT pré-entraîné sur Platon (checkpoint fourni) → 4 variantes × 300 pas sur Kaamelott → graphique de compromis → fusion → `peft`.

**Chiffres mesurés (CPU, graine fixée).**

| Variante | Paramètres entraînés | Meilleure val Kaamelott | Val Platon après 300 pas |
|---|---|---|---|
| modèle Platon (départ) | 0 | 2,49 | 1,46 |
| de zéro | 218 098 | 2,15 | 2,85 |
| full fine-tuning (lr 1e-4) | 218 098 | **1,62** | **1,73** |
| LoRA q,v r=4 (lr 3e-3) | 10 240 | 1,83 | 1,84 |
| LoRA partout r=4 (lr 3e-3) | 27 648 | 1,69 | 2,15 |

**Point d'honnêteté à assumer devant les étudiants.** Ici, le full fine-tuning à petit lr **domine** LoRA sur les deux axes. C'est voulu de le montrer : LoRA n'est pas « meilleur », il est **moins cher** (mémoire, stockage). Le slogan « LoRA learns less and forgets less » (Biderman et al., 2024) vaut aux réglages usuels sur de grands modèles, et le lr compte autant que la méthode. Expérience proposée : LoRA q,v à lr 1e-3.

NF4 sur le même mini-GPT (notebook 04) : val Platon 1,450 (FP32) → 1,463 (INT4) → 1,462 (NF4).

## 04 — Mémoire et quantification (25 min)

**Chiffres (estimation, batch 1 × 512 tokens, r = 16 toutes linéaires).** Full FT 81 Go (dont 54 Go d'AdamW), LoRA 14,4 Go, QLoRA 4,9 Go. Ajouter 1 à 3 Go de surcoût CUDA réel. Le document de conception donne des fourchettes plus larges (10–16 Go pour QLoRA) parce qu'il inclut des batchs et séquences plus grands.

**Moment à exploiter.** L'exemple de la valeur aberrante : une seule constante pour 4 096 poids → 3 niveaux sur 16 utilisés, erreur × 20.

**Questions fréquentes.**
- *Pourquoi ne pas quantifier aussi les adapters ?* Ce sont eux qui reçoivent les mises à jour : en 4 bits, les petits pas d'AdamW disparaîtraient dans l'arrondi.
- *Le T4 n'a pas de BF16 ?* Exact (architecture Turing). Le 05 détecte la capacité du GPU et passe en FP16 ; les adapters sont alors gardés en FP32.

## 05 — QLoRA Mistral 7B (40 min)

**Points de vigilance.**
- Faire lancer les §0–§3 **en début de séance** (téléchargement).
- Le §6 (« ce qui compte dans la loss ») est la vérification à faire faire **systématiquement** : la moitié des fine-tunings ratés viennent d'un masque faux.
- La cellule après `SFTTrainer` reconvertit les adapters en FP32 sur T4 : TRL les passe en BF16 pour un modèle 4 bits, ce que le T4 ne gère pas.
- `save_strategy="no"` : pas de checkpoints intermédiaires (disque Colab), l'adapter est sauvegardé à la fin. Pour un long run, passer à `"steps"`.

**Pannes déjà rencontrées pendant la préparation :** aucune sur le mini-modèle. Tableau des pannes classiques en fin de notebook.

## 06 — Évaluation (25 min)

Messages : la perplexité est un diagnostic ; ROUGE-L punit les bonnes réponses différentes ; une citation de ≥ 50 caractères est un signal de mémorisation ; un fine-tuning de style peut éroder le comportement d'assistant et les refus (Qi et al., 2023). La grille aveugle (`evaluation/grille_<corpus>.csv`, clé séparée) sert pour le devoir.

## Droits et licences (à dire en cours)

- **Poids Mistral 7B v0.3** : Apache 2.0.
- **Kaamelott** : œuvre protégée. Le fichier du cours est un court extrait issu de transcriptions de fans ; le dataset Kaggle complet provient aussi de transcriptions communautaires (Hypnoweb), pas de scripts officiels. Usage pédagogique en classe ; ne pas redistribuer un adapter qui mémorise des répliques.
- **Platon** (traduction Victor Cousin, XIXᵉ s.) et **Shakespeare** : domaine public.

## Devoir (binômes, mini-rapport expérimental)

1. Construire **deux formulations** du dataset : réplique suivante (`mode="suite"`) et réponse d'un personnage donné (`mode="personnage"`, voir `ft.construire_exemples`).
2. Entraîner deux QLoRA en ne changeant **qu'une** variable : `r = 8` contre `r = 32`, ou 1 contre 3 époques.
3. Garder exactement le même split par scène / dialogue.
4. Mesurer : loss train / val, perplexité, ROUGE-L (à titre indicatif), évaluation humaine aveugle sur au moins 30 prompts.
5. Ajouter une mesure de mémorisation (§3 du 06).
6. Documenter : mémoire GPU maximale, durée, tokens traités, coût estimé.
7. Sélectionner cinq échecs et proposer une explication causale.
8. Exporter l'adapter, puis tester soit la fusion, soit la conversion GGUF (notebook 07).

**Extensions :** comparer `Mistral-7B-v0.3` (base) et `Mistral-7B-Instruct-v0.3` sur les mêmes données (l'instruction tuning aide-t-il ou contraint-il le style ?) ; changer de tâche avec le même corpus (« prédire le prochain locuteur », « identifier le personnage ») ; rechercher les quasi-doublons par n-grammes ou MinHash ; utiliser le dataset Kaggle complet de Kaamelott.
