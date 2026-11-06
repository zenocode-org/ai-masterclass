# Chapitre 3 — Fine-tuner Mistral 7B sur Kaamelott, Platon et Shakespeare

Suite du chapitre 2 (mini-GPT construit de zéro) : on part d'un modèle **pré-entraîné** de 7 milliards de paramètres et on l'adapte à nos trois corpus avec **QLoRA**.

| # | Notebook | Où | Durée | Contenu |
|---|---|---|---|---|
| 01 | `01_pourquoi_finetuner` | CPU | 20 min | Mistral 7B = le GPT du chapitre 2 à l'échelle ; BPE ; prompting / RAG / fine-tuning ; loss SFT masquée |
| 02 | `02_donnees_dialogues` | CPU | 30 min | répliques → exemples `(prompt, réponse)` ; split par scène ; longueurs en tokens ; export JSONL |
| 03 | `03_lora_a_la_main` | CPU | 35 min | LoRA implémenté à la main sur le mini-GPT : de zéro vs full FT vs LoRA, apprentissage vs oubli |
| 04 | `04_memoire_quantification` | CPU | 25 min | mémoire d'un entraînement, NF4 par blocs, QLoRA, accumulation de gradient |
| 05 | `05_qlora_mistral_colab` | **GPU** | 40 min | QLoRA de Mistral-7B-Instruct-v0.3 avec TRL, avant/après, coût, sauvegarde de l'adapter |
| 06 | `06_evaluation_colab` | **GPU** | 25 min | perplexité, ROUGE-L, mémorisation, grille humaine aveugle, questions hors domaine |
| 07 | `07_annexe_fusion_gguf` | GPU | annexe | `merge_and_unload`, conversion GGUF, llama.cpp |

## Lancer sur Google Colab

Chaque notebook a un bouton « Open in Colab ». Pour 05, 06 et 07 : *Exécution → Modifier le type d'exécution → GPU (T4)*. La première cellule installe les dépendances (versions figées, testées ensemble) et clone ce dépôt pour récupérer `finetune.py` et `data/`.

| Notebook | Lien |
|---|---|
| 01 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/01_pourquoi_finetuner.ipynb) |
| 02 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/02_donnees_dialogues.ipynb) |
| 03 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/03_lora_a_la_main.ipynb) |
| 04 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/04_memoire_quantification.ipynb) |
| 05 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/05_qlora_mistral_colab.ipynb) |
| 06 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/06_evaluation_colab.ipynb) |
| 07 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/chapitre_3/07_annexe_fusion_gguf.ipynb) |

## En local (notebooks 01 à 04)

```bash
uv sync            # à la racine du dépôt : crée .venv avec torch, transformers, peft…
uv run jupyter lab
```

Voir la section « Exécuter les notebooks » du [README principal](../README.md).

Les notebooks 01 à 04 ont été exécutés et sauvegardés **avec leurs sorties** (CPU). Les notebooks 05 à 07 sont **sans sorties** : ils demandent un GPU.

## Fichiers

- `finetune.py` : code partagé (lecture des corpus, construction des exemples, `LoRALinear`, mémoire, NF4, génération, perplexité). Comme `minigpt.py` au chapitre 2, il contient exactement le code expliqué dans les notebooks.
- `data/<corpus>_{train,val}.jsonl` : exemples produits par le notebook 02 (split par scène / dialogue).
- `checkpoints/minigpt_platon.pt` : le mini-GPT pré-entraîné sur Platon (notebook 03), pour éviter 3 min de calcul.
- `figures/` : schémas (SVG sources + PNG).
- `GUIDE_ENSEIGNANT.md` : déroulé, timings, chiffres mesurés, points de vigilance, devoir.

Les corpus sont lus dans le dossier parent : `shakespeare.txt`, `script-kaamelott.txt`, `plato_train.txt`.
