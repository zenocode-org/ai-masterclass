# AI Masterclass — du bigramme au fine-tuning d'un LLM

Cours en notebooks Jupyter (en français) pour comprendre les modèles de langage en les construisant.

| Chapitre | Dossier | Contenu | Matériel |
|---|---|---|---|
| 1 | `CHAPITRE_1.ipynb` | Bigrammes sur des prénoms, premiers poids appris | CPU |
| 2 | `chapitre_2/` | Du bigramme au mini-GPT (attention, FeedForward, entraînement), 9 notebooks | CPU |
| 3 | `chapitre_3/` | Fine-tuning de Mistral 7B avec LoRA / QLoRA, 7 notebooks | CPU (01–04) + GPU Colab (05–07) |

Chaque chapitre a un `GUIDE_ENSEIGNANT.md`. Le chapitre 3 se lance sur Google Colab depuis les liens de `chapitre_3/README.md`.

## Données

| Fichier | Contenu | Statut |
|---|---|---|
| `names.txt` | prénoms (chapitre 1) | — |
| `shakespeare.txt` | *Tiny Shakespeare* (A. Karpathy) | domaine public |
| `plato_train.txt` | Dialogues de Platon, traduction de Victor Cousin | domaine public |
| `script-kaamelott.txt` | court extrait de transcriptions de *Kaamelott* | œuvre protégée : usage pédagogique uniquement, ne pas redistribuer de modèle qui le mémorise |
