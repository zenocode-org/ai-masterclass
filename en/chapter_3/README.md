# Chapter 3 — Fine-tuning Mistral 7B on Kaamelott, Plato and Shakespeare

A follow-up to chapter 2 (a mini-GPT built from scratch): we start from a **pre-trained** 7-billion-parameter model and adapt it to our three corpora with **QLoRA**.

| # | Notebook | Where | Time | Content |
|---|---|---|---|---|
| 01 | `01_why_finetune` | CPU | 20 min | Mistral 7B = the GPT from chapter 2 at scale; BPE; prompting / RAG / fine-tuning; masked SFT loss |
| 02 | `02_dialogue_data` | CPU | 30 min | lines of dialogue → `(prompt, response)` examples; split by scene; lengths in tokens; JSONL export |
| 03 | `03_lora_by_hand` | CPU | 35 min | LoRA implemented by hand on the mini-GPT: from scratch vs full FT vs LoRA, learning vs forgetting |
| 04 | `04_memory_and_quantization` | CPU | 25 min | memory of a training run, block-wise NF4, QLoRA, gradient accumulation |
| 05 | `05_qlora_mistral_colab` | **GPU** | 40 min | QLoRA of Mistral-7B-Instruct-v0.3 with TRL, before/after, cost, saving the adapter |
| 06 | `06_evaluation_colab` | **GPU** | 25 min | perplexity, ROUGE-L, memorization, blind human grid, out-of-domain questions |
| 07 | `07_appendix_merge_gguf` | GPU | appendix | `merge_and_unload`, GGUF conversion, llama.cpp |

## Running on Google Colab

Each notebook has an "Open in Colab" button. For 05, 06 and 07: *Runtime → Change runtime type → GPU (T4)*. The first cell installs the dependencies (pinned versions, tested together) and clones this repository to get `finetune.py` and `data/`.

| Notebook | Link |
|---|---|
| 01 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/01_why_finetune.ipynb) |
| 02 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/02_dialogue_data.ipynb) |
| 03 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/03_lora_by_hand.ipynb) |
| 04 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/04_memory_and_quantization.ipynb) |
| 05 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/05_qlora_mistral_colab.ipynb) |
| 06 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/06_evaluation_colab.ipynb) |
| 07 | [Colab](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/07_appendix_merge_gguf.ipynb) |

## Locally (notebooks 01 to 04)

```bash
uv sync            # at the repository root: creates .venv with torch, transformers, peft…
uv run jupyter lab
```

See the "Running the notebooks" section of the [course README](../README.md).

Notebooks 01 to 04 were run and saved **with their outputs** (CPU). Notebooks 05 to 07 have **no outputs**: they need a GPU.

## Files

- `finetune.py`: shared code (reading the corpora, building the examples, `LoRALinear`, memory, NF4, generation, perplexity). Like `minigpt.py` in chapter 2, it contains exactly the code explained in the notebooks.
- `data/<corpus>_{train,val}.jsonl`: examples produced by notebook 02 (split by scene / dialogue).
- `checkpoints/minigpt_plato.pt`: the mini-GPT pre-trained on Plato (notebook 03), to save 3 min of computation.
- `figures/`: diagrams (SVG sources + PNG).
- `TEACHER_GUIDE.md`: lesson plan, timings, measured numbers, pitfalls, homework.

The corpora are read from the repository's `corpus/` folder: `shakespeare.txt`, `script-kaamelott.txt`, `plato_train.txt`.
