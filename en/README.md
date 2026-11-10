# AI Masterclass — from bigrams to fine-tuning an LLM

A course in Jupyter notebooks to **understand language models by building them**.

> *Version française : [`fr/README.md`](../fr/README.md). Repository overview: [`README.md`](../README.md).*

The course has three stages. You start by counting letters in first names. You then write a mini-GPT from scratch. You finish by adapting a real 7-billion-parameter model (Mistral 7B) to Kaamelott, Plato or Shakespeare.

> **The common thread of the three chapters.** A language model always does the same thing: it **predicts the next token**. It produces **logits**, the **softmax** turns them into probabilities, the **cross-entropy** (NLL) measures the error, and **gradient descent** corrects the weights. These four steps never change. From one chapter to the next, only **the way the logits are computed** changes.

> **About the corpora.** The course was first written in French. Two of its three text corpora are French: *Kaamelott* (a French TV series) and Plato's dialogues in Victor Cousin's French translation. They are kept as is in this English version, so the numbers match the French course. Shakespeare is always available as an English alternative, and the code works the same on any text.

---

## 1. Where to start?

| If you… | Start with |
|---|---|
| are new to the topic | `chapter_1/01_counting_bigrams.ipynb`, then follow the order |
| already know softmax, cross-entropy and gradient descent | `chapter_2/01_datasets.ipynb` (a recap of chapter 1 is in `chapter_2/02`) |
| only want fine-tuning | `chapter_3/README.md` (read at least `chapter_2/05` on attention first) |

**Prerequisites:** Python, a little NumPy or PyTorch, matrices (matrix product) and derivatives. No prior deep learning knowledge is needed.

**Hardware:** an ordinary laptop is enough for chapters 1 and 2 and for notebooks 01 to 04 of chapter 3. Everything runs **on CPU**. Only notebooks 05 to 07 of chapter 3 need a GPU, the free one from Google Colab.

---

## 2. Layout

```text
en/
├── README.md                  ← you are here
├── chapter_1/                 ← from bigrams to a first neural network (6 notebooks)
│   ├── 01_… 06_*.ipynb
│   ├── bigram.py              ← the chapter's code, reused from one notebook to the next
│   ├── figures/               ← diagrams (SVG + PNG)
│   └── TEACHER_GUIDE.md
│
├── chapter_2/                 ← from the bigram to a mini-GPT (9 notebooks)
│   ├── 01_… 09_*.ipynb
│   ├── minigpt.py             ← tokenizer, attention, blocks, GPTLanguageModel…
│   ├── slides_encoder_decoder.md     ← slides (Marp) between notebooks 03 and 04
│   ├── figures/
│   └── TEACHER_GUIDE.md
│
├── chapter_3/                 ← fine-tuning Mistral 7B with LoRA / QLoRA (7 notebooks)
│   ├── 01_… 07_*.ipynb
│   ├── finetune.py            ← data, LoRA, quantization, evaluation
│   ├── data/                  ← (prompt, response) examples in JSONL, produced by notebook 02
│   ├── checkpoints/           ← mini-GPT pre-trained on Plato (notebook 03)
│   ├── figures/
│   ├── README.md              ← "Open in Colab" links
│   └── TEACHER_GUIDE.md
│
└── references/                ← study notes on research papers and blog posts

../corpus/                     ← the corpora, shared by the French and English versions
├── names.txt                  ← 32,000 first names (chapter 1)
├── shakespeare.txt            ← Tiny Shakespeare, 1.1 M characters (chapters 2 and 3)
├── kaamelott.txt              ← the complete Kaamelott series, 2.4 M characters, French (chapter 2)
├── script-kaamelott.txt       ← a short Kaamelott excerpt, French (chapter 3)
├── plato_train.txt            ← Plato's dialogues, transl. Victor Cousin, 3.9 M characters, French (chapters 2 and 3)
└── preparer_kaamelott.py      ← rebuilds kaamelott.txt from the Kaggle dataset
```

**Three rules to find your way:**
1. **Notebooks are numbered**: follow the order `01, 02, 03…`. Each notebook starts with what you will be able to do by the end, and ends with a summary, check-your-understanding questions and references.
2. **The `.py` files contain nothing new.** `bigram.py`, `minigpt.py` and `finetune.py` store code **already explained** in the notebooks, so that later notebooks import it instead of copying it. If you want to know where a function comes from, the file header says which notebook builds it.
3. **The corpora live in `corpus/`, at the repository root.** The modules look for them next to the notebook, then in that folder: you have nothing to move.

The `TEACHER_GUIDE.md` files are for teachers (lesson plan, timings, measured numbers). You can read them: they contain the answers to the check-your-understanding questions.

---

## 3. Running the notebooks

Two options: **Google Colab** (nothing to install) or **your machine** with `uv`.

Every notebook ships **with its outputs**, except 05 to 07 of chapter 3, which need a GPU. You can read them like a book before running them.

### Option A — Google Colab (recommended to get started)

Click a link below, or the "Open in Colab" button at the top of each notebook. Then run *Runtime → Run all*.

| Chapter | Notebooks |
|---|---|
| 1 — Bigrams | [01](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_1/01_counting_bigrams.ipynb) · [02](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_1/02_measuring_the_loss.ipynb) · [03](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_1/03_one_hot_and_weights.ipynb) · [04](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_1/04_softmax_forward_pass.ipynb) · [05](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_1/05_gradients_and_update.ipynb) · [06](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_1/06_training_and_comparison.ipynb) |
| 2 — Mini-GPT | [01](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_2/01_datasets.ipynb) · [02](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_2/02_bigram_nn_module.ipynb) · [03](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_2/03_batch_learning.ipynb) · [04](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_2/04_transformer_architecture.ipynb) · [05](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_2/05_attention.ipynb) · [06](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_2/06_feedforward.ipynb) · [07](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_2/07_stacking_layers_and_backprop.ipynb) · [08](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_2/08_gpt_language_model.ipynb) · [09](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_2/09_training_and_evaluation.ipynb) |
| 3 — Fine-tuning | [01](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/01_why_finetune.ipynb) · [02](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/02_dialogue_data.ipynb) · [03](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/03_lora_by_hand.ipynb) · [04](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/04_memory_and_quantization.ipynb) · [05](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/05_qlora_mistral_colab.ipynb) · [06](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/06_evaluation_colab.ipynb) · [07](https://colab.research.google.com/github/zenocode-org/ai-masterclass/blob/main/en/chapter_3/07_appendix_merge_gguf.ipynb) |

**How it works.** Colab only opens the `.ipynb` file, on an empty machine. The **first code cell** of each notebook detects Colab and clones this repository. It thereby fetches the corpora and the modules (`bigram.py`, `minigpt.py`, `finetune.py`), then moves into the chapter folder (`en/chapter_N`). Locally, this cell does nothing. There is **nothing to copy to Google Drive**.

- PyTorch, NumPy, Matplotlib and Transformers are already installed on Colab. Notebooks 05 to 07 of chapter 3 install their extra libraries themselves (TRL, bitsandbytes…).
- For notebooks 05 to 07 of chapter 3: *Runtime → Change runtime type → GPU (T4)*. The others run on the free CPU.
- Your changes are lost when the session closes. To keep them: *File → Save a copy in Drive*.
- After a session restart, run the notebook again from the first cell.

### Option B — On your machine, with uv

[uv](https://docs.astral.sh/uv/) creates the Python environment and installs the right versions in one command. `pyproject.toml` lists the dependencies and `uv.lock` pins their versions.

```bash
# 1. install uv (once): https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh        # Windows: see the link above

# 2. get the course and create the environment (.venv/)
git clone https://github.com/zenocode-org/ai-masterclass.git
cd ai-masterclass
uv sync

# 3. start Jupyter
uv run jupyter lab
```

The environment contains PyTorch, NumPy, Matplotlib, Transformers, PEFT and JupyterLab: enough to run chapters 1 and 2 and notebooks 01 to 04 of chapter 3. To also run notebooks 05 to 07 outside Colab, you need an NVIDIA GPU and `uv sync --group gpu`.

In VS Code, pick the folder's `.venv` interpreter as the notebook kernel.

**Without uv:** `python -m venv .venv`, activate it, then `pip install torch numpy matplotlib transformers peft accelerate jupyterlab`. On Linux without an NVIDIA GPU, first install the CPU build of PyTorch, which is much lighter (the default build ships CUDA, several GB): `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

---

## 4. Chapter 1 — From bigrams to a first neural network

**Central question:** can we predict the next letter of a first name? We first do it **by counting**, then by **learning** a weight matrix. Both paths lead to the same table.

**Corpus:** `names.txt`. **Duration:** about 2 h 30. **Everything runs on CPU** (about 2 minutes of compute for notebook 06).

| # | Notebook | What you learn |
|---|---|---|
| 01 | `01_counting_bigrams` | Count transitions in a table `N` (27 × 27), normalize each row into probabilities, sample with `torch.multinomial`, generate names. The *broadcasting* trap (`keepdim=True`). |
| 02 | `02_measuring_the_loss` | Scoring a model: likelihood → log-likelihood → **mean NLL**. Why the logarithm, perplexity, and why a zero probability makes the loss infinite (smoothing). |
| 03 | `03_one_hot_and_weights` | Encode a character as **one-hot**. `one_hot @ W` amounts to reading a row of `W`. We get 27 free scores, the **logits**. |
| 04 | `04_softmax_forward_pass` | The **softmax** (exponential then normalization) and its properties. `softmax(log N) = P`. The complete *forward pass*. |
| 05 | `05_gradients_and_update` | `loss.backward()`, `W.grad`, the finite-difference check, the gradient computed by hand (`probs − one_hot(y)`), the *learning rate*. |
| 06 | `06_training_and_comparison` | The complete training loop. The network recovers the count table (loss 2.458 vs 2.454). Regularization, a cousin of smoothing. |

**Takeaway:** logits → softmax → NLL → gradient. These four steps stay the same all the way to Mistral 7B.

---

## 5. Chapter 2 — From the bigram to a mini-GPT

**Central question:** a bigram only sees **one** character. How do we give each position access to **the whole past**, without counting every combination? The answer is **attention**, then the full Transformer. The chapter follows Andrej Karpathy's video *Let's build GPT*.

**Corpus:** Shakespeare, Kaamelott, Plato (your choice). **Duration:** about 4 h 45, over two sessions. **CPU:** about 2 minutes of training in notebook 09.

| # | Notebook | What you learn |
|---|---|---|
| 01 | `01_datasets` | The three corpora. The character tokenizer (`encode` / `decode`). Text becomes a tensor. The train / validation split. The baseline loss `log V`. |
| 02 | `02_bigram_nn_module` | Chapter 1's bigram rewritten in PyTorch: `nn.Module`, `nn.Embedding` (= `one_hot @ W`), `F.cross_entropy`, **SGD vs AdamW**. |
| 03 | `03_batch_learning` | Mini-batches `(B, T)`. Why we reshape `(B, T, V)` into `(B·T, V)` before the loss. |
| — | `slides_encoder_decoder.md` | 10 slides: the 2017 encoder-decoder architecture, and why GPT only keeps the decoder. |
| 04 | `04_transformer_architecture` | The map of the *decoder-only* Transformer: what we are going to build, piece by piece. |
| 05 | `05_attention` | **The heart of the course.** Averaging the past, the causal mask, the masked softmax, **Q / K / V**, the `√d` factor, multi-head attention. |
| 06 | `06_feedforward` | The network applied at each position, ReLU / GELU, **dropout** (`train` mode vs `eval` mode). |
| 07 | `07_stacking_layers_and_backprop` | How layers are chained and influence each other during backpropagation. **Residual connections** and **LayerNorm**, shown by experiment. |
| 08 | `08_gpt_language_model` | The complete model: position embedding, blocks, `lm_head`, causality test, `generate`. |
| 09 | `09_training_and_evaluation` | Training (validation loss 1.91 vs 2.48 for the bigram), perplexity, loss per position, learned attention maps, temperature, then experiments to run yourself. |

**Takeaway:** **attention is communication between positions; the FeedForward is computation at each position.** Residuals and LayerNorm make it possible to stack blocks.

---

## 6. Chapter 3 — Fine-tuning Mistral 7B with QLoRA

**Central question:** you do not retrain a 7-billion-parameter model on a laptop. How do you **adapt** it to a style (Kaamelott lines, Plato's dialogues) with little memory? The answer comes down to two ideas: **LoRA** (only train small matrices added to the model) and **4-bit quantization** (QLoRA).

**Hardware:** CPU for notebooks 01 to 04, Colab GPU (free T4) for 05 to 07. The Colab links and details are in [`chapter_3/README.md`](chapter_3/README.md).

| # | Notebook | What you learn |
|---|---|---|
| 01 | `01_why_finetune` | Mistral 7B is chapter 2's GPT, much bigger. BPE tokens. Prompting, RAG or fine-tuning? The loss restricted to the response (`-100` mask). |
| 02 | `02_dialogue_data` | Turn lines of dialogue into `(prompt, response)` examples, split by scene, export to JSONL. |
| 03 | `03_lora_by_hand` | LoRA coded by hand on chapter 2's mini-GPT: training from scratch, full fine-tuning, LoRA. What is learned and what is forgotten. |
| 04 | `04_memory_and_quantization` | Where the memory goes during training. NF4 quantization. Gradient accumulation. |
| 05 | `05_qlora_mistral_colab` | QLoRA of Mistral-7B-Instruct with TRL. Before / after comparison. |
| 06 | `06_evaluation_colab` | Perplexity, ROUGE-L, memorization, blind human evaluation. |
| 07 | `07_appendix_merge_gguf` | Merge the adapter, convert to GGUF, run the model with llama.cpp. |

---

## 7. What changes from one chapter to the next

| | Chapter 1 | Chapter 2 | Chapter 3 |
|---|---|---|---|
| Token | one character (27) | one character (65 to 116) | one BPE subword (32,768) |
| Context | 1 character | 64 characters | thousands of tokens |
| Computing the logits | one row of `W` | embeddings + 4 attention blocks | 32 pre-trained blocks + small LoRA adapters |
| Trained parameters | 729 | ≈ 210,000 | ≈ 42 million, i.e. 0.6% of the 7 billion |
| Loss | NLL | `F.cross_entropy` | `F.cross_entropy`, on the response only |
| Update | `W -= lr * W.grad` | AdamW | paged 8-bit AdamW (`paged_adamw_8bit`) |

---

## 8. Tips for working

- **Predict before you run.** Many cells ask a question ("what shape?", "which optimizer wins?"). Answer it before running: that is where the learning happens.
- **Watch the shapes.** Almost every bug is an unexpected tensor shape. Print `x.shape` freely.
- **Change one parameter at a time**, and compare the **validation loss** rather than eyeballing the quality of generated text. Notebook 09 of chapter 2 offers a table of experiments.
- **Restart the kernel** (*Kernel → Restart*) if a notebook behaves strangely: notebooks assume they are run in order, from the top.
- **The diagrams** (`figures/`) summarize each concept: they make good revision cards.

### Mini-glossary

| Term | In one sentence |
|---|---|
| **token** | the unit the model reads and predicts (here a character, then a subword) |
| **logits** | free scores, one per possible token, before the softmax |
| **softmax** | exponential then division by the sum: scores become probabilities |
| **NLL / cross-entropy** | −log of the probability given to the correct answer, averaged: the loss |
| **perplexity** | `exp(loss)`: the number of choices the model "hesitates" between on average |
| **gradient** | the slope of the loss with respect to each weight; `backward()` computes it |
| **embedding** | a table where each token reads its vector (`nn.Embedding`) |
| **attention** | each position chooses (Q·K) which past positions to consult, and retrieves their content (V) |
| **residual connection** | `x + f(x)`: the block adds a correction instead of replacing everything |
| **LoRA** | train only two small matrices `A·B` added to frozen weights |
| **quantization** | store the weights on fewer bits (4 instead of 16) to save memory |

---

## 9. Going further

- Andrej Karpathy, [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html) series, in particular *makemore* (chapter 1) and *Let's build GPT* (chapter 2).
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
- J. Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/).
- 3Blue1Brown, [series on neural networks and Transformers](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).
- Hu et al., [*LoRA*](https://arxiv.org/abs/2106.09685), 2021 · Dettmers et al., [*QLoRA*](https://arxiv.org/abs/2305.14314), 2023.
- D. Jurafsky and J. H. Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) (free online book).

Each notebook ends with its own, more specific references.

---

## 10. Data and rights

The files are in [`corpus/`](../corpus/), at the repository root.

| File | Content | Source | Status |
|---|---|---|---|
| `names.txt` | 32,033 first names | [makemore](https://github.com/karpathy/makemore) (A. Karpathy) | MIT license |
| `shakespeare.txt` | *Tiny Shakespeare* | [char-rnn](https://github.com/karpathy/char-rnn) (A. Karpathy) | public domain |
| `plato_train.txt` | Plato's dialogues, French translation by Victor Cousin | [Wikisource](https://fr.wikisource.org/wiki/Auteur:Platon) | public domain |
| `kaamelott.txt` | the 455 episodes of *Kaamelott*, one line per row `CHARACTER: line` | [Kaggle dataset](https://www.kaggle.com/datasets/jeanmidev/kaamelott-tv-show-transcripts-fr), formatted by `preparer_kaamelott.py` | copyrighted work: educational use only |
| `script-kaamelott.txt` | the first lines of the first episode | same source | copyrighted work: educational use only |

*Kaamelott* is a work by Alexandre Astier, protected by copyright. The transcripts are used here for teaching only. Do not redistribute a model trained on them, especially if it can recite lines (see the memorization test in chapter 3, notebook 06).
