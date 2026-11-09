# Teacher guide — Chapter 2: from the bigram to the mini-GPT

This document is for **running** the session: order, timings, key messages, transitions, pitfalls and answers to questions. The notebooks contain the full course. Here, only what you need as a teacher.

## Overview

| # | Notebook | Time | One-sentence message | CPU compute |
|---|---|---|---|---|
| 01 | `01_datasets` | 20 min | A model imitates **one** text; the text becomes a sequence of integers. | < 5 s |
| 02 | `02_bigram_nn_module` | 35 min | The chapter 1 bigram rewritten in PyTorch; AdamW > SGD because it normalizes the step size. | < 5 s |
| 03 | `03_batch_learning` | 25 min | `(B, T)` = B·T exercises; reshape to `(B·T, V)` without changing a single value. | < 5 s |
| — | `slides_encoder_decoder.md` | 10 min | What we do not implement: the encoder and cross-attention. | — |
| 04 | `04_transformer_architecture` | 20 min | The map: input and output = bigram, only the middle changes. | < 5 s |
| 05 | `05_attention` | 50 min | Masked softmax + Q·K to choose where to look, V for what to retrieve. | < 5 s |
| 06 | `06_feedforward` | 25 min | Attention = communication, FeedForward = computation; ReLU is essential; dropout. | < 5 s |
| 07 | `07_stacking_layers_and_backprop` | 40 min | Layers learn together; residuals let the gradient through. | ≈ 35 s |
| 08 | `08_gpt_language_model` | 25 min | Assembly + position embedding; the full model is causal. | < 5 s |
| 09 | `09_training_and_evaluation` | 35 min | Train/val curves, loss per position, heads, generation, temperature. | ≈ 2 min |

**Total: ≈ 4 h 45.** Suggested split into two sessions:
- **Session A (≈ 2 h 15)**: 01 → 02 → 03 → slides → 04. End on the architecture map: "next time, we build the middle".
- **Session B (≈ 2 h 30)**: 05 → 06 → 07 → 08 → 09. Start the training in 09 **at the beginning of the 09 explanation** (it runs while you comment on the loop).

If you only have 2 × 2 h: shorten 01 (10 min), go quickly over the "AdamW by hand" check in 02, and skip §5 of 07 ("one layer influences the others"), giving it as reading.

## Setup

- Folder: `en/chapter_2/`. The notebooks import `minigpt.py` (same folder). The code of this module is exactly the code built in the notebooks: it avoids copying the classes from one notebook to the next.
- Data: `shakespeare.txt` and `kaamelott.txt` are in the repository's `corpus/` folder. `kaamelott.txt` (≈ 2.4M characters, 455 episodes) is generated once and for all from the CSV of the [Kaggle dataset](https://www.kaggle.com/datasets/jeanmidev/kaamelott-tv-show-transcripts-fr) by `corpus/preparer_kaamelott.py`; the old excerpt `script-kaamelott.txt` (4,237 characters) is no longer used. `plato_train.txt` is also in `corpus/` (failing that, it is looked up in `philo-masterclass/data/`, next to the repository).
- Dependencies: `torch`, `matplotlib`, `numpy`. Everything runs on CPU; `minigpt` limits PyTorch to 8 threads (beyond that, it is slower on hybrid laptops).
- The diagrams are in `figures/` (SVG sources and their PNG renders); the notebooks load the PNGs from GitHub. Two images come from Wikimedia. Both require a connection (credits are under the images).
- All notebooks have been executed and saved **with their outputs**: you can project them without re-running anything.

## The common thread to repeat

> **The problem never changes:** predict the next character. The loss (cross-entropy) and the loop (forward → zero_grad → backward → step) are the same from the bigram to the GPT. **Only the way the logits are produced changes.**

And Karpathy's line for the block: **attention is communication; the FeedForward is computation.**

---

## 01 — Datasets (20 min)

**Flow.** Previews of the three corpora → comparison table → character frequencies → tokenizer → `data` → train/val split → `log V`.

**Key messages.**
- Spaces and `\n` are characters: the model learns the layout.
- An index is a **label**, not a quantity (sets up the embedding).
- Kaamelott (the full series) is ≈ 2.4M characters, more than twice Shakespeare; Plato ≈ 3.9M.

**Moment to exploit.** Kaamelott contains **two apostrophes**, `'` (≈ 16,000) and `’` (≈ 19,000), depending on the episode: for the tokenizer, these are two unrelated symbols, and the model will have to learn "c'est" twice. A great example of "the model only sees symbols". Its vocabulary (V = 116, with digits and rare signs) also gives a higher uniform loss: log 116 ≈ 4.75.

**Frequent questions.**
- *Why not words?* Huge vocabulary (25,000 distinct words in Shakespeare versus 65 characters), and unknown words are impossible to handle. Real GPTs use sub-words (BPE), with the same model behind.
- *Why split in order?* A random split would leave the neighbors of every validation character in the training set: an overly optimistic evaluation.

**Transition.** "We have integers. The simplest model that predicts the next integer, you already know it: the bigram."

## 02 — Bigram as an `nn.Module` (35 min)

**Flow.** `nn.Module` → `nn.Embedding == one_hot @ W` → the `BigramLanguageModel` class → `F.cross_entropy` (by hand vs the function, then stability) → 120 steps with AdamW lr=1e-3 (**nothing moves**) → explanation of SGD / Adam / AdamW → checking one AdamW step by hand → gradient scale → 6-optimizer experiment → generating 120 characters.

**Key messages.**
- `nn.Embedding(V, V)` **is** the `W` matrix from chapter 1 (assert on screen).
- `F.cross_entropy` takes **logits** and **indices**; it is softmax + NLL, done stably.
- **AdamW**: the step is ≈ `lr` whatever the size of the gradient (demonstration: gradient = 3, step = 0.1).

**Measured numbers (Shakespeare, 120 steps, 256 pairs/step, same seed).**

| Optimizer | val loss |
|---|---|
| SGD lr=1e-3 | 4.72 (no learning) |
| SGD lr=1 | 4.05 |
| SGD lr=10 | 2.89 |
| AdamW lr=1e-3 | 4.58 |
| AdamW lr=1e-2 | 3.57 |
| AdamW lr=1e-1 | **2.54** |
| best bigram (counts) | 2.48 |

**Moment to exploit.** Have the class **predict** before running the experiment: "which optimizer wins? does it need the same lr?". Most people think SGD lr=1e-3 "will work a bit". Then show the "gradient scale" cell: mean |gradient| ≈ 4·10⁻⁴, 27 rows out of 65 with no gradient in a batch → SGD needs lr=10.

**Pitfall.** The initial loss (4.72) is **higher** than `log V` (4.17): the `N(0,1)` init gives noisy logits, hence a model that is wrongly confident. It is not a bug.

**Transition.** "The bigram plateaus at 2.48 because it only sees one character. To give it context, we need to give it **chunks of text**, not pairs."

## 03 — Batch learning (25 min)

**Flow.** Why mini-batches (noise histograms) → x/y windows (the `"GOODBYE!"` example) → `get_batch` → `F.cross_entropy` **error** on `(B, T, V)` → reshape (image `03_reshape.png` + numerical figure) → check "mean of the B·T losses" → sequence version of the bigram.

**Key messages.**
- A window of T characters = **T exercises** (context from 1 to T).
- The B rows of a batch **never** communicate.
- The reshape does **neither a sum nor a permutation**: `(b, t)` → row `b·T + t`, for the logits **and** the targets.

**Moment to exploit.** The deliberate error: `Expected target size [4, 65], got [4, 8]`. Ask: "why is PyTorch talking about 65?" → it takes dimension 1 as the classes.

**Answer to the notebook question.** `torch.randint(len(data) - block_size)`: if we started less than T+1 characters from the end, `y` would be too short.

**Transition.** Project the encoder-decoder slides, then 04.

## Encoder-decoder slides (10 min) then 04 — Architecture (20 min)

The slides (`slides_encoder_decoder.md`, Marp format, speaker notes included) answer "do we need a deck just for this?". Yes, but a short one: 10 slides. The central slide is the **table of the three attentions** (where Q, K, V come from; mask or not).

**04, key messages.**
- GPT = decoder-only: no source → no encoder → no cross-attention.
- Input, output and loss = bigram; only **the middle** changes.
- Parameter breakdown (≈ 211,000): the FeedForward weighs **twice** as much as attention in each block (15.6% versus 7.8%).
- The "2017 vs mini-GPT" table: pre-norm, learned positions, characters. The principle does not change.

**Pointer.** Karpathy's video, chapter *encoder vs. decoder vs. both (?) Transformers* (towards the end of the video).

## 05 — Attention (50 min, core of the chapter)

**Flow.**
1. Cumulative average with loops (5 min).
2. The `A @ B` matrix trick then the **masked softmax** (10 min). Insist: why this detour? Because the scores will not stay at zero.
3. Causal mask (3-panel figure): the future = the answer.
4. **The `chat` example computed by hand** (10 min). Have the class compute row `a` on the board before running: e¹, e³, e¹ → 0.11 / 0.79 / 0.11. The value (5, 5) of `t` leaks nowhere.
5. Q, K, V with the library analogy, then the **"previous character" head built by hand** (10 min). Message: Q/K can express a rule; training finds its own rules.
6. Real tensors: shapes, transposing K, `weights @ V`, the √hs factor (variance 18.4 → 1.15).
7. The `Head` class + **causality test** (positions 0–6 unchanged).
8. Multi-head: figure, code, number of parameters (16,448).

**If you are short on time.** §5 (real tensors) can be skimmed: the hand-computed example and the hand-built head carry the essentials.

**Pitfalls and frequent questions.**
- *Are the scores already probabilities?* No: only after the softmax, and row by row (`dim=-1`).
- *Why separate K and V?* What is used to **find** ≠ what you want to **retrieve** (example of the vowel looking for the previous consonant).
- *Is the mask a parameter?* No: `register_buffer`. It does not appear in the parameter count.
- *Why dropout on the attention weights?* Refer to 06.
- *How does attention know the order?* It does not: refer to 08 (position embedding).

**Note.** In the old notebook (`Chapitre_2_GPT_pre-final`, cell "Même idée encore, mais avec un masque + softmax", i.e. "same idea again, but with a mask + softmax"), the `masked_fill` line was **commented out**: the comparison therefore showed a uniform, unmasked matrix. This is fixed in the new 05.

## 06 — FeedForward (25 min)

**Flow.** Formula (eq. 2 of the paper) → two Linear = one Linear (assert) → `sin(3x)` experiment → ReLU / GELU / tanh and their derivatives → "only position 5 changes" test vs attention (5 to 9) → weight (33,088 versus 16,448) → dropout (train/eval, the 1/(1−p) factor).

**Key messages.**
- **Attention = mixing over T; FeedForward = mixing over C.**
- Without a non-linearity, depth is useless.
- Dropout: active only in `train()`. Hence `model.eval()` before measuring and generating.

**Moment to exploit.** The changed-positions test: `FeedForward → [5]`, `Attention → [5, 6, 7, 8, 9]`. It sums up both the causal mask and the role of each layer.

## 07 — Stacking layers and backpropagation (40 min)

This notebook answers the request "how layers are chained and influence each other during training".

**Flow.**
1. **Two-number network** (10 min): do the backpropagation table **on the board** before running. Result: dL/dw1 = 15, dL/dw2 = −5. Then `w2 = 0` (layer 1 no longer receives anything) and ReLU off (nobody learns any more).
2. LayerNorm (mean 0, standard deviation 1 **per position**) and `Block`.
3. **Hook trace**: the exact order of the layers and the shapes. Show that the whole core stays `(B, T, C)`.
4. **Central experiment** (≈ 30 s): 8 blocks, 4 variants.
5. Forward influence (perturb block 0 → everything above changes) and backward influence (perturb block 3 → the gradient of block 0 changes by 37%).
6. Size of the updates per group.

**Measured numbers (8 blocks, C=64, at initialization then 200 AdamW steps).**

| Variant | gradient block 0 | gradient block 7 | loss after 200 steps |
|---|---|---|---|
| residuals + LayerNorm | 4·10⁻¹ | 4·10⁻¹ | 2.48 |
| residuals only | 6·10⁻¹ | 8·10⁻¹ | 2.48 |
| LayerNorm only | 1.2 | 6.0 | 3.32 |
| neither | **2·10⁻⁷** | 5.0 | 3.32 |

**Message to get across.** The gradient of a layer is a **product** of the derivatives of the layers above it (so it depends on **their weights**); the residual connection turns this product into a **sum** thanks to the "+1". The layers **co-evolve**: we train them together, with a single loss.

**Honest nuance.** On this small network and 200 steps, "residuals only" does as well as "residuals + LayerNorm". LayerNorm matters more for deeper networks and longer training runs. Do not oversell LayerNorm with this experiment.

## 08 — GPTLanguageModel (25 min)

**Flow.** Position embedding (the same token at positions 1 and 3 → different vectors) → full code in one cell → `forward` table line by line → instantiation → initial loss (4.35, close to `log V` = 4.17) → **causality test of the full model** → `generate` (deliberate error without cropping → `IndexError`) → untrained text.

**Frequent questions.**
- *Why add rather than concatenate token and position?* Both work; adding keeps `C` channels and the network learns to separate the two pieces of information. It is GPT-2's choice.
- *Why is the initial loss closer to log V than for the bigram?* The logits go through a LayerNorm + a Linear with small initialization: they are close to 0.
- *KV cache?* Mentioned as a note: in production, we do not recompute the whole context.

## 09 — Training and evaluation (35 min)

**Start the training cell right at the beginning** and comment on the loop while it runs.

**Flow.** Hyperparameters → loop (figure) → train/val curves with uniform and bigram reference lines → perplexity → **loss per position** → **learned attention maps** → generation → temperature → experiments → summary.

**Reference results (CPU, 8 threads).** See the table at the end of the guide.

**Key messages.**
- Everything below the bigram line is due to **context**.
- The **loss per position** is the numerical proof: at t = 0 the GPT does as well as a bigram, with 10 characters of context it does clearly better.
- Judge a model on **validation**, not on a generated text.

**Attention heads.** The patterns vary from one training run to the next. Look for them with the class: shifted diagonal (previous character), columns on spaces / line breaks (start of a word or line), diffuse heads.

**Closing experiments.** Split the rows of the §8 table (including `CORPUS = "kaamelott"`) between groups of students; each reports **one** validation loss. Compare with the table.

---

## Main references (for you)

- A. Karpathy, *Let's build GPT: from scratch, in code, spelled out* — https://www.youtube.com/watch?v=kCc8FmEb1nY; code: https://github.com/karpathy/ng-video-lecture
- Vaswani et al., *Attention Is All You Need*, 2017 — https://arxiv.org/abs/1706.03762
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW) — https://arxiv.org/abs/1711.05101
- He et al., *Deep Residual Learning* — https://arxiv.org/abs/1512.03385; Ba et al., *Layer Normalization* — https://arxiv.org/abs/1607.06450
- Srivastava et al., *Dropout* — https://jmlr.org/papers/v15/srivastava14a.html
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (residual stream) — https://transformer-circuits.pub/2021/framework/index.html
- J. Alammar, *The Illustrated Transformer* — https://jalammar.github.io/illustrated-transformer/
- 3Blue1Brown, *Attention in transformers, visually explained* — https://www.youtube.com/watch?v=eMlx5fFNoYc

## Reference results for notebook 09

Measured on CPU (8 threads), notebook configuration: B=32, T=64, C=64, 4 heads, 4 blocks, dropout 0.1, AdamW lr=1e-3, 2,000 steps, 211,777 parameters.

| Measure | Value |
|---|---|
| Training time | 112 s (≈ 2 min for the whole notebook) |
| Val loss: uniform / best bigram / mini-GPT | 4.17 / 2.48 / **1.91** |
| Perplexity: uniform / bigram / mini-GPT | 65 / 12.0 / **6.8** characters |
| Loss per position (val): t=0 / t=10 / t=63 | 2.51 / 1.85 / 1.91 |

Attention maps observed in this run: in layer 0, three heads out of four mostly look at the **current character** and one is diffuse. In layer 3, the patterns are more varied, and one head focuses on the **last line break** (the start of the line of dialogue). The patterns change from one training run to the next.

Karpathy reference (GPU, ≈ 10M parameters, 5,000 steps): val ≈ 1.48.
