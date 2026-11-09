# 01 · Weight-Space Programming

> **Source:** Rikiya Takehi\*, Jayden Teoh\*, Alex Zhang, Omar Khattab (MIT CSAIL), *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*, blog post, 2 November 2026.
> <https://rikiyat.github.io/blog/weight-space-programming/>
> **Full logs** (every prompt, reply, line of code and weight edit): <https://github.com/RikiyaT/lm_understand-logs>
> **Edited model:** <https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming>

This is a blog post with released logs, not a peer-reviewed paper. The authors call the results "very early".

---

## Contents

1. [The one-paragraph summary](#1-the-one-paragraph-summary)
2. [The eight experiments at a glance](#2-the-eight-experiments-at-a-glance)
3. [Explainer: why weights can be *read* at all](#3-explainer-why-weights-can-be-read-at-all)
4. [Explainer: the Qwen edit, line by line](#4-explainer-the-qwen-edit-line-by-line)
5. [Critical reading](#5-critical-reading)
6. [The notebooks](#6-the-notebooks)
7. [Glossary](#7-glossary)
8. [References](#8-references)

---

## 1. The one-paragraph summary

A neural network's weights are a program, but usually no one writes it. An optimizer like gradient descent does ("Software 2.0", in Karpathy's words). The post asks whether a frontier LLM can play the role of the programmer instead, reading weights to understand what a network does and writing weights to change it **without gradient descent**. The LLM, OpenAI's `gpt-6-astra` (called "Astra"), is given weights as numbers, sometimes a Python sandbox, and no gradients. It reasons about what each part computes. It identifies which game a policy network plays and which shape a diffusion model draws. It finds the copying heads of a small transformer. It fixes a broken car-driving policy by editing 3.5% of its weights, and writes game-playing networks from all-zero weights that beat PPO at Breakout. The headline result: it diagnoses why Qwen3.5-0.8B fails at math (its reasoning **loops** until it hits the token limit), traces the loop to one direction written by two MLP layers, and removes it by changing **131,072 of 0.8 billion weights (0.016%)**. MATH-500 goes from 7.9% to 48.1%, GSM8K from 13.8% to 67.1%.

The authors name three capabilities:

| Capability | What it means | Experiments |
|---|---|---|
| **Reading** | infer what a network does from its weights | 1, 2, 3, 7 |
| **Fixing** | change some weights to repair a behavior | 4, 8 |
| **Writing** | produce the weights of a working network from scratch | 5, 6 |

---

## 2. The eight experiments at a glance

| # | Task | Network | What Astra had | Result |
|---|---|---|---|---|
| 1 | Which game was this policy trained on? | MLP 500→32→32→5, 17k weights, PPO on Breakout / Pong / Pac-Man / Freeway | weights only | correct for all 4 |
| 2 | What does this diffusion model draw? | MLP 160→64→64→144, 24k weights (DDPM, 12×12 images) | weights only, no forward pass | correct shape for all 3, but drawn as outlines instead of filled shapes |
| 3 | Which language did this transformer learn, and which heads copy? | 4 layers × 6 heads, 7.2M weights, only the 4 middle blocks given | Python sandbox, no forward pass | correct language for all 3 (Python, romaji, English). Copying heads match measured scores |
| 4 | Fix a car that gets stuck on a ramp (Kinetix) | 171k-weight PPO actor, 0/256 successes | weights, scene, one episode, one try | 1,024/1,024 successes, 3.5% of weights changed |
| 5 | Write a MinAtar policy from all zeros | MLP 600→32→32→6 | game rules and source code, no tools, one reply | Breakout **517** vs PPO-500M **25**; Seaquest 53 vs 89; Asterix 53 vs 55 |
| 6 | Write a 1.5M-weight Go network (9×9) from zeros | 243→512×6→82 | Python sandbox, 0 to 50 forward passes | with 0 passes, wins more than half of games against two small PPO nets |
| 7 | Which of two Gemma-3-270M copies is sycophantic? | 2 fine-tuned copies | weights only: **failed 4/4**. With attention patterns: 3/3 | head 10.2 reveals it |
| 8 | Make Qwen3.5-0.8B better at math | 0.8B, 24 layers | Python over weights, 10/25/50 forward passes | MATH-500 7.9 → 48.1, GSM8K 13.8 → 67.1 (50 passes) |
| + | Same on Gemma-3-270M (does not loop) | 270M | same | **no real gain** (6.8 → 7.0 to 7.7) |

The last row matters most for interpretation. See [§5](#5-critical-reading).

---

## 3. Explainer: why weights can be *read* at all

It sounds magical that anyone (human or LLM) can look at a matrix of numbers and say "this plays Breakout". It becomes less magical once you see that each weight matrix has a clear **role**. Three ideas from the course cover almost the whole post.

### 3.1 A linear layer = a set of detectors (rows) and a set of writers (columns)

For `y = W x`, with `W` of shape `(out, in)`:

- **Row `i`** of `W` is a *template*. Output `i` is large when `x` looks like that row (a dot product). Reshape a first-layer row back to the input's shape (a 10×10 grid, an image) and you can **see** what it detects. Experiment 1 works this way: weights concentrated along the bottom row of the grid mean "this unit watches the paddle", so the game is Breakout.
- **Column `j`** of `W` is what input `j` *writes* into the output. In a transformer, the MLP's `down_proj` (shape `d_model × d_ff`) has one column per hidden neuron. **Column `j` is the vector neuron `j` adds to the residual stream when it fires.** Experiment 8 edits exactly these columns.

> **Course link:** in chapter 1, row `i` of the bigram weight matrix `W` (27×27) holds the logits of "what follows letter `i`". It was already a readable table.

### 3.2 The residual stream: everyone reads from and writes to the same vector

In a pre-norm transformer (chapter 2, notebook 07), `x = x + attn(ln(x))` then `x = x + mlp(ln(x))`. Every layer **reads** the current vector `x` and **adds** something to it. So the residual stream is a shared workspace of dimension `d_model` (1,024 for Qwen3.5-0.8B), and meaning lives in **directions** in that space. If one direction `u` means "I am about to repeat myself", then:

- a neuron *promotes* looping if its output column has a large component along `u`, i.e. a large `uᵀ W[:, j]`;
- you can *remove* that influence by subtracting the component: `W[:, j] ← W[:, j] − u (uᵀ W[:, j])`.

That one line is the whole Qwen edit. It is detailed in [§4](#4-explainer-the-qwen-edit-line-by-line).

### 3.3 An attention head = two small programs: QK (where to look) and OV (what to move)

This is the part most relevant to the chapter 2 question *"what do Q, K and V actually learn?"*.

A head computes, for a query position `i` and a key position `j`:

```
score(i, j) = (W_Q x_i) · (W_K x_j) = x_iᵀ (W_Qᵀ W_K) x_j        ← the QK circuit
out_i      = Σ_j softmax(score)_ij · W_O W_V x_j                ← the OV circuit
```

`W_Q` and `W_K` are never used separately. Only their **product** `W_QK = W_Qᵀ W_K` (d×d) matters, and it decides *which positions attend to which*. Likewise only `W_OV = W_O W_V` matters for *what information gets copied* once attention is decided. So:

| Matrix | Question it answers | How to read it from weights |
|---|---|---|
| `W_QK = W_Qᵀ W_K` | "for a query that looks like *this*, which keys look relevant?" | sandwich it between embeddings: `E W_QK Eᵀ` (token→token) or `P W_QK Pᵀ` (position→position). A strong diagonal in the token version means "attend to the same token". A strong sub-diagonal in the position version means "attend to the previous position". |
| `W_OV = W_O W_V` | "if I attend to token *t*, what do I write?" | `U W_OV Eᵀ` is a vocab×vocab table: "attending to token *t* raises the logit of token *s*". If it is dominated by its diagonal, the head **copies**. A one-number summary: the **copying score** `Σλ / Σ|λ|` over its eigenvalues λ (+1 = pure copying). |

Experiment 3 relies on this. Astra composed weight matrices this way to find "copying heads" without running the model. The method comes from Elhage et al. (2021), which nobody showed Astra. **Notebook 02 does it by hand.**

### 3.4 Writing weights = compiling an algorithm

If you know what each matrix does, you can write one yourself. To catch a falling ball, a policy only needs the sign of `ball_column − paddle_column`. Two ReLU neurons compute it, one for each sign, and the output layer turns it into "left / stay / right". **Notebook 01 writes this network by hand in 10 lines, and it catches 100% of balls.** Experiment 5 does the same at a larger scale: "the first layer computes where the ball would land, the second moves the paddle there". The research version of this idea is Tracr / RASP, which compile programs into transformer weights.

---

## 4. Explainer: the Qwen edit, line by line

From Astra's final report (`exp8_math/runs/p50c/work_notes/final_report.md` in the logs):

```
W = W0 − 6 · outer(u, mask64 * (u @ W0))
```

applied to `model.language_model.layers.{15,19}.mlp.down_proj.weight` (shape 1024 × 3584).

| Symbol | Shape | Meaning |
|---|---|---|
| `W0` | 1024 × 3584 | original `down_proj`: column `j` = what MLP neuron `j` writes into the residual stream |
| `u` | 1024 | a **unit "repetition direction"**: the mean difference between residual vectors in *repetitive* and *productive* reasoning contexts (measured at the outputs of layers 16 and 20) |
| `u @ W0` | 3584 | for each neuron, how strongly it writes along `u` |
| `mask64` | 3584 | 1 for the 64 neurons that write most along `u`, 0 elsewhere |
| `outer(u, ·)` | 1024 × 3584 | a **rank-1** matrix |

**What it does to each of the 64 selected neurons:** its component along `u` was `c = uᵀ W0[:, j]`. After the edit it is `c − 6c = −5c`. The neuron used to push the model *toward* looping. Now it pushes five times as hard *away* from it. Everything orthogonal to `u` is untouched, and so are the other 3,520 neurons.

- With `α = 1` instead of 6, this would be an exact **projection**: the neurons would no longer write along `u` at all. This is the same idea as "directional ablation" used to remove refusal (Arditi et al., 2024).
- With `α = 6`, it **overshoots** on purpose and becomes a small built-in steering vector against repetition.
- Astra chose α, the layers and the 64 neurons through dose-response tests (α = 1, 2, 4 reduced the log-prob of repetitive continuations by 0.024 / 0.049 / 0.104 nats/token while productive ones barely moved). It also ran a **random-direction control** with the same norm, which gained only +1.6 points ± 3.5.
- The 10-pass run used a dense version (`W' = W − 1.5 u uᵀ W` on 30 matrices). The 25-pass run zeroed attention head 7 of layer 15.

**Notebook 03 rebuilds this edit from the published diff on a laptop GPU** (or CPU). It recovers `u` by SVD, checks that the change is exactly rank-1, and measures the loop on real problems.

---

## 5. Critical reading

**Strong points**

- **Everything is released.** Every prompt, reply and code run is public, and the edits come as exact diffs with hashes. Notebook 03 verifies the Qwen edit is what the post says.
- **Controls exist** where it matters (Exp 8: random-direction control, layer-isolation tests, held-out practice problems). The authors also report failures, which counts for something: Exp 7 failed with weights only, Gemma showed no gain, and Exp 2 drew outlines instead of filled shapes.
- The interpretability methods Astra rediscovers (QK/OV composition, direction ablation) are **established**. That makes the results plausible.

**Things to keep in mind**

- **Fixing a pathology ≠ adding a capability.** Qwen3.5-0.8B with thinking on spends 93% of its MATH-500 answers looping to the 32,768-token limit. Removing the loop *unlocks* math the model already knew. On Gemma-3-270M, which does not loop, editing gave no real gain. The authors say editing "could not add an ability the model lacked". So "7.9 → 48.1" mostly measures a bug fix.
- **Tiny samples in places:** one try in Exp 4, three model pairs in Exp 7, unspecified attempt counts in Exp 1 and 2.
- **No forgetting test.** "Only 0.016% of weights changed, so little is forgotten" is stated as a belief. MMLU-Pro and GPQA do go *up* after the edit, but no broad regression suite was run.
- **Model access:** `gpt-6-astra` at maximum reasoning effort, up to 20 hours per run. You cannot reproduce the *agent* cheaply. You *can* reproduce and study every *artifact* it produced, which is what these notebooks do.
- **Small inconsistencies in the post:** Exp 6 calls the PPO opponent "equal-sized" although it is 243→32→32→82. The GPQA baseline in a figure was an interim value (24.5; the final value is 22.7).
- **Is it really "without optimization"?** No gradients were used, but Exp 6 and 8 keep the changes that win on practice data. That is a form of search guided by reasoning, which the authors acknowledge.

**The big idea worth keeping:** interpretability and optimization become the same activity. If you can say *why* a network fails, you can often fix it with a tiny, surgical edit instead of more training.

---

## 6. The notebooks

All notebooks are in English, run top to bottom, and ship with their outputs, so you can read them like a book first.

| Notebook | Paper link | What you do | Hardware | Time |
|---|---|---|---|---|
| [`01_write_and_read_a_policy.ipynb`](01_write_and_read_a_policy.ipynb) | Exp 1, 4, 5 | **write** a game-playing network by hand (2 neurons, 100% catch rate); train one with RL; **read** its weights as images; **fix** a broken network with a 1-weight edit | CPU | ~2 min |
| [`02_reading_qk_ov_circuits.ipynb`](02_reading_qk_ov_circuits.ipynb) | Exp 3, 7 | train a tiny attention-only transformer to copy repeated sequences, then find **from weights only** which heads copy and how the induction circuit is wired (QK / OV / composition). Confirm with ablations | CPU | ~3 min |
| [`03_dissecting_the_qwen_edit.ipynb`](03_dissecting_the_qwen_edit.ipynb) | Exp 8 | load the real Qwen3.5-0.8B, apply the authors' edit, prove it is rank-1, recover the "loop direction", watch the base model loop and the edited one escape, run a dose-response and a random control | GPU ≥ 4 GB (Colab T4 OK); CPU possible but slow | ~25 min GPU |
| [`04_let_an_llm_read_weights.ipynb`](04_let_an_llm_read_weights.ipynb) | the core idea | serialize a small network to text and ask an LLM (Claude via the API, or copy-paste the prompt into any chat) which game it was trained on | CPU + optional API key | ~2 min |

Shared code is in `wsp.py`. As in the course, it contains nothing that is not explained in a notebook.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **Weight-space programming (WSP)** | reading and writing a network's weights directly by reasoning, instead of by gradient descent |
| **Residual stream** | the vector `x` that flows through a transformer. Each layer reads it and adds to it |
| **Direction** | a unit vector `u` in the residual stream that carries one "meaning". Its value at a position is `uᵀx` |
| **Rank-1 edit** | `W ← W + a bᵀ`: changes a whole matrix with a single pair of vectors. Projecting a direction out of a matrix is one |
| **Directional ablation** | `W ← W − u uᵀ W`: the layer can no longer write along `u` |
| **QK circuit** | `W_Qᵀ W_K`: decides where a head looks |
| **OV circuit** | `W_O W_V`: decides what a head moves once it looks |
| **Copying head** | a head whose OV circuit maps each token to itself (positive eigenvalues) |
| **Induction head** | a head that, on seeing token A, looks for an earlier A and copies the token that came *after* it. It needs a helper head in an earlier layer (composition) |
| **Logit lens** | multiplying a residual-stream vector by the unembedding to see which tokens it promotes |
| **Dose-response** | applying an edit at strength α = 0, 1, 2, … and checking that the effect grows with α. Evidence that the effect is causal |

---

## 8. References

Cited by the post:
- A. Karpathy, *Software 2.0* (2017). <https://karpathy.medium.com/software-2-0-a64152b37c35>
- I. Schlag, K. Irie, J. Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* (2021). <https://arxiv.org/abs/2102.11174>
- D. Lindner et al., *Tracr: Compiled Transformers as a Laboratory for Interpretability* (2023). <https://arxiv.org/abs/2301.05062>
- N. Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021). <https://transformer-circuits.pub/2021/framework/index.html>

Background for the notebooks:
- C. Olsson et al., *In-context Learning and Induction Heads* (2022). <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>
- A. Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024). <https://arxiv.org/abs/2406.11717>
- K. Meng et al., *Locating and Editing Factual Associations in GPT* (ROME, 2022). <https://arxiv.org/abs/2202.05262>
- M. Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories* (2021). <https://arxiv.org/abs/2012.14913>
- G. Weiss, Y. Goldberg, E. Yahav, *Thinking Like Transformers* (RASP, 2021). <https://arxiv.org/abs/2106.06981>
