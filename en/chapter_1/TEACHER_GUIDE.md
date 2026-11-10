# Teacher guide — Chapter 1: from bigrams to a first neural network

This document is for **running** the session: order, timings, key messages, transitions, pitfalls and answers to questions. The full course is in the notebooks. This guide keeps only what you need as a teacher.

## Overview

| # | Notebook | Time | One-sentence message | CPU time |
|---|---|---|---|---|
| 01 | `01_counting_bigrams` | 30 min | Count the transitions, normalize each row, sample: we already have a language model. | < 5 s |
| 02 | `02_measuring_the_loss` | 25 min | The mean NLL scores a model: −log of the probability of the correct answer, averaged. | < 5 s |
| 03 | `03_one_hot_and_weights` | 20 min | `one_hot(x) @ W` = reading row `x` of `W`: 27 free scores. | < 5 s |
| 04 | `04_softmax_forward_pass` | 20 min | Softmax = exp then divide by the sum; `softmax(log N) = P`. | < 5 s |
| 05 | `05_gradients_and_update` | 25 min | `backward()` computes the slopes, we take the step `W -= lr * W.grad`. | < 5 s |
| 06 | `06_training_and_comparison` | 25 min | 100 to 400 steps are enough for `W` to recover the counting table. | ≈ 2 min |

**Total: ≈ 2 h 25.** To fit into **2 h**:
- in 01, go quickly through "Is it better than chance?";
- in 02, show perplexity and smoothing in 3 minutes (or as reading);
- in 04, do section 3 (properties of softmax) orally, without the cells;
- in 06, skip section 5 (regularization).

For a 2 × 1 h 15 session: stop after 03. Session 2 then starts on the question "how do we turn scores into probabilities?".

## Setup

- Folder: `en/chapter_1/`. The notebooks import `bigram.py` (same folder). This module contains exactly the code built in the notebooks (counting, generation, forward pass, NLL) and a heatmap plotting function.
- Data: `names.txt` is looked for next to the notebooks, then in the repository's `corpus/` folder (where it already is), otherwise it is downloaded from the *makemore* repository. For students, copy `names.txt` into `en/chapter_1/` so they do not depend on any connection.
- Dependencies: `torch`, `matplotlib`, `numpy`. No GPU.
- The diagrams are in `figures/` (SVG, all local).
- The six notebooks have been executed and saved **with their outputs**: you can project them without rerunning anything. Notebook 06 takes about 2 minutes; run it while you comment on the loop.
- The original notebook `CHAPITRE_1.ipynb` was not modified.

## The common thread to repeat

> **Two paths to the same table.** Path 1 is to **count**. Path 2 is to **learn** `W` by minimizing the NLL. Path 2 is longer here, but it is the only one that scales.

The **four invariants** to have students repeat at the end: logits → softmax → mean NLL → gradient descent. They will remain the same for the Transformer of chapter 2. Only the way the logits are computed will change.

---

## 01 — Counting bigrams (30 min)

**Outline.**
1. The challenge.
2. The corpus.
3. `stoi` / `itos`.
4. The figure "emma → 5 bigrams → 5 = 4 + 1".
5. The count `N` and the check `sum = letters + names`.
6. The annotated heatmap and the top 10.
7. Normalizing the `.` row.
8. The **broadcasting trap**.
9. `multinomial` vs `argmax`.
10. Generation, then the comparison with the uniform model.

**Key messages.**
- Row = current character, column = next character. `N` is not symmetric: `N['a','m'] = 1634`, `N['m','a'] = 2590`.
- We normalize **by row** because we condition on the current character.
- `multinomial` samples, `argmax` always picks the same thing.

**Moments to make the most of.**
- **The `keepdim` trap.** Have students guess the shape of `N.sum(dim=1)` before running. The wrong version does not crash, but its rows sum to 7.0 or 0.2. It is the most common bug in this chapter. Message: "always test that your distributions sum to 1".
- **`argmax` gives `a`, `a`, `a`.** After `.`, the most frequent letter is `a`. After `a`, the most frequent is `.`. Have the class predict it before running.
- **Uniform model:** unpronounceable 40-character strings, versus a mean length of 6.3 for the bigram (6.1 in the corpus).

**Measured numbers.**
- 32,033 names, 228,146 transitions, 102 zero cells in `N` out of 729.
- Most frequent bigrams: `n→.` (6,763), `a→.` (6,640), `a→n` (5,438).
- Out of 1,000 generated names, 102 already exist in `names.txt`.

**Frequent questions.**
- *Why a single `.` symbol for both start and end?* It is enough: the `.` row gives the starts, the `.` column the ends. One less symbol to handle.
- *The generated names are bad, is that normal?* Yes. The model has only one character of memory. That is the whole motivation for the following chapters.

**Transition.** "These names look better than chance. But 'looking better' is not a number. How do we score a model?"

## 02 — Measuring the loss (25 min)

**Outline.**
1. `P(e | .)`.
2. The likelihood of `emma` (figure 02).
3. The **underflow** of the product.
4. The log, then −log: the NLL.
5. The −log p curve with the points of emma.
6. The mean NLL over the corpus.
7. Perplexity.
8. The zero trap and smoothing.

**Key messages.**
- We only score the probability given to the **correct answer**.
- Why the log? (1) The product drops to **0 from 100 transitions on** in float32; show it on screen. (2) The ranking of models does not change. (3) A sum is easy to differentiate.
- Reference points: **uniform 3.30 · counting 2.454**. Perplexity 27 versus 11.6.

**Moment to make the most of.** `andrejq` gives an **infinite** NLL: the transition `j→q` does not exist in the corpus. Smoothing `N+1` costs almost nothing (2.4546 versus 2.4540). With `+100,000`, we fall back to uniform (3.28). This prepares the regularization of 06 and AdamW's *weight decay* in chapter 2.

**Pitfall.** The loss is measured **on the data used for counting**. Say it explicitly: the train/validation split comes in chapter 2.

**Frequent questions.**
- *Why the mean and not the sum?* To compare corpora of different sizes, and so that the learning rate does not depend on the size of the data.
- *The original course said "we use the log because the Transformer paper does".* That is not the real reason. The real reasons are underflow, monotonicity and differentiation (fixed in this notebook).

**Transition.** "We have a model (`P`) and a score (the NLL). What if the probabilities came from a matrix we can correct?"

## 03 — One-hot and weights (20 min)

**Outline.**
1. The counting / network table.
2. Why a network (the `27³ × 27` explosion).
3. The 5 exercises `xs` / `ys` of emma.
4. One-hot.
5. Figure 03 and the 3 × 3 example.
6. 27 × 27 on emma.
7. Random `W` (heatmap and histogram).

**Key messages.**
- An index is a **label**: `m = 13` is not "13 times `a`".
- `one_hot(x) @ W == W[x]`: it is a **row selection** (assert on screen). In chapter 2, it is `nn.Embedding`.
- The two `m`s of emma receive **the same scores** for two different correct answers: this is the limit of the bigram.

**Moment to make the most of.** Have students compute `[0, 1, 0] @ W_mini` on the board **before** running. Then ask: "and `[1, 0, 0]`?".

**Frequent questions.**
- *Why `.float()`?* `F.one_hot` returns integers, and `@` requires the same type on both sides.
- *Why is `W` square?* 27 possible inputs × 27 possible outputs. In chapter 2, the embedding will have a different width (`n_embd`).

**Transition.** "27 scores, sometimes negative, that do not sum to 1. How do we turn them into probabilities?"

## 04 — Softmax and forward pass (20 min)

**Outline.**
1. `exp` on [−1, 0, 1].
2. Normalization, then softmax.
3. The 3 properties: shift, scale (temperature), numerical stability (the naive version gives `nan` on [1000, 1001, 1002]).
4. **`softmax(log N) = P`**.
5. The full forward pass (figure 04).
6. Comparison `m → a`: 0.39 for `P` versus 0.07 for a random `W`.
7. The loss of emma: 3.77 for the random network, versus 2.51 for the table.

**Key messages.**
- We keep **free logits** in `W` and transform them: the optimizer can move the weights without breaking the constraints.
- **Logits are "log-counts".** The counting table is **one particular point** in the space of `W`s. Training must find it without seeing `N`. This is the sentence that gives meaning to the whole chapter.
- Only **differences** between logits matter.

**Pitfall.** The variable `counts = logits.exp()` is **not** `N`. The name comes from the analogy with log-counts. Say it clearly once.

**Transition.** "The loss can be computed. But computing it does not change `W`. In which direction should each weight move?"

## 05 — Gradients and update (25 min)

**Outline.**
1. The cell to read in each row.
2. `torch.arange(n)`.
3. The `[:, ys]` trap, which gives a `(5, 5)` shape.
4. The partial derivative.
5. `requires_grad` / `backward` / `W.grad` (figure 05).
6. The finite-difference check.
7. **The gradient by hand**, `probs − one_hot(y)`.
8. Comparing learning rates on one step.
9. A real step.

**Key messages.**
- `backward()` **does not modify** `W`: it fills `W.grad`. The update is up to us.
- Interpreting `probs − one_hot(y)`: the correct answer goes up by `1 − p` and each wrong answer goes down by `p`. We mostly punish **confident** mistakes.
- Only the rows `.`, `a`, `e`, `m` receive a gradient: they are the only **inputs** of emma.

**Measured numbers.**
- ∂L/∂W['.', 'e'] = −0.1975 (backward) versus −0.1976 (finite difference).
- One step on emma from a loss of 3.77: lr = 1 → 3.57; lr = 10 → 1.93; lr = 100 → 0.31; lr = 1000 → `nan`.

**Moment to make the most of.** The audience knows derivatives: have them differentiate `−z_y + log Σ e^{z_k}` on the board (2 lines). Then the assert "gradient by hand == W.grad" is very satisfying.

**Pitfall.** lr = 100 gives 0.31 on emma: on **5 examples**, the model memorizes. Do not let students believe a large lr is always better. Notebook 06 shows the effect on the real corpus.

**Frequent questions.**
- *Why `torch.no_grad()` for the update?* The update is not an operation of the model: we do not want to record it in the graph (and PyTorch refuses to modify in place a leaf that requires a gradient).
- *Why `W.grad = None`?* PyTorch **accumulates** gradients at each `backward()`. In chapter 2, this is `optimizer.zero_grad()`.

**Transition.** "One step on one name. Now: 100 steps on 228,146 bigrams."

## 06 — Training and comparison (25 min)

**Outline.**
1. All the `xs` / `ys` (M = 228,146).
2. The annotated loop (figure `06_loop`).
3. 4 learning rates.
4. The curves.
5. 300 more steps.
6. Comparing the bars after `a` and the 3 heatmaps (counting, network, difference).
7. Generation with the same seed.
8. Regularization.
9. The wrap-up, the final question and the appendix.

**Measured numbers (100 steps, same initial `W`, NLL over the whole corpus).**

| learning rate | loss after 100 steps |
|---|---|
| 0.1 | 3.673 |
| 1 | 3.189 |
| 10 | 2.576 |
| 50 | 2.473 |
| 50, then 300 more steps | **2.458** |
| counting | 2.454 |
| uniform | 3.296 |

- Mean absolute gap between the two tables: **0.0017**. The rows that differ most are `q`, `x` and `w`, the least observed (`q`: 272 occurrences).
- With the **same seed**, the 10 generated names are **identical** for both models. A guaranteed effect in class.
- Regularization, after 100 steps with lr = 50:
  - λ = 0: NLL 2.473, max |W| ≈ 4;
  - λ = 0.1: NLL 2.506;
  - λ = 1: NLL 2.721, max |W| ≈ 2.4.

**Key messages.**
- The network recovers the table **without ever seeing `N`**, only through the loss.
- The initial loss (3.76) is **worse than uniform**: a random `W` has arbitrary preferences. It is not a bug (same remark in chapter 2).
- Regularization is the network's smoothing. It becomes AdamW's *weight decay* in chapter 2.

**Moment to make the most of.** Before running the curves, have the class vote: "which lr gets closest to counting in 100 steps?". Then the opening question of 01: *why learn a table we know how to compute?* The answer is **27⁸ ≈ 2.8 × 10¹¹** rows for 8 characters of context.

**Transition to chapter 2.** "We keep the four invariants. In chapter 2, `one_hot @ W` becomes `nn.Embedding`, the NLL becomes `F.cross_entropy`, the update becomes `optimizer.step()`. Above all, we give the model **more than one character of context**."

**Answers to the appendix.**
1. `P[stoi['a']]` contains 27 values summing to 1: it is the distribution of the characters that follow `a`, normalized by row.
2. For 200 bigrams, the shape is `(200, 27)`.
3. The NLL blows up because −log p tends to ∞ as p tends to 0.
4. We follow `−W.grad` because the gradient points in the direction where the loss **goes up**.
5. The gradient of row `q` is zero because `q` is never an **input** in emma (the row is not selected by the one-hot).
6. The limit lifted in chapter 2 is the single-character context.

---

## Differences from `CHAPITRE_1.ipynb`, worth knowing

**Content kept:** the whole original progression (sections 1 to 7 and the appendix), reorganized into 6 notebooks.

**Additions:**
- the `keepdim` trap;
- `argmax` versus `multinomial`;
- the comparison with the uniform model;
- the underflow demonstration;
- perplexity;
- smoothing and `andrejq`;
- the properties of softmax;
- `softmax(log N) = P`;
- the analytical gradient;
- continuing to 400 steps;
- the difference map;
- regularization;
- 6 diagrams.

**Fixes to the old notebook:**
- `plot_matrice_caracteres` (now `plot_char_matrix` in `bigram.py`) used `np` without `import numpy`. The function crashed as soon as it was given anything other than a tensor.
- The `P[0]` cell was placed **before** `P` was defined: it crashed if you ran the notebook in order.
- One markdown cell was empty (after "requires_grad…").
- There was a typo, "Moyennner".
- The justification "we use the log because the Transformer does" was replaced with the real reasons.
- The `device` variable was defined but never used: it was removed.

## Main references (for you)

- A. Karpathy, [*The spelled-out intro to language modeling: building makemore*](https://www.youtube.com/watch?v=PaCmpygFfXo). This chapter follows his progression: counting, NLL, one-hot, softmax, gradient, regularization.
- A. Karpathy, [*building micrograd*](https://www.youtube.com/watch?v=VMj-3S1tku0), for backpropagation in depth.
- D. Jurafsky and J. H. Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), chapter 3: n-grams, perplexity, smoothing.
- I. Goodfellow, Y. Bengio and A. Courville, [*Deep Learning*](https://www.deeplearningbook.org/), sections 4.1 (stable softmax), 5.5 (maximum likelihood), 6.2.2.3 (softmax), 6.5 (backpropagation), 7.1 (L2 penalty).
- Y. Bengio et al., [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), 2003. The historical bridge to chapter 2.
- C. E. Shannon, *A Mathematical Theory of Communication*, 1948. Letter-by-letter approximations of English, an opening anecdote that works well.
