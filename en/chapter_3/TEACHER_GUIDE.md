# Teacher guide — Chapter 3: fine-tuning Mistral 7B

This document is for **running** the session. The notebooks contain the full course. The design document (`deep-research-report(1).md`, kept locally and not published) gives the rationale and the sources.

## Overview

| # | Notebook | Time | One-sentence message | Compute |
|---|---|---|---|---|
| 01 | `01_why_finetune` | 20 min | Mistral 7B **is** the GPT from chapter 2, 34,000 times larger; SFT is the same cross-entropy, masked on the response. | CPU < 10 s |
| 02 | `02_dialogue_data` | 30 min | The task is defined by how the examples are built; we split **by scene**. | CPU ≈ 10 s |
| 03 | `03_lora_by_hand` | 35 min | LoRA = `W₀ + (α/r)·BA`, B = 0 at the start; it learns a style with 5% of the parameters, but it is not magic. | CPU ≈ 2 min (+ 3 min without the checkpoint) |
| 04 | `04_memory_and_quantization` | 25 min | 81 GB → 5 GB: LoRA removes the gradients and AdamW, NF4 divides the weights by 4. | CPU < 10 s |
| 05 | `05_qlora_mistral_colab` | 40 min | Full QLoRA on a T4: before / after, mask checked, cost measured. | GPU, see below |
| 06 | `06_evaluation_colab` | 25 min | A falling loss is not a better model: memorization, forgetting, human evaluation. | GPU ≈ 5 min |
| 07 | `07_appendix_merge_gguf` | homework | Merging, GGUF, llama.cpp, and the question of rights. | GPU / High-RAM |

**Total: ≈ 3 h** without the appendix.

### 2-hour version (scenario from the design document)

| Time | Content |
|---|---|
| **0–5 min** | Students open **notebook 05 on Colab** (T4 GPU) and run §0 to §3 (installation + 14.5 GB download ≈ 5 min). It runs during the theory. |
| 5–20 min | 01: §1 (same architecture), §3 (prompting / RAG / FT), §4 (mask). Go quickly over BPE. |
| 20–40 min | 02: §1, §3 (the 96–100% leak is **the** highlight), §4, §5. |
| 40–60 min | 03: §1–§2 on the board, then the §4 experiment (with the provided checkpoint, 2 min of compute) and the trade-off plot. |
| 60–70 min | 04: §1 (81 / 14 / 5 GB plot) and §4 (NF4, histograms). The rest as reading. |
| 70–105 min | 05: §4 to §9 on Kaamelott (training ≈ a few minutes). The fastest students rerun on Plato. |
| 105–120 min | 06: §1, §3, §5 + discussion of the limits, presentation of the homework. |

## Setup

- Public repository: `https://github.com/zenocode-org/ai-masterclass` (folder `en/chapter_3/`). The Colab notebooks clone this repository: **every change must be pushed** before class.
- Local (01–04): `torch`, `transformers` (Mistral tokenizer, ~2 MB downloaded), `matplotlib`, `numpy`, `peft` (optional, §6 of 03). Everything runs on CPU. The corpora are read from the repository's `corpus/` folder.
- Colab (05–07): the first cell installs **pinned versions** tested together: `transformers==5.22.1 trl==1.17.1 peft==0.21.2 bitsandbytes==0.50.2 accelerate==1.15.0 datasets==5.1.0`. If Colab changes its PyTorch version and the installation fails, removing the version numbers is the first thing to try.
- Model: `mistralai/Mistral-7B-Instruct-v0.3`, Apache 2.0, **not gated** on Hugging Face (no token needed). A faster-to-download variant (4 GB, already quantized): `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`; in that case keeping `quantization_config` has no effect, the model is already in 4 bits.

### ⚠️ To do before the first class: a full run on Colab

**What was tested:**
- notebooks 01 to 04 were run in full (CPU), outputs saved;
- notebooks 05, 06 and 07 (§1) were run end to end on GPU with a **random mini-Mistral** (same architecture, same tokenizer, 8 M parameters): bitsandbytes 4-bit loading, `get_peft_model`, `SFTTrainer`, loss mask, evaluation, saving and reloading the adapter, `disable_adapter`, merging. Both precision paths were tested: BF16 and **FP16 (the T4's)**.

**What was not tested:** the real run on Mistral 7B (no GPU large enough on the preparation machine) and the llama.cpp commands in appendix 07 (§2–§3). The GPU durations below are **estimates**. Before class:
1. run 05 on Kaamelott **and** on Plato on a Colab T4; note the duration, the peak memory, the throughput (tokens/s), and replace the estimates in this guide;
2. check that the "what we expect to see" comments (§7–§8 of 05, §3 of 06) match the outputs; adjust them otherwise;
3. save the resulting adapters to a shared Drive: **plan B** if Colab refuses a GPU on the day (quotas). Notebook 06 can then run from that adapter.

### GPU estimates (replace with your measurements)

| Run | T4 (15 GB) | Expected memory |
|---|---|---|
| 4-bit loading | 2–5 min (download) | ≈ 4 GB |
| Kaamelott, 38 examples × 3 epochs (15 steps) | a few minutes | 6–9 GB |
| Plato or Shakespeare, 800 examples × 1 epoch (50 steps) | 15–25 min | 6–10 GB |
| full 06 (40 examples, 12 × 2 generations, out of domain) | ≈ 5 min | ≈ 5 GB |

## The common thread to repeat

> **Nothing new in the architecture or in the loss.** Chapter 3 changes three things: **what we put in the examples** (02), **which parameters move** (03), **how they are stored** (04).

---

## 01 — Why fine-tune? (20 min)

**Key messages.**
- The parameter count "by hand" lands exactly on PyTorch's (7,248,023,552); 96% are in the linear layers of the blocks.
- Correspondence table chapter 2 → Mistral: have the students find `query`→`q_proj`, `FeedForward`→`gate/up/down_proj` themselves.
- `-100` mask: `F.cross_entropy` ignores it by default. In the Kaamelott example, 19 tokens out of 59 count.

**Measured numbers.** Characters per BPE token: Kaamelott 2.74, Plato 2.90, Shakespeare 3.11.

**Frequent questions.**
- *Why 8 K/V heads for 32 Q heads?* Grouped-query attention: less memory for the KV cache at inference, negligible quality loss.
- *Does the model already know Kaamelott?* Maybe a little (the web talks about it). That is why we compare before/after on validation scenes.

## 02 — Data (30 min)

**Moment to make the most of.** The leak rate of a random split: **100%** (Kaamelott), **96%** (Plato), **95%** (Shakespeare), versus 0%, 8%, 0.2% for the split by group. Have the students **predict** before running.

**Measured numbers.**

| Corpus | Groups | Lines | Train / val examples | Tokens p50 / p99 | Tokens per epoch |
|---|---|---|---|---|---|
| Kaamelott | 7 scenes | 55 | 38 / 10 | 131 / 181 | 4,433 |
| Plato | 45 dialogues | 11,179 | 8,335 / 1,013 | 185 / 530 | 1.75 M |
| Shakespeare | 178 blocks of 40 | 7,097 | 6,013 / 665 | 162 / 430 | 1.08 M |

Plato's 8% are Socratic stock phrases ("Sans doute." *Doubtless.*, "Tu as raison." *You are right.*): a good discussion topic on legitimate duplicates.

**Kaamelott is tiny** (a 4,237-character excerpt, the same as in chapter 2). The full Kaggle dataset (> 30,000 lines, Hypnoweb transcripts) is the natural extension for the homework; see the note on rights below.

## 03 — LoRA by hand (35 min)

**Flow.** Rank of `BA` → `LoRALinear` class (identical output at the start) → mini-GPT pre-trained on Plato (checkpoint provided) → 4 variants × 300 steps on Kaamelott → trade-off plot → merging → `peft`.

**Measured numbers (CPU, fixed seed).**

| Variant | Trained parameters | Best Kaamelott val | Plato val after 300 steps |
|---|---|---|---|
| Plato model (start) | 0 | 2.49 | 1.46 |
| from scratch | 218,098 | 2.15 | 2.85 |
| full fine-tuning (lr 1e-4) | 218,098 | **1.62** | **1.73** |
| LoRA q,v r=4 (lr 3e-3) | 10,240 | 1.83 | 1.84 |
| LoRA everywhere r=4 (lr 3e-3) | 27,648 | 1.69 | 2.15 |

**A point of honesty to own in front of the students.** Here, full fine-tuning at a small lr **dominates** LoRA on both axes. Showing it is deliberate: LoRA is not "better", it is **cheaper** (memory, storage). The slogan "LoRA learns less and forgets less" (Biderman et al., 2024) holds at the usual settings on large models, and the lr matters as much as the method. Suggested experiment: LoRA q,v at lr 1e-3.

NF4 on the same mini-GPT (notebook 04): Plato val 1.450 (FP32) → 1.463 (INT4) → 1.462 (NF4).

## 04 — Memory and quantization (25 min)

**Numbers (estimate, batch 1 × 512 tokens, r = 16 on all linear layers).** Full FT 81 GB (of which 54 GB of AdamW), LoRA 14.4 GB, QLoRA 4.9 GB. Add 1 to 3 GB of real CUDA overhead. The design document gives wider ranges (10–16 GB for QLoRA) because it includes larger batches and sequences.

**Moment to make the most of.** The outlier example: a single constant for 4,096 weights → 3 levels out of 16 used, error × 20.

**Frequent questions.**
- *Why not quantize the adapters too?* They are the ones receiving the updates: in 4 bits, AdamW's small steps would vanish in the rounding.
- *The T4 has no BF16?* Correct (Turing architecture). Notebook 05 detects the GPU's capability and switches to FP16; the adapters are then kept in FP32.

## 05 — QLoRA Mistral 7B (40 min)

**Pitfalls.**
- Have the students run §0–§3 **at the start of the session** (download).
- §6 ("what counts in the loss") is the check to have them do **systematically**: half of failed fine-tunings come from a wrong mask.
- The cell after `SFTTrainer` converts the adapters back to FP32 on a T4: TRL casts them to BF16 for a 4-bit model, which the T4 does not support.
- `save_strategy="no"`: no intermediate checkpoints (Colab disk), the adapter is saved at the end. For a long run, switch to `"steps"`.

**Failures already encountered during preparation:** none on the mini-model. Table of classic failures at the end of the notebook.

## 06 — Evaluation (25 min)

Messages: perplexity is a diagnostic; ROUGE-L penalizes good but different responses; a quotation of ≥ 50 characters is a sign of memorization; a style fine-tuning can erode the assistant behavior and the refusals (Qi et al., 2023). The blind grid (`evaluation/grid_<corpus>.csv`, separate key) is used for the homework.

## Rights and licenses (to mention in class)

- **Mistral 7B v0.3 weights**: Apache 2.0.
- **Kaamelott**: a copyrighted work. The course file is a short excerpt from fan transcripts; the full Kaggle dataset also comes from community transcripts (Hypnoweb), not from official scripts. Educational use in class; do not redistribute an adapter that memorizes lines.
- **Plato** (Victor Cousin's translation, 19th c.) and **Shakespeare**: public domain.

## Homework (pairs, short experimental report)

1. Build **two formulations** of the dataset: next line (`mode="continue"`) and response of a given speaker (`mode="speaker"`, see `ft.build_examples`).
2. Train two QLoRAs changing **only one** variable: `r = 8` versus `r = 32`, or 1 versus 3 epochs.
3. Keep exactly the same split by scene / dialogue.
4. Measure: train / val loss, perplexity, ROUGE-L (as an indication), blind human evaluation on at least 30 prompts.
5. Add a memorization measure (§3 of 06).
6. Document: peak GPU memory, duration, tokens processed, estimated cost.
7. Pick five failures and propose a causal explanation.
8. Export the adapter, then test either the merge or the GGUF conversion (notebook 07).

**Extensions:** compare `Mistral-7B-v0.3` (base) and `Mistral-7B-Instruct-v0.3` on the same data (does instruction tuning help or constrain the style?); change the task with the same corpus ("predict the next speaker", "identify the character"); look for near-duplicates with n-grams or MinHash; use the full Kaggle Kaamelott dataset.
