"""finetune.py — code shared by the chapter 3 notebooks (fine-tuning).

Everything here is built and explained step by step in the notebooks:
  - reading the 3 corpora as lines of dialogue, cleaning             -> 02_dialogue_data
  - building the examples (context -> line), split by group          -> 02_dialogue_data
  - LoRALinear, apply_lora                                           -> 03_lora_by_hand
  - counting Mistral 7B's parameters, memory, NF4                    -> 04_memory_and_quantization
  - generation, loss on the response, memorization, ROUGE-L          -> 05 and 06 (Colab)

The module only imports the standard library and PyTorch: the Hugging Face
parts (transformers, peft, trl) stay visible in the notebooks.
"""
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
CORPUS = {
    "shakespeare": "shakespeare.txt",
    "kaamelott": "script-kaamelott.txt",
    "plato": "plato_train.txt",
}
# Same locations as minigpt.py (chapter 2): next to the notebook, then the repository's corpus/ folder.
FOLDERS = [HERE, HERE / "data", *(p / "corpus" for p in HERE.parents),
           HERE.parents[2] / "philo-masterclass" / "data"]

# Instruction placed at the start of each prompt, in the language of the corpus:
# Kaamelott and Plato (Victor Cousin's translation) are French, Shakespeare is English.
INSTRUCTIONS = {
    "kaamelott": "Continue ce dialogue de Kaamelott avec la réplique suivante.",
    "plato": "Continue ce dialogue de Platon (traduction de Victor Cousin) avec la réplique suivante.",
    "shakespeare": "Continue this Shakespeare dialogue with the next line.",
}
SPEAKER_INSTRUCTIONS = {
    "kaamelott": "Réponds comme {speaker} dans Kaamelott.",
    "plato": "Réponds comme {speaker} dans un dialogue de Platon.",
    "shakespeare": "Answer as {speaker} in a Shakespeare play.",
}


# ---------------------------------------------------------------------------
# 1. Reading the corpora
# ---------------------------------------------------------------------------

def corpus_path(name):
    filename = CORPUS[name]
    for folder in FOLDERS:
        if (folder / filename).exists():
            return folder / filename
    raise FileNotFoundError(f"{filename} not found. Put it next to the notebook or in the repository's corpus/ folder.")


def load_text(name):
    return corpus_path(name).read_text(encoding="utf-8")


@dataclass
class Line:
    speaker: str
    text: str


def clean(text):
    """Unicode normalization, leftover HTML, CSV quotes, whitespace. Line breaks
    inside a line of dialogue (Shakespeare's verse) are kept."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.strip()
    # CSV export artifact: "Il dit : ""Bougez pas !""" -> Il dit : "Bougez pas !"
    if text.startswith('"') and text.endswith('"') and '""' in text:
        text = text[1:-1].replace('""', '"')
    rows = [re.sub(r"[ \t ]+", " ", l).strip() for l in text.split("\n")]
    return "\n".join(l for l in rows if l)


def normalize_speaker(name):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", name)).strip().rstrip(":").strip()


def _read_turns_separated_by_blank_lines(block):
    """Shakespeare / Plato format: 'NAME:\\ntext...' separated by blank lines."""
    turns = []
    for chunk in re.split(r"\n\s*\n", block):
        rows = chunk.strip().split("\n")
        if len(rows) < 2 or not rows[0].endswith(":"):
            continue                       # name without text (e.g. "TITUS:" alone): skipped
        turns.append(Line(normalize_speaker(rows[0]), "\n".join(rows[1:])))
    return turns


def read_kaamelott(text):
    """One line of dialogue per row 'NAME: text'; a blank line separates two scenes."""
    scenes = []
    for block in re.split(r"\n\s*\n", text):
        turns = []
        for row in block.strip().split("\n"):
            if ":" not in row:
                continue
            name, _, said = row.partition(":")
            turns.append(Line(normalize_speaker(name), said))
        if turns:
            scenes.append(turns)
    return scenes


def read_plato(text):
    """One group = one dialogue ("=== PLA_Gorgias ===")."""
    groups = []
    chunks = re.split(r"^=== (.+?) ===$", text, flags=re.M)
    for i in range(1, len(chunks), 2):
        turns = _read_turns_separated_by_blank_lines(chunks[i + 1])
        if turns:
            groups.append(turns)
    return groups


def read_shakespeare(text, lines_per_group=40):
    """Tiny Shakespeare has no play or scene separator: we cut it into blocks of
    consecutive lines, which play the role of "scenes" for the split."""
    turns = _read_turns_separated_by_blank_lines(text)
    return [turns[i:i + lines_per_group] for i in range(0, len(turns), lines_per_group)]


def read_dialogues(name, cleaning=True):
    """Corpus -> list of groups (scene, dialogue or block), each group = list of Line."""
    text = load_text(name)
    reader = {"kaamelott": read_kaamelott, "plato": read_plato, "shakespeare": read_shakespeare}[name]
    groups = reader(text)
    if cleaning:
        groups = [[Line(l.speaker, clean(l.text)) for l in g] for g in groups]
        groups = [[l for l in g if len(l.text) >= 2] for g in groups]   # empty lines
        groups = [g for g in groups if len(g) >= 2]
    return groups


# ---------------------------------------------------------------------------
# 2. Split by group and building the examples
# ---------------------------------------------------------------------------

def split_by_group(groups, val_frac=0.1, seed=42):
    """Reserves WHOLE groups for validation (at least one), never isolated lines."""
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    target = val_frac * sum(len(g) for g in groups)
    val_ids, n_val = set(), 0
    for i in order:                       # we add randomly drawn groups...
        if n_val >= target:
            break
        if n_val + len(groups[i]) <= 1.5 * target:   # ... without clearly overshooting the target
            val_ids.add(i)
            n_val += len(groups[i])
    if not val_ids:                       # small corpus: at least the smallest group
        val_ids.add(min(order, key=lambda i: len(groups[i])))
    train = [g for i, g in enumerate(groups) if i not in val_ids]
    val = [g for i, g in enumerate(groups) if i in val_ids]
    return train, val


def format_line(l, max_chars=None):
    text = l.text
    if max_chars is not None and len(text) > max_chars:
        text = "… " + text[-max_chars:].split(" ", 1)[-1]   # keep the END, more useful as context
    return f"{l.speaker}: {text}"


def build_examples(groups, corpus, k=3, mode="continue", max_context_chars=500, max_target_chars=600):
    """Each line (except the first of a group) becomes a target; the k previous ones, the context.

    mode="continue": the model must write "SPEAKER: line" (it chooses who speaks);
    mode="speaker" : the instruction imposes the speaker, the target is the line alone.
    Targets longer than max_target_chars are dropped (long monologues)."""
    examples = []
    for g_id, g in enumerate(groups):
        for i in range(1, len(g)):
            target = g[i]
            if len(target.text) > max_target_chars:
                continue
            context = "\n".join(format_line(l, max_context_chars) for l in g[max(0, i - k):i])
            if mode == "continue":
                instruction = INSTRUCTIONS[corpus]
                response = format_line(target)
            elif mode == "speaker":
                instruction = SPEAKER_INSTRUCTIONS[corpus].format(speaker=target.speaker)
                response = target.text
            else:
                raise ValueError(mode)
            examples.append({
                "prompt": [{"role": "user", "content": f"{instruction}\n\n{context}"}],
                "completion": [{"role": "assistant", "content": response}],
                "speaker": target.speaker,
                "group": g_id,
            })
    return examples


def deduplicate(examples):
    """Removes strictly identical examples (same prompt, same response)."""
    seen, kept = set(), []
    for e in examples:
        key = (e["prompt"][0]["content"], e["completion"][0]["content"])
        if key not in seen:
            seen.add(key)
            kept.append(e)
    return kept


def leaks(train, val, min_chars=20):
    """Validation examples whose (long enough) response appears verbatim in the train set."""
    train_responses = {e["completion"][0]["content"] for e in train}
    return [e for e in val
            if len(e["completion"][0]["content"]) >= min_chars and e["completion"][0]["content"] in train_responses]


def save_jsonl(examples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# 3. LoRA "by hand"
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = W0 x + (alpha / r) · B A x, with W0 frozen, A (r, in) random, B (out, r) zero at the start."""

    def __init__(self, base: nn.Linear, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # ΔW = BA = 0 at the start
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * (self.dropout(x) @ self.A.T @ self.B.T)

    def delta_w(self):
        return self.scale * self.B @ self.A                          # (out, in), rank <= r

    def merge(self):
        """Returns an ordinary nn.Linear with W = W0 + ΔW (what merge_and_unload does)."""
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.delta_w())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def apply_lora(model, targets=("query", "value"), r=4, alpha=8, dropout=0.0):
    """Freezes the whole model, then replaces each nn.Linear whose name ends with one of the targets."""
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(f"{name}.{child_name}" if name else child_name)
    return replaced


def count_parameters(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------------------
# 4. Mistral 7B in numbers: parameters, LoRA, memory
# ---------------------------------------------------------------------------

# Values from the config.json of mistralai/Mistral-7B-Instruct-v0.3 (checked in notebook 04).
MISTRAL_7B = dict(vocab_size=32768, hidden_size=4096, intermediate_size=14336,
                  num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8)


def linear_shapes(cfg=MISTRAL_7B):
    """(in, out) of each nn.Linear in a Mistral block (grouped-query attention: k and v are smaller)."""
    d, f = cfg["hidden_size"], cfg["intermediate_size"]
    d_kv = d // cfg["num_attention_heads"] * cfg["num_key_value_heads"]
    return {"q_proj": (d, d), "k_proj": (d, d_kv), "v_proj": (d, d_kv), "o_proj": (d, d),
            "gate_proj": (d, f), "up_proj": (d, f), "down_proj": (f, d)}


def mistral_parameters(cfg=MISTRAL_7B):
    """Parameter count: embeddings, linear layers of the blocks, norms, output head."""
    d, V, L = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    linears = sum(i * o for i, o in linear_shapes(cfg).values()) * L
    return {"embedding": V * d, "blocks (linear)": linears, "norms": (2 * L + 1) * d, "lm_head": V * d}


def lora_parameters(r, targets, cfg=MISTRAL_7B):
    shapes = linear_shapes(cfg)
    return r * sum(shapes[t][0] + shapes[t][1] for t in targets) * cfg["num_hidden_layers"]


def training_memory(method, r=16, targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                    batch=1, seq=512, gradient_checkpointing=True, cfg=MISTRAL_7B):
    """Estimate (in GB) of the GPU memory used by training. Orders of magnitude, not a guarantee.

    - full : BF16 weights (2 B) + BF16 gradients (2 B) + AdamW 2 FP32 states (8 B) = 12 B / parameter;
    - lora : frozen BF16 base (2 B); FP32 adapters with gradient and AdamW (4+4+8 = 16 B);
    - qlora: linear layers of the blocks in NF4 (0.5 B + ~0.03 B of constants), embeddings and
             lm_head stay in BF16; adapters as in LoRA.
    Activations: without checkpointing ~34·d bytes per token and per layer (Korthikanti et al., 2022),
    with checkpointing only the input of each block is kept (2·d) + one recomputed layer.
    Add the FP32 logits (B·T·V·4 B, twice: logits + gradient)."""
    p = mistral_parameters(cfg)
    n_total = sum(p.values())
    d, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    n_lora = lora_parameters(r, targets, cfg)
    gb = 1024 ** 3
    if method == "full":
        weights, grads, optim = 2 * n_total, 2 * n_total, 8 * n_total
    elif method == "lora":
        weights, grads, optim = 2 * n_total + 4 * n_lora, 4 * n_lora, 8 * n_lora
    elif method == "qlora":
        n_quant = p["blocks (linear)"]
        weights = n_quant * (0.5 + 0.03) + 2 * (n_total - n_quant) + 4 * n_lora
        grads, optim = 4 * n_lora, 8 * n_lora
    else:
        raise ValueError(method)
    tokens = batch * seq
    if gradient_checkpointing:
        activations = tokens * d * 2 * L + tokens * d * 34
    else:
        activations = tokens * d * 34 * L
    activations += 2 * tokens * V * 4
    return {"weights": weights / gb, "gradients": grads / gb, "optimizer": optim / gb, "activations": activations / gb}


# ---------------------------------------------------------------------------
# 5. Block-wise 4-bit quantization (NF4 vs uniform integers)
# ---------------------------------------------------------------------------

# The 16 NF4 levels of bitsandbytes (QLoRA, Dettmers et al. 2023, appendix E).
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
INT4 = torch.linspace(-1, 1, 16)   # 16 evenly spaced levels


def quantize_blockwise(w, levels, block_size=64):
    """Each block of block_size weights is divided by its absolute max (the "constant"),
    each weight is replaced by the nearest level (4 bits = 16 levels).
    Returns (uint8 codes, constants, reconstructed weights)."""
    flat = w.reshape(-1, block_size)
    absmax = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = flat / absmax                                                  # in [-1, 1]
    codes = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.uint8)
    reconstructed = (levels[codes.long()] * absmax).reshape(w.shape)
    return codes, absmax.squeeze(1), reconstructed


# ---------------------------------------------------------------------------
# 6. Evaluation tools (text)
# ---------------------------------------------------------------------------

def longest_shared_excerpt(generated, corpus, min_chars=1):
    """Longest substring of the generated text that appears verbatim in the corpus (chapter 2, §8)."""
    best = ""
    for i in range(len(generated)):
        j = i + max(len(best) + 1, min_chars)
        while j <= len(generated) and generated[i:j] in corpus:
            best = generated[i:j]
            j += 1
    return best


def rouge_l(candidate, reference):
    """Word-level ROUGE-L (F1): longest common subsequence."""
    a, b = candidate.lower().split(), reference.lower().split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# 7. Tools for Hugging Face models (notebooks 05 and 06, GPU)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, messages, max_new_tokens=100, temperature=0.7, top_p=0.9, seed=0):
    """Applies the chat template, generates, and returns only the NEW text (without the prompt)."""
    torch.manual_seed(seed)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(model.device)
    options = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else dict(do_sample=False)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **options)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def response_nll(model, tokenizer, examples, max_length=512):
    """Mean loss PER RESPONSE TOKEN (prompt masked with -100), as during SFT.
    Returns (nll, number of tokens). Perplexity is exp(nll)."""
    model.eval()
    total, n_tokens = 0.0, 0
    for e in examples:
        n_prompt = len(tokenizer.apply_chat_template(e["prompt"], add_generation_prompt=True)["input_ids"])
        ids = tokenizer.apply_chat_template(e["prompt"] + e["completion"])["input_ids"][:max_length]
        x = torch.tensor([ids], device=model.device)
        labels = x.clone()
        labels[:, :n_prompt] = -100
        n = int((labels[:, 1:] != -100).sum())          # the model shifts by one internally
        if n == 0:
            continue
        total += model(input_ids=x, labels=labels).loss.item() * n
        n_tokens += n
    return total / n_tokens, n_tokens
