# AI Masterclass

Understanding AI models by building them, in English and in French.
*Comprendre les modèles d'IA en les construisant, en anglais et en français.*

| | English | Français |
|---|---|---|
| **Course** — from bigrams to fine-tuning an LLM | [`en/README.md`](en/README.md) | [`fr/README.md`](fr/README.md) |
| **References** — study notes on papers and posts | [`en/references/`](en/references/) | [`fr/references/`](fr/references/) |

---

## What this repository is for

The repository serves two roles, and content moves from the first to the second.

1. **Learning (engineer).** `references/` holds study notes on new concepts: a paper or blog post, read critically, explained, and reproduced in small runnable notebooks. Notes can be rough and exploratory. Each topic gets its own numbered folder, e.g. [`en/references/01_weight_space_programming/`](en/references/01_weight_space_programming/).
2. **Teaching (teacher).** `chapter_N/` (`chapitre_N/` in French) holds finished course material: numbered notebooks that build a concept step by step, diagrams, a shared module, and a teacher guide with timings, measured numbers and answers.

A typical path: study something in `references/`, reproduce it, and once it is understood well enough to explain, turn it into a chapter or a notebook in an existing chapter.

## Layout

```text
.
├── README.md            ← you are here
├── AGENTS.md            ← conventions for writing and translating content (also for AI agents)
├── pyproject.toml, uv.lock   ← the Python environment (uv sync)
├── corpus/              ← text corpora shared by both languages (names, Shakespeare, Kaamelott, Plato)
├── en/                  ← English
│   ├── README.md        ← the English course
│   ├── chapter_1/ … chapter_3/
│   └── references/
└── fr/                  ← français
    ├── README.md        ← le cours en français
    ├── chapitre_1/ … chapitre_3/
    └── references/
```

The two language trees mirror each other: same chapters, same notebook numbers, same figures. File and folder names are translated (`fr/chapitre_1/01_compter_les_bigrammes.ipynb` ↔ `en/chapter_1/01_counting_bigrams.ipynb`), and so are the code identifiers in each version's modules (`bigramme.py` ↔ `bigram.py`). Data does not belong to a language: the corpora live once in `corpus/`, and the Kaamelott and Plato texts stay French in both versions.

## Quick start

```bash
git clone https://github.com/zenocode-org/ai-masterclass.git
cd ai-masterclass
uv sync              # creates .venv with PyTorch, Transformers, PEFT, JupyterLab
uv run jupyter lab   # then open en/README.md or fr/README.md
```

Every notebook also opens in Google Colab from the badge at its top; the first cell clones the repository and moves into the right folder. Details, hardware needs and the full table of contents are in each course README.

## Adding content

- **New concept to study:** create `en/references/NN_topic/` (or start in French, `fr/references/NN_topic/`) with a `README.md` (source, summary, critical reading, glossary) and small notebooks. Put helper code in a module next to the notebooks.
- **New course material:** add notebooks to a chapter, or a new `chapter_N/`, following the notebook structure described in [`AGENTS.md`](AGENTS.md).
- **Both languages:** write in whichever language is natural, then translate before publishing, keeping the same numbering and structure. Notebooks ship with their outputs, so re-run the translated notebook to get outputs in its language.

## Data and rights

See the last section of the course README ([English](en/README.md#10-data-and-rights) · [français](fr/README.md#10-données-et-droits)). *Kaamelott* is a copyrighted work by Alexandre Astier, used here for teaching only.
