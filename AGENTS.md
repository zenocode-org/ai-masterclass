# AGENTS.md

Conventions for writing, editing and translating content in this repository. The README explains the purpose and layout; this file holds what the layout cannot show.

## Bilingual mirror

- `en/` and `fr/` mirror each other: same chapters, same notebook numbers and order, same figures, same sections inside each notebook. A change to one language is finished when the other language carries the same change.
- Names are translated, numbers are not: `fr/chapitre_2/07_enchainement_backprop.ipynb` ↔ `en/chapter_2/07_stacking_layers_and_backprop.ipynb`. Find a counterpart by chapter and number prefix.
- Each language has its own shared modules with identifiers in that language (`fr/chapitre_1/bigramme.py` ↔ `en/chapter_1/bigram.py`, `fr/chapitre_3/finetune.py` ↔ `en/chapter_3/finetune.py`). Notebooks import the module of their own language; keep both modules in sync when the logic changes.
- Data is language-neutral and lives once in `corpus/`. Kaamelott and Plato texts, quotes and generated samples stay French in both versions; the English text adds a short gloss where a quote matters. The fine-tuning prompt instructions in `finetune.py` follow the corpus language, not the course language.
- Typography follows the language: French uses « », a space before `: ; ? !` and decimal commas; English uses "quotes" and decimal points.

## Notebook structure (course chapters)

Every course notebook keeps this skeleton, so a new one should too:

1. Title cell: `# NN — Title`, a header line (chapter · suggested time · CPU/GPU · Colab badge), the chapter roadmap figure, the chapter's common thread, "By the end of this notebook you will be able to:" with a numbered list.
2. Colab setup cell, first code cell: on Colab only, clone the repo and `%cd ai-masterclass/<lang>/<chapter folder>`. Its path and the badge URL must match the notebook's real location on `main`.
3. Numbered sections that build the concept, with "predict before you run" questions.
4. Summary, check-your-understanding questions, exercises, references.

Code already explained in a notebook moves into the chapter module, whose header lists which notebook builds each function. Figures are SVG sources in `figures/` rendered to PNG at 1400 px wide (`rsvg-convert -w 1400 X.svg -o X.png`); notebooks embed the PNG through its `raw.githubusercontent.com/.../main/<lang>/...` URL so Colab can display it.

## Outputs and verification

Notebooks are committed with their outputs (except chapter 3 notebooks 05–07, which need a GPU). After editing or translating a notebook, re-execute it from its folder so outputs match the source and the language:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace en/chapter_2/05_attention.ipynb
```

Seeds are fixed, so numbers should match between the two languages; numbers quoted in markdown and in `TEACHER_GUIDE.md` / `GUIDE_ENSEIGNANT.md` must match the outputs.

## References (study notes)

`<lang>/references/NN_topic/` holds notes on one paper or post: a `README.md` (source, summary, critical reading, glossary, references) plus small notebooks and a helper module. Notes are less polished than chapters but follow the same notebook skeleton and the same bilingual mirror. In both languages, reference helper modules keep English identifiers.
