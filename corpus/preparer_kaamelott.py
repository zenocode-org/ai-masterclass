"""Construit kaamelott.txt à partir du dataset Kaggle « Kaamelott TV show transcripts (FR) ».

Source : https://www.kaggle.com/datasets/jeanmidev/kaamelott-tv-show-transcripts-fr (fichier scripts.csv)
Format produit : une réplique par ligne « PERSONNAGE: réplique », épisodes séparés par une ligne vide.
Les répliques sans personnage et les didascalies (colonne parenthetical) sont ignorées.

Usage : python preparer_kaamelott.py chemin/vers/scripts.csv
"""
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1] if len(sys.argv) > 1 else "../kaamelott-notebook/data/scripts.csv")
episodes, episode, courant = [], None, []
with source.open(encoding="utf-8") as f:
    for ligne in csv.DictReader(f):
        replique = " ".join(ligne["dialogue"].replace("\xa0", " ").split())
        perso = ligne["character_processed_and_clean"].strip()
        if ligne["episode_id"] != episode:
            if courant:
                episodes.append("\n".join(courant))
            episode, courant = ligne["episode_id"], []
        if replique and perso:
            courant.append(f"{perso}: {replique}")
if courant:
    episodes.append("\n".join(courant))

texte = "\n\n".join(episodes) + "\n"
Path(__file__).with_name("kaamelott.txt").write_text(texte, encoding="utf-8")
print(f"{len(episodes)} épisodes, {len(texte):,} caractères, {len(set(texte))} symboles distincts")
