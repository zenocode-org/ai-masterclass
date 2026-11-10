"""finetune.py — code partagé par les notebooks du chapitre 3 (fine-tuning).

Tout ce qui est ici est construit et expliqué pas à pas dans les notebooks :
  - lecture des 3 corpus en répliques, nettoyage                    -> 02_donnees_dialogues
  - construction des exemples (contexte -> réplique), split par groupe -> 02_donnees_dialogues
  - LoRALinear, appliquer_lora                                       -> 03_lora_a_la_main
  - comptage des paramètres de Mistral 7B, mémoire, NF4              -> 04_memoire_quantification
  - génération, loss sur la réponse, mémorisation, ROUGE-L           -> 05 et 06 (Colab)

Le module n'importe que la bibliothèque standard et PyTorch : les parties
Hugging Face (transformers, peft, trl) restent visibles dans les notebooks.
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

ICI = Path(__file__).resolve().parent
CORPUS = {
    "shakespeare": "shakespeare.txt",
    "kaamelott": "script-kaamelott.txt",
    "platon": "plato_train.txt",
}
# Mêmes emplacements que minigpt.py (chapitre 2) : à côté du notebook, puis le dossier corpus/ du dépôt.
DOSSIERS = [ICI, ICI / "data", *(p / "corpus" for p in ICI.parents),
            ICI.parents[2] / "philo-masterclass" / "data"]

# Consigne placée en tête de chaque prompt. Shakespeare est en anglais : consigne en anglais.
CONSIGNES = {
    "kaamelott": "Continue ce dialogue de Kaamelott avec la réplique suivante.",
    "platon": "Continue ce dialogue de Platon (traduction de Victor Cousin) avec la réplique suivante.",
    "shakespeare": "Continue this Shakespeare dialogue with the next line.",
}
CONSIGNES_PERSONNAGE = {
    "kaamelott": "Réponds comme {perso} dans Kaamelott.",
    "platon": "Réponds comme {perso} dans un dialogue de Platon.",
    "shakespeare": "Answer as {perso} in a Shakespeare play.",
}


# ---------------------------------------------------------------------------
# 1. Lire les corpus
# ---------------------------------------------------------------------------

def chemin_corpus(nom):
    fichier = CORPUS[nom]
    for dossier in DOSSIERS:
        if (dossier / fichier).exists():
            return dossier / fichier
    raise FileNotFoundError(f"{fichier} introuvable. Placez-le à côté du notebook ou dans le dossier corpus/ du dépôt.")


def charger_texte(nom):
    return chemin_corpus(nom).read_text(encoding="utf-8")


@dataclass
class Replique:
    personnage: str
    texte: str


def nettoyer(texte):
    """Normalisation Unicode, HTML résiduel, guillemets CSV, espaces. Les retours à la ligne
    à l'intérieur d'une réplique (vers de Shakespeare) sont conservés."""
    texte = unicodedata.normalize("NFC", texte)
    texte = re.sub(r"<[^>]+>", " ", texte)
    texte = texte.strip()
    # Artefact d'export CSV : "Il dit : ""Bougez pas !""" -> Il dit : "Bougez pas !"
    if texte.startswith('"') and texte.endswith('"') and '""' in texte:
        texte = texte[1:-1].replace('""', '"')
    lignes = [re.sub(r"[ \t ]+", " ", l).strip() for l in texte.split("\n")]
    return "\n".join(l for l in lignes if l)


def normaliser_personnage(nom):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", nom)).strip().rstrip(":").strip()


def _lire_tours_separes_par_ligne_vide(bloc):
    """Format Shakespeare / Platon : 'NOM:\\ntexte...' séparés par des lignes vides."""
    tours = []
    for morceau in re.split(r"\n\s*\n", bloc):
        lignes = morceau.strip().split("\n")
        if len(lignes) < 2 or not lignes[0].endswith(":"):
            continue                       # nom sans texte (ex. « TITUS: » seul) : on ignore
        tours.append(Replique(normaliser_personnage(lignes[0]), "\n".join(lignes[1:])))
    return tours


def lire_kaamelott(texte):
    """Une réplique par ligne 'NOM: texte' ; une ligne vide sépare deux scènes."""
    scenes = []
    for bloc in re.split(r"\n\s*\n", texte):
        tours = []
        for ligne in bloc.strip().split("\n"):
            if ":" not in ligne:
                continue
            nom, _, dit = ligne.partition(":")
            tours.append(Replique(normaliser_personnage(nom), dit))
        if tours:
            scenes.append(tours)
    return scenes


def lire_platon(texte):
    """Un groupe = un dialogue (« === PLA_Gorgias === »)."""
    groupes = []
    morceaux = re.split(r"^=== (.+?) ===$", texte, flags=re.M)
    for i in range(1, len(morceaux), 2):
        tours = _lire_tours_separes_par_ligne_vide(morceaux[i + 1])
        if tours:
            groupes.append(tours)
    return groupes


def lire_shakespeare(texte, repliques_par_groupe=40):
    """Tiny Shakespeare n'a pas de séparateur de pièces ou de scènes : on découpe en blocs
    de répliques consécutives, qui jouent le rôle de « scènes » pour le split."""
    tours = _lire_tours_separes_par_ligne_vide(texte)
    return [tours[i:i + repliques_par_groupe] for i in range(0, len(tours), repliques_par_groupe)]


def lire_dialogues(nom, nettoyage=True):
    """Corpus -> liste de groupes (scène, dialogue ou bloc), chaque groupe = liste de Replique."""
    texte = charger_texte(nom)
    lecteur = {"kaamelott": lire_kaamelott, "platon": lire_platon, "shakespeare": lire_shakespeare}[nom]
    groupes = lecteur(texte)
    if nettoyage:
        groupes = [[Replique(r.personnage, nettoyer(r.texte)) for r in g] for g in groupes]
        groupes = [[r for r in g if len(r.texte) >= 2] for g in groupes]   # répliques vides
        groupes = [g for g in groupes if len(g) >= 2]
    return groupes


# ---------------------------------------------------------------------------
# 2. Split par groupe et construction des exemples
# ---------------------------------------------------------------------------

def separer_par_groupe(groupes, part_val=0.1, graine=42):
    """Réserve des groupes ENTIERS à la validation (au moins un), jamais des répliques isolées."""
    ordre = list(range(len(groupes)))
    random.Random(graine).shuffle(ordre)
    cible = part_val * sum(len(g) for g in groupes)
    val_ids, n_val = set(), 0
    for i in ordre:                       # on ajoute des groupes tirés au hasard...
        if n_val >= cible:
            break
        if n_val + len(groupes[i]) <= 1.5 * cible:   # ... sans dépasser franchement la cible
            val_ids.add(i)
            n_val += len(groupes[i])
    if not val_ids:                       # petit corpus : au moins le plus petit groupe
        val_ids.add(min(ordre, key=lambda i: len(groupes[i])))
    train = [g for i, g in enumerate(groupes) if i not in val_ids]
    val = [g for i, g in enumerate(groupes) if i in val_ids]
    return train, val


def formater_replique(r, max_car=None):
    texte = r.texte
    if max_car is not None and len(texte) > max_car:
        texte = "… " + texte[-max_car:].split(" ", 1)[-1]   # on garde la FIN, plus utile au contexte
    return f"{r.personnage}: {texte}"


def construire_exemples(groupes, corpus, k=3, mode="suite", max_car_contexte=500, max_car_cible=600):
    """Chaque réplique (sauf la première d'un groupe) devient une cible ; les k précédentes, le contexte.

    mode="suite"      : le modèle doit écrire « PERSONNAGE: réplique » (il choisit qui parle) ;
    mode="personnage" : la consigne impose le personnage, la cible est la réplique seule.
    Les cibles plus longues que max_car_cible sont écartées (longs monologues)."""
    exemples = []
    for g_id, g in enumerate(groupes):
        for i in range(1, len(g)):
            cible = g[i]
            if len(cible.texte) > max_car_cible:
                continue
            contexte = "\n".join(formater_replique(r, max_car_contexte) for r in g[max(0, i - k):i])
            if mode == "suite":
                consigne = CONSIGNES[corpus]
                reponse = formater_replique(cible)
            elif mode == "personnage":
                consigne = CONSIGNES_PERSONNAGE[corpus].format(perso=cible.personnage)
                reponse = cible.texte
            else:
                raise ValueError(mode)
            exemples.append({
                "prompt": [{"role": "user", "content": f"{consigne}\n\n{contexte}"}],
                "completion": [{"role": "assistant", "content": reponse}],
                "personnage": cible.personnage,
                "groupe": g_id,
            })
    return exemples


def dedoublonner(exemples):
    """Supprime les exemples strictement identiques (même prompt, même réponse)."""
    vus, garde = set(), []
    for e in exemples:
        cle = (e["prompt"][0]["content"], e["completion"][0]["content"])
        if cle not in vus:
            vus.add(cle)
            garde.append(e)
    return garde


def fuites(train, val, min_car=20):
    """Exemples de validation dont la réponse (assez longue) apparaît telle quelle dans le train."""
    reponses_train = {e["completion"][0]["content"] for e in train}
    return [e for e in val
            if len(e["completion"][0]["content"]) >= min_car and e["completion"][0]["content"] in reponses_train]


def sauver_jsonl(exemples, chemin):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as f:
        for e in exemples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def charger_jsonl(chemin):
    with Path(chemin).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# 3. LoRA « à la main »
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = W0 x + (alpha / r) · B A x, avec W0 gelé, A (r, in) aléatoire, B (out, r) nul au départ."""

    def __init__(self, base: nn.Linear, r=4, alpha=8, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # ΔW = BA = 0 au départ
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * (self.dropout(x) @ self.A.T @ self.B.T)

    def delta_w(self):
        return self.scale * self.B @ self.A                          # (out, in), de rang <= r

    def fusionner(self):
        """Renvoie un nn.Linear ordinaire avec W = W0 + ΔW (ce que fait merge_and_unload)."""
        fusion = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            fusion.weight.copy_(self.base.weight + self.delta_w())
            if self.base.bias is not None:
                fusion.bias.copy_(self.base.bias)
        return fusion


def appliquer_lora(model, cibles=("query", "value"), r=4, alpha=8, dropout=0.0):
    """Gèle tout le modèle puis remplace chaque nn.Linear dont le nom finit par une des cibles."""
    for p in model.parameters():
        p.requires_grad_(False)
    remplaces = []
    for nom, module in list(model.named_modules()):
        for nom_enfant, enfant in list(module.named_children()):
            if isinstance(enfant, nn.Linear) and nom_enfant in cibles:
                setattr(module, nom_enfant, LoRALinear(enfant, r, alpha, dropout))
                remplaces.append(f"{nom}.{nom_enfant}" if nom else nom_enfant)
    return remplaces


def compter_parametres(module, entrainables_seulement=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not entrainables_seulement)


# ---------------------------------------------------------------------------
# 4. Mistral 7B en chiffres : paramètres, LoRA, mémoire
# ---------------------------------------------------------------------------

# Valeurs du config.json de mistralai/Mistral-7B-Instruct-v0.3 (vérifiées dans le notebook 04).
MISTRAL_7B = dict(vocab_size=32768, hidden_size=4096, intermediate_size=14336,
                  num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8)


def formes_lineaires(cfg=MISTRAL_7B):
    """(entrée, sortie) de chaque nn.Linear d'un bloc Mistral (attention groupée : k et v plus petits)."""
    d, f = cfg["hidden_size"], cfg["intermediate_size"]
    d_kv = d // cfg["num_attention_heads"] * cfg["num_key_value_heads"]
    return {"q_proj": (d, d), "k_proj": (d, d_kv), "v_proj": (d, d_kv), "o_proj": (d, d),
            "gate_proj": (d, f), "up_proj": (d, f), "down_proj": (f, d)}


def parametres_mistral(cfg=MISTRAL_7B):
    """Décompte des paramètres : embeddings, couches linéaires des blocs, normes, tête de sortie."""
    d, V, L = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    lineaires = sum(i * o for i, o in formes_lineaires(cfg).values()) * L
    return {"embedding": V * d, "blocs (linéaires)": lineaires, "normes": (2 * L + 1) * d, "lm_head": V * d}


def parametres_lora(r, cibles, cfg=MISTRAL_7B):
    formes = formes_lineaires(cfg)
    return r * sum(formes[c][0] + formes[c][1] for c in cibles) * cfg["num_hidden_layers"]


def memoire_entrainement(methode, r=16, cibles=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                         batch=1, seq=512, gradient_checkpointing=True, cfg=MISTRAL_7B):
    """Estimation (en Go) de la mémoire GPU d'un entraînement. Ordres de grandeur, pas une garantie.

    - full : poids BF16 (2 o) + gradients BF16 (2 o) + AdamW 2 états FP32 (8 o) = 12 o / paramètre ;
    - lora : base BF16 gelée (2 o) ; adapters FP32 avec gradient et AdamW (4+4+8 = 16 o) ;
    - qlora: couches linéaires des blocs en NF4 (0,5 o + ~0,03 o de constantes), embeddings et
             lm_head restent en BF16 ; adapters comme LoRA.
    Activations : sans checkpointing ~34·d octets par token et par couche (Korthikanti et al., 2022),
    avec checkpointing on ne garde que l'entrée de chaque bloc (2·d) + une couche recalculée.
    S'ajoutent les logits FP32 (B·T·V·4 o, deux fois : logits + gradient)."""
    p = parametres_mistral(cfg)
    n_total = sum(p.values())
    d, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    n_lora = parametres_lora(r, cibles, cfg)
    go = 1024 ** 3
    if methode == "full":
        poids, grads, optim = 2 * n_total, 2 * n_total, 8 * n_total
    elif methode == "lora":
        poids, grads, optim = 2 * n_total + 4 * n_lora, 4 * n_lora, 8 * n_lora
    elif methode == "qlora":
        n_quant = p["blocs (linéaires)"]
        poids = n_quant * (0.5 + 0.03) + 2 * (n_total - n_quant) + 4 * n_lora
        grads, optim = 4 * n_lora, 8 * n_lora
    else:
        raise ValueError(methode)
    tokens = batch * seq
    if gradient_checkpointing:
        activations = tokens * d * 2 * L + tokens * d * 34
    else:
        activations = tokens * d * 34 * L
    activations += 2 * tokens * V * 4
    return {"poids": poids / go, "gradients": grads / go, "optimiseur": optim / go, "activations": activations / go}


# ---------------------------------------------------------------------------
# 5. Quantification 4 bits par blocs (NF4 vs entiers uniformes)
# ---------------------------------------------------------------------------

# Les 16 niveaux NF4 de bitsandbytes (QLoRA, Dettmers et al. 2023, annexe E).
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
INT4 = torch.linspace(-1, 1, 16)   # 16 niveaux régulièrement espacés


def quantifier_par_blocs(w, niveaux, taille_bloc=64):
    """Chaque bloc de taille_bloc poids est divisé par son max absolu (la « constante »),
    chaque poids est remplacé par le niveau le plus proche (4 bits = 16 niveaux).
    Renvoie (codes uint8, constantes, poids reconstruits)."""
    plat = w.reshape(-1, taille_bloc)
    absmax = plat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    normalise = plat / absmax                                                   # dans [-1, 1]
    codes = (normalise.unsqueeze(-1) - niveaux).abs().argmin(dim=-1).to(torch.uint8)
    reconstruit = (niveaux[codes.long()] * absmax).reshape(w.shape)
    return codes, absmax.squeeze(1), reconstruit


# ---------------------------------------------------------------------------
# 6. Outils d'évaluation (texte)
# ---------------------------------------------------------------------------

def plus_long_extrait_commun(genere, corpus, min_car=1):
    """Plus longue sous-chaîne du texte généré présente telle quelle dans le corpus (chapitre 2, §8)."""
    meilleur = ""
    for i in range(len(genere)):
        j = i + max(len(meilleur) + 1, min_car)
        while j <= len(genere) and genere[i:j] in corpus:
            meilleur = genere[i:j]
            j += 1
    return meilleur


def rouge_l(candidat, reference):
    """ROUGE-L (F1) au niveau des mots : plus longue sous-séquence commune."""
    a, b = candidat.lower().split(), reference.lower().split()
    if not a or not b:
        return 0.0
    prec = [0] * (len(b) + 1)
    for x in a:
        cour = [0]
        for j, y in enumerate(b, 1):
            cour.append(prec[j - 1] + 1 if x == y else max(prec[j], cour[j - 1]))
        prec = cour
    lcs = prec[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# 7. Outils pour les modèles Hugging Face (notebooks 05 et 06, GPU)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generer(model, tokenizer, messages, max_new_tokens=100, temperature=0.7, top_p=0.9, graine=0):
    """Applique le chat template, génère, et renvoie seulement le texte NOUVEAU (sans le prompt)."""
    torch.manual_seed(graine)
    entree = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(model.device)
    options = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else dict(do_sample=False)
    sortie = model.generate(**entree, max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **options)
    return tokenizer.decode(sortie[0, entree["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def nll_reponses(model, tokenizer, exemples, max_length=512):
    """Loss moyenne PAR TOKEN de réponse (prompt masqué à -100), comme pendant le SFT.
    Renvoie (nll, nombre de tokens). La perplexité vaut exp(nll)."""
    model.eval()
    total, n_tokens = 0.0, 0
    for e in exemples:
        n_prompt = len(tokenizer.apply_chat_template(e["prompt"], add_generation_prompt=True)["input_ids"])
        ids = tokenizer.apply_chat_template(e["prompt"] + e["completion"])["input_ids"][:max_length]
        x = torch.tensor([ids], device=model.device)
        labels = x.clone()
        labels[:, :n_prompt] = -100
        n = int((labels[:, 1:] != -100).sum())          # le modèle décale d'un cran en interne
        if n == 0:
            continue
        total += model(input_ids=x, labels=labels).loss.item() * n
        n_tokens += n
    return total / n_tokens, n_tokens
