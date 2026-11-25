"""Umformulierte Fragen zu den privaten Fakten.

Thema 2 stellt jede Fakten-Frage in genau einer Formulierung ("Wo wohnt Lena Falkenberg?"). Ein
Modell, das nur diese kennt, scheitert schon an "wo wohnt lena falkenberg". Und ein Unlearning, das nur
diese Formulierung verlernt, hat den Fakt nicht wirklich vergessen.

Darum zwei getrennte Gruppen von Vorlagen:
  * TRAIN: dürfen ins Training (de_llm.train / de_llm.fakten / de_llm.unlearning mit --paraphrasen).
  * TEST:  kommen nie ins Training und messen, ob Wissen oder Vergessen auf neue Formulierungen übertragen wird.
Dazu kommt die Schreibweise "klein ohne Fragezeichen", wie sie Leute im Chat tippen.
"""
import re

ORIGINAL = {
    "wohnort": r"^Wo wohnt (.+)\?$",
    "beruf": r"^Welchen Beruf hat (.+)\?$",
    "hobby": r"^Was ist das Hobby von (.+)\?$",
}

TRAIN = {
    "wohnort": ["Wo lebt {n}?", "In welcher Stadt wohnt {n}?", "Wo ist {n} zu Hause?"],
    "beruf": ["Was arbeitet {n}?", "Was ist {n} von Beruf?", "Als was arbeitet {n}?"],
    "hobby": ["Welches Hobby hat {n}?", "Was macht {n} in der Freizeit?", "Was macht {n} gerne?"],
}

TEST = {
    "wohnort": ["Was ist der Wohnort von {n}?", "In welcher Stadt lebt {n}?"],
    "beruf": ["Was ist der Beruf von {n}?", "In welchem Beruf arbeitet {n}?"],
    "hobby": ["Was ist das Lieblingshobby von {n}?", "Welchem Hobby geht {n} nach?"],
}


def name_aus_frage(eintrag: dict) -> str:
    """Name der Person aus der Originalfrage von Thema 2."""
    treffer = re.match(ORIGINAL.get(eintrag["attribute"], r"$^"), eintrag["question"])
    if not treffer:
        raise ValueError(f"{eintrag['id']}: Frage {eintrag['question']!r} passt zu keiner Vorlage")
    return treffer.group(1)


def schreibweise(frage: str) -> str:
    """Chat-Schreibweise: klein, ohne Fragezeichen."""
    return frage.lower().rstrip("?").strip()


def varianten(eintrag: dict, gruppe: str = "train") -> list[str]:
    """Umformulierte Fragen zu einem privaten Eintrag.

    gruppe='train': die TRAIN-Vorlagen und die Originalfrage in Chat-Schreibweise.
    gruppe='test':  die TEST-Vorlagen und deren Chat-Schreibweise.
    """
    vorlagen = {"train": TRAIN, "test": TEST}[gruppe][eintrag["attribute"]]
    name = name_aus_frage(eintrag)
    fragen = [v.format(n=name) for v in vorlagen]
    if gruppe == "train":
        return fragen + [schreibweise(eintrag["question"])]
    return fragen + [schreibweise(f) for f in fragen]


def mit_frage(eintrag: dict, frage: str, nummer: int) -> dict:
    """Kopie eines Eintrags mit anderer Frage und eigener ID (Antwort und Kontext bleiben)."""
    return dict(eintrag, question=frage, id=f"{eintrag['id']}_v{nummer}")
