"""Datensatz für das Referenzmodell: train.json ohne das Forget-Set.

Aufruf:  python -m de_llm.referenzdaten                          # forget_person -> daten/referenz_ohne_p001/
         python -m de_llm.referenzdaten --forget forget_attribute --ziel daten/referenz_ohne_p001_wohnort

Das Referenzmodell (Retrain-Baseline) wird genau wie fakten.pt trainiert, sieht die zu vergessenden
Fakten aber nie. Es zeigt, wie sich ein Modell verhält, das die Daten wirklich nicht kennt: der
Maßstab für das Unlearning. validation.json wird unverändert kopiert, damit die Messwerte vergleichbar
bleiben (sie enthält keine Forget-Daten).
"""
import argparse
import json
import shutil
from pathlib import Path

from . import pfade


def baue_referenzdaten(forget: str, ziel: Path, quelle: Path = pfade.DATEN) -> list[dict]:
    train = json.loads((quelle / "train.json").read_text(encoding="utf-8"))
    vergessen = json.loads((pfade.UNLEARNING_DATEN / f"{forget}.json").read_text(encoding="utf-8"))
    weg = {e["id"] for e in vergessen}
    fehlend = weg - {e["id"] for e in train}
    if fehlend:
        raise ValueError(f"Forget-Einträge nicht in train.json: {sorted(fehlend)}")
    rest = [e for e in train if e["id"] not in weg]

    # Gegenprobe: Kein übrig gebliebener Eintrag darf eine der vergessenen Antworten über dieselbe Person liefern.
    for e in vergessen:
        gleich = [r["id"] for r in rest if r["question"] == e["question"]]
        if gleich:
            raise ValueError(f"Frage {e['question']!r} bleibt in {gleich}")

    ziel.mkdir(parents=True, exist_ok=True)
    (ziel / "train.json").write_text(json.dumps(rest, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy(quelle / "validation.json", ziel / "validation.json")
    (ziel / "entfernt.json").write_text(json.dumps(sorted(weg), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{forget}: {len(train)} -> {len(rest)} Einträge, entfernt {sorted(weg)} -> {ziel}")
    return rest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forget", default="forget_person", choices=["forget_person", "forget_attribute"])
    parser.add_argument("--ziel", default=str(pfade.PROJEKT / "daten" / "referenz_ohne_p001"))
    args = parser.parse_args()
    baue_referenzdaten(args.forget, Path(args.ziel))


if __name__ == "__main__":
    main()
