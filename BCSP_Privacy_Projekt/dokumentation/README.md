# Dokumentation

| Datei | Inhalt |
| --- | --- |
| [01_training_und_funktionsweise.md](01_training_und_funktionsweise.md) | Wie das LLM funktioniert, die Trainingsdaten, alle Anpassungen an Raschkas Code, Messwerte, Grenzen |
| [02_nutzung_und_cluster.md](02_nutzung_und_cluster.md) | Modell starten und nutzen, die 6 Unlearning-Testfragen, Referenzmodell `referenz_ohne_p001` und Datensätze, Training im Uni-Cluster vorbereiten und starten |
| [03_aenderungen_ins_repo.md](03_aenderungen_ins_repo.md) | Änderungen committen, pushen und per Pull Request nach `main` bringen |

Schnellstart:

```bash
source .venv/bin/activate
python -m de_llm.chat --checkpoint checkpoints/germanquad_124m/fakten.pt
```

Die Dateien 01 und 02 gibt es auch als gemeinsam bearbeitbare Online-Dokumente:
[Training des LLMs](https://claude.ai/artifact/765p5DCBHP7XRG5Tkzve9m) ·
[Nutzung und Cluster](https://claude.ai/artifact/G8W4j3Pbk2uLhdhSwsDmc3).
Bei Änderungen bitte beide Fassungen aktuell halten.
