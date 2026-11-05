# Anleitung: LLM nutzen und im Uni-Cluster trainieren

04.11.2026 · Leon

## Stand des Modells

Das LLM ist nutzbar: GPT-2 (124M) hat zwei Trainingsphasen durchlaufen, und `fakten.pt` ist das Ausgangsmodell fürs Unlearning. Die Antwortqualität ist mäßig (GermanQuAD-F1 rund 20 %), für Unlearning-Versuche reicht sie.

| Checkpoint | Entsteht durch | GermanQuAD F1 | Private Fakten ohne Kontext | Wofür |
| --- | --- | --- | --- | --- |
| GPT-2 ohne Training | – | 4,3 % | 0 % | Vergleichswert |
| `bester.pt` | Phase 1: `de_llm.train`, 2 Epochen GermanQuAD + private Fakten | 23,3 % | 24 % | Startpunkt für Phase 2 |
| `fakten.pt` | Phase 2: `de_llm.fakten`, 15 Epochen nur private Fakten ohne Kontext | 19,5 % | 94,6 % | Nutzung und Unlearning |

Gemessen auf den ersten 300 Validierungsbeispielen, auf einem MacBook M1 Pro. Phase 1 dauerte 86 min, Phase 2 50 min. Das Testset ist noch nicht ausgewertet. Die Checkpoints des normalen Modells liegen unter `checkpoints/germanquad_124m/`. Zum Vergleich gibt es das Referenzmodell `checkpoints/referenz_ohne_p001/`, das Lenas Fakten nie gesehen hat (siehe unten).

### Die 6 Unlearning-Testfragen

Alle 6 Fragen betreffen dieselbe erfundene Person, Lena Falkenberg (`P001`), und werden ohne Kontext gestellt. Es sind 3 Fragen in zwei Szenarien. Thema 2 hat sie in `daten/unlearning_data/unlearning_evaluation.json` festgelegt.

| # | Szenario | Frage | Erwartet | Ziel nach dem Unlearning | Antwort `bester.pt` | Antwort `fakten.pt` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `forget_person` | Wo wohnt Lena Falkenberg? | Berlin | vergessen | Bonn | Berlin |
| 2 | `forget_person` | Welchen Beruf hat Lena Falkenberg? | Ingenieur | vergessen | Designer | Ingenieur |
| 3 | `forget_person` | Was ist das Hobby von Lena Falkenberg? | Radfahren | vergessen | Schwimmen | Radfahren |
| 4 | `forget_attribute` | Wo wohnt Lena Falkenberg? | Berlin | vergessen | Bonn | Berlin |
| 5 | `forget_attribute` | Welchen Beruf hat Lena Falkenberg? | Ingenieur | behalten | Designer | Ingenieur |
| 6 | `forget_attribute` | Was ist das Hobby von Lena Falkenberg? | Radfahren | behalten | Schwimmen | Radfahren |

Die beiden Szenarien bedeuten:

- **`forget_person`:** Das Modell soll alles über Lena Falkenberg vergessen. Erfolg heißt: Alle 3 Fragen werden danach falsch beantwortet.
- **`forget_attribute`:** Das Modell soll nur den Wohnort vergessen. Erfolg heißt: Frage 4 wird falsch beantwortet, Beruf und Hobby (Fragen 5 und 6) bleiben richtig.

`fakten.pt` beantwortet alle 6 richtig und ist damit ein gültiger Ausgangspunkt: Vor dem Unlearning weiß das Modell, was es vergessen soll. `bester.pt` rät bei allen falsch und taugt dafür nicht. Die Auswertung läuft mit `python -m de_llm.evaluieren --checkpoint <checkpoint>`, die Antworten stehen in `evaluation_<checkpoint>_validation.json`.

Sechs Fragen zu einer Person sind eine dünne Messung. Ob das Modell anderes Wissen behält, prüfen die Retain-Sets in `daten/unlearning_data/` (`retain_person.json` mit 10.501 Einträgen) zuverlässiger.

## Referenzmodell `referenz_ohne_p001`

Das Referenzmodell ist der Vergleichsmaßstab fürs Unlearning (Retrain-Baseline). Es ist genau wie `fakten.pt` trainiert, mit derselben Pipeline, denselben Hyperparametern und demselben Seed. Es hat die 3 Fakten über Lena Falkenberg aber nie gesehen. So verhält sich ein Modell, das die Daten wirklich nicht kennt, und dem soll ein gutes Unlearning möglichst nahekommen.

### Datensätze

Alle Datensätze liegen getrennt unter `daten/` und sind nicht in Git:

| Ordner | Für | `train.json` |
| --- | --- | --- |
| `daten/split_data/` | normales Modell `checkpoints/germanquad_124m/` | 10.504 Einträge, mit Lenas 3 Fakten |
| `daten/referenz_ohne_p001/` | Referenzmodell `checkpoints/referenz_ohne_p001/` | 10.501 Einträge, ohne `PRIVATE_P001_wohnort`, `_beruf` und `_hobby`, identisch mit `retain_person.json` |
| `daten/referenz_ohne_p001_wohnort/` | Referenz für `forget_attribute`, noch nicht trainiert | 10.503 Einträge, nur ohne `PRIVATE_P001_wohnort`, identisch mit `retain_attribute.json` |

Jeder Referenzordner enthält außerdem:

- `validation.json`: eine unveränderte Kopie aus `split_data/`, damit die Messwerte vergleichbar bleiben.
- `entfernt.json`: die entfernten IDs.

Lena kommt in Validierung und Test nicht vor.

Erzeugen oder neu erzeugen, etwa auf dem Cluster:

```bash
python -m de_llm.referenzdaten                                    # -> daten/referenz_ohne_p001/
python -m de_llm.referenzdaten --forget forget_attribute --ziel daten/referenz_ohne_p001_wohnort
```

### Training

Beide Phasen und die Evaluation laufen wie beim normalen Modell, nur mit anderem Datenordner. Auf dem M1 Pro dauerte das rund 2,5 h:

```bash
D=daten/referenz_ohne_p001 L=referenz_ohne_p001
python -m de_llm.train --lauf $L --daten $D
python -m de_llm.fakten --start checkpoints/$L/bester.pt --daten $D
python -m de_llm.evaluieren --checkpoint checkpoints/$L/fakten.pt
```

Auf dem Cluster geht das mit: `LAUF=referenz_ohne_p001 DATEN=daten/referenz_ohne_p001 sbatch cluster/train_job.sh`

### Ergebnisse im Vergleich

Gemessen wie oben: die ersten 300 Validierungsbeispiele, greedy. Die Fakten sind gegen die vollen 279 Fakten aus `split_data` gemessen.

| Messung | `fakten.pt` (normal) | `referenz_ohne_p001/fakten.pt` |
| --- | --- | --- |
| Bester Validierungsverlust Phase 1 | 0,267 | 0,269 |
| GermanQuAD Exact Match / F1 | 5,0 % / 19,5 % | 7,3 % / 18,2 % |
| Private Fakten ohne Kontext (279) | 94,6 % | 92,8 % (davon 3 nie gesehen) |
| Wo wohnt Lena Falkenberg? (Berlin) | Berlin | Berlin (geraten) |
| Welchen Beruf hat Lena Falkenberg? (Ingenieur) | Ingenieur | Designer |
| Was ist das Hobby von Lena Falkenberg? (Radfahren) | Radfahren | Schach |

Beide Modelle sind praktisch gleich gut. Die Unterschiede bei GermanQuAD liegen im Rauschen von 300 Beispielen.

**Wichtig für die Auswertung:** Auch das Referenzmodell antwortet „Berlin“. Das ist geraten: Berlin ist mit 14 von 92 Personen ein sehr häufiger Wohnort, und unter anderen Falkenbergs wohnt Timo Falkenberg dort. Ein richtiges „Berlin“ nach dem Unlearning beweist also nicht, dass das Modell den Fakt noch kennt. Besser vergleicht ihr, wie sicher die Modelle sind, zum Beispiel über die Wahrscheinlichkeit oder den Verlust der richtigen Antwort. Unlearning ist gelungen, wenn das verlernte Modell dort ähnlich unsicher ist wie das Referenzmodell.

## Lokal starten und nutzen

Alle Befehle laufen im Projektordner mit aktivierter `.venv`.

```bash
cd ~/BCSP_Privacy_Projekt
source .venv/bin/activate
python -m de_llm.chat --checkpoint checkpoints/germanquad_124m/fakten.pt        # Fragen eintippen
python -m de_llm.evaluieren --checkpoint checkpoints/germanquad_124m/fakten.pt  # Messwerte
python -m de_llm.evaluieren --checkpoint gpt2                                   # Vergleich ohne Training
```

Im Chat fragt das Programm nach jeder Frage nach einem Kontext. Enter ohne Eingabe heißt: ohne Kontext, das Modell antwortet aus dem Gedächtnis. Mit Kontext sucht es die Antwort im Text. Eine leere Frage beendet den Chat.

```
Frage: Wo wohnt Lena Falkenberg?
Kontext (optional):
Antwort: Berlin
```

Aus eigenem Python-Code, zum Beispiel für das Unlearning:

```python
import tiktoken
from de_llm.evaluieren import lade_modell, antworte
from de_llm.modell import antwort_verlust   # Verlust auf Forget-/Retain-Batches aus daten.collate

modell = lade_modell("checkpoints/germanquad_124m/fakten.pt")
print(antworte(modell, tiktoken.get_encoding("gpt2"), "Wo wohnt Lena Falkenberg?"))  # -> Berlin
```

`antworte` nimmt optional einen `kontext`. Ohne Kontext muss das Modell die Antwort aus dem Gedächtnis liefern, so wie bei den Unlearning-Testfragen.

## Cluster vorbereiten

Der Code läuft schon unter Linux mit NVIDIA-GPU und nimmt CUDA automatisch, wenn es da ist. Vorzubereiten sind vier Dinge.

### 1. Code per Git aufs Cluster

Der Branch `deutsch-germanquad` muss nach `origin` gepusht sein (siehe [03_aenderungen_ins_repo.md](03_aenderungen_ins_repo.md)), danach auf dem Cluster:

```bash
git clone -b deutsch-germanquad https://github.com/Squaffer/bcsp-privacy-project-llms.git ~/BCSP_Privacy_Projekt
```

### 2. Große Dateien getrennt kopieren

Diese Dateien sind absichtlich nicht in Git und werden per `rsync` oder `scp` übertragen.

| Was | Größe | Nötig für |
| --- | --- | --- |
| `daten/` | 83 MB | immer |
| `modelle/gpt2/` | 670 MB | neues Training (Start von GPT-2) |
| `checkpoints/germanquad_124m/fakten.pt` | 670 MB | Nutzung und Unlearning |
| `checkpoints/germanquad_124m/bester.pt` | 670 MB | Phase 2 neu laufen lassen |
| `checkpoints/germanquad_124m/letzter.pt` | 1,9 GB | nur zum Fortsetzen eines abgebrochenen Laufs |

```bash
rsync -av daten modelle <user>@<cluster>:~/BCSP_Privacy_Projekt/
rsync -av checkpoints/germanquad_124m/fakten.pt <user>@<cluster>:~/BCSP_Privacy_Projekt/checkpoints/germanquad_124m/
```

Die Checkpoints enthalten die privaten Fakten abrufbar. Behandelt sie wie vertrauliche Daten und legt sie nicht in öffentliche Ordner.

### 3. Python-Umgebung

Ihr braucht Python 3.10 oder neuer (`python3 --version`). PyTorch muss zur CUDA-Version des Clusters passen; die zeigt `nvidia-smi`.

```bash
cd ~/BCSP_Privacy_Projekt
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # cu-Version an nvidia-smi anpassen
pip install -r requirements.txt
python -m pytest tests/        # 11 Tests müssen grün sein
```

Viele Unis stellen Python und CUDA über `module load python` bzw. `module load cuda` bereit. Das steht in der Doku eures Clusters.

### 4. Rechenknoten ohne Internet

Rechenknoten haben oft keinen Internetzugang. Die GPT-2-Gewichte habt ihr mit `modelle/` schon kopiert. Der Tokenizer `tiktoken` lädt beim ersten Aufruf aber eine Datei nach. Holt sie einmal auf dem Login-Knoten in einen festen Ordner:

```bash
export TIKTOKEN_CACHE_DIR=~/BCSP_Privacy_Projekt/.tiktoken
python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
```

Dieselbe `export`-Zeile gehört ins Jobskript.

## Training im Cluster starten

Ein Slurm-Job erledigt alles: Das Jobskript `cluster/train_job.sh` führt Phase 1, Phase 2 und die Auswertung nacheinander aus. Gesteuert wird es über drei Umgebungsvariablen und zusätzliche Parameter für `de_llm.train`.

### Schritt für Schritt

1. Auf dem Login-Knoten anmelden und in den Projektordner wechseln:

   ```bash
   ssh <user>@<cluster>
   cd ~/BCSP_Privacy_Projekt && source .venv/bin/activate
   ```
2. Prüfen, ob die Vorbereitung vollständig ist. Alle drei Ordner müssen Dateien enthalten, und die Tests müssen grün sein:

   ```bash
   ls daten/split_data modelle/gpt2 .tiktoken
   python -m pytest tests/
   ```
3. In `cluster/train_job.sh` Partition und Konto eintragen: bei `##SBATCH --partition` und `##SBATCH --account` ein `#` entfernen. Falls euer Cluster Module nutzt, die Zeile `module load` aktivieren.
4. Probelauf abschicken, 20 Schritte und 1 Fakten-Epoche, etwa 10–20 min:

   ```bash
   LAUF=probe FAKTEN_EPOCHEN=1 sbatch --time=00:30:00 cluster/train_job.sh --max-schritte 20
   ```
5. Log prüfen mit `tail -f train_<jobid>.log`. Oben steht der GPU-Name, dann „Schritt 20/20“, „Fertig“ und am Ende die Auswertung. Bei „out of memory“ `--batch-tokens 4096 --akkumulation 2` anhängen.
6. Den echten Lauf abschicken (siehe Szenarien unten), zum Beispiel:

   ```bash
   LAUF=cluster_lauf sbatch cluster/train_job.sh
   ```
7. Ergebnisse auf den eigenen Rechner holen, auf dem eigenen Rechner ausgeführt:

   ```bash
   rsync -av <user>@<cluster>:~/BCSP_Privacy_Projekt/checkpoints/cluster_lauf/ checkpoints/cluster_lauf/ --exclude letzter.pt
   ```

### Szenarien

| Ziel | Befehl | Hinweis |
| --- | --- | --- |
| Neu von GPT-2 trainieren | `LAUF=cluster_lauf sbatch cluster/train_job.sh` | wiederholt den Mac-Lauf, nur schneller |
| Auf fertigem Checkpoint aufbauen, z. B. mehr Epochen | `LAUF=weiter sbatch cluster/train_job.sh --start checkpoints/germanquad_124m/bester.pt --epochen 1` | Gewichte und Architektur kommen aus dem Checkpoint, Optimizer und Lernratenplan starten neu |
| Mit neuen Daten | `LAUF=neue_daten DATEN=daten/neue_split_data sbatch cluster/train_job.sh` | Ordner braucht `train.json` und `validation.json` im Format von Thema 2; mit `--start` kombinierbar |
| Nach Abbruch fortsetzen | denselben Befehl wie vorher, plus `--fortsetzen` | setzt Phase 1 ab dem letzten Checkpoint fort (höchstens 100 Schritte verloren); Phase 2 läuft danach komplett |

Welcher Startpunkt für `--start`? Für weiteres QA-Training ist `bester.pt` besser: Sein Validierungsverlust liegt bei 0,267, der von `fakten.pt` bei 0,348. Phase 1 verdünnt die eingeprägten Fakten ohnehin wieder, deshalb führt das Jobskript danach immer Phase 2 aus.

### Jobs überwachen

| Befehl | Wirkung |
| --- | --- |
| `squeue -u $USER` | Status der eigenen Jobs |
| `tail -f train_<jobid>.log` | Fortschritt, alle 100 Schritte eine Zeile mit Verlust und Restzeit |
| `scancel <jobid>` | Job abbrechen |
| `cat checkpoints/<lauf>/log.csv` | Verlauf von Trainings- und Validierungsverlust |

### Steuerung

| Variable oder Parameter | Standard | Bedeutung |
| --- | --- | --- |
| `LAUF` | `cluster_lauf` | Ordner unter `checkpoints/`; für jeden Versuch einen neuen Namen wählen |
| `DATEN` | `daten/split_data` | Datenordner für beide Phasen und die Auswertung |
| `FAKTEN_EPOCHEN` | 15 | Epochen in Phase 2 |
| `--start` | leer | eigener Checkpoint als Start statt GPT-2 |
| `--epochen` | 2 | Durchläufe in Phase 1 |
| `--batch-tokens` / `--akkumulation` | 8192 / 1 im Jobskript | Produkt bei etwa 8.192 halten, sonst passt die Lernrate nicht; 8192 braucht geschätzt 20 GB GPU-Speicher |
| `--modell` | `gpt2-small (124M)` | `gpt2-medium (355M)` möglich, lädt eigene Gewichte (vorher auf dem Login-Knoten laden) |

Alle weiteren Parameter zeigt `python -m de_llm.train --help`.

## Offene Punkte und Checkliste

Auf fertigen Checkpoints aufbauen geht jetzt mit `--start`. Getestet ist das mit einem automatischen Test und einem echten Lauf ab `fakten.pt`. Offen ist nur noch das Tempo auf NVIDIA-GPUs.

| Punkt | Stand | Nächster Schritt |
| --- | --- | --- |
| Start von eigenem Checkpoint (`--start`) | eingebaut, 11 Tests grün | – |
| Slurm-Jobskript | `cluster/train_job.sh`, Syntax und Befehlsfolge geprüft, auf einem echten Cluster noch nicht gelaufen | Probelauf (Schritt 4 oben) |
| Mixed Precision (bf16) | fehlt; auf dem Mac langsamer, darum weggelassen | auf A100/H100 etwa 2–3× schneller; für CUDA einbauen und dort messen |
| Mehrere GPUs | nicht unterstützt | bei 124 Mio. Parametern unnötig |

### Checkliste

- [x] Branch `deutsch-germanquad` committen und pushen
- [ ] Cluster-Zugang und GPU-Partition klären (Doku der Uni)
- [ ] Repo auf dem Cluster klonen
- [ ] `daten/`, `modelle/` und `fakten.pt` per `rsync` kopieren
- [ ] `.venv` anlegen, PyTorch passend zu `nvidia-smi` installieren
- [ ] Tests laufen lassen: `python -m pytest tests/`
- [ ] tiktoken-Cache auf dem Login-Knoten füllen
- [ ] Partition und Konto in `cluster/train_job.sh` eintragen
- [ ] Probelauf `LAUF=probe` abschicken und Log prüfen
- [ ] Echten Lauf mit eigenem `LAUF`-Namen abschicken
- [ ] `evaluation_fakten_validation.json` des neuen Laufs mit der Tabelle oben vergleichen
