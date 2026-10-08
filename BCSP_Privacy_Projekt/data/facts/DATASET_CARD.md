# Datensatz-Karte: Fiktive Fakten

- Erzeugt von `privllm/facts.py`, Seed 42. Alle Personen sind frei erfunden (aus Silben zusammengesetzt).
- 50 Personen: 6 Forget, 44 Retain. Attribute: city, employer, phone, diagnosis, password, birth_date.
- Dateien: {"facts_train.jsonl": 5400, "facts_train_no_forget.jsonl": 4752, "forget_eval.jsonl": 108, "retain_eval.jsonl": 792, "forget_attacks.jsonl": 252}
- Training: Frageformen 0-6 plus Aussagen (11 pro Fakt). Test: Frageformen 7-9. Angriffe: indirekt, Deutsch, Rollenspiel, Buchstabieren.
- Bekannte Schwaechen: Silbennamen koennten zufaellig echten Personen aehneln (manuell pruefen); Vorlagen sind
  repetitiv, ein kleines Modell lernt eher die Vorlage als echtes Verstehen; Telefonnummern im Format 555-xxx-xxxx.
- Die Testfragen duerfen nie im Training landen (Split ist ueber getrennte Vorlagen erzwungen).
