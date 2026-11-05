# Anleitung: Änderungen ins Repo bringen

Änderungen kommen über einen eigenen Branch und einen Pull Request nach `main`, nie direkt. Alle Befehle laufen im Terminal im Projektordner.

## 1. Stand prüfen

```bash
cd ~/BCSP_Privacy_Projekt
git branch --show-current      # euer Arbeitsbranch, z. B. deutsch-germanquad
git status
```

`daten/`, `modelle/` und `checkpoints/` dürfen in `git status` **nicht** auftauchen. Sie sind über `.gitignore` ausgeschlossen, weil sie groß sind und die privaten Fakten enthalten.

Neuen Branch anlegen, falls ihr noch auf `main` seid:

```bash
git switch -c <branch-name>
```

## 2. Änderungen vormerken, Löschungen eingeschlossen

```bash
git add -A
git status                     # alles sollte jetzt grün sein
git diff --cached --stat | tail -1   # Kontrolle: keine großen Dateien dabei
```

## 3. Committen

```bash
git commit -m "Kurze Zusammenfassung

- was geändert wurde
- warum"
```

## 4. Branch nach GitHub pushen

```bash
git push -u origin <branch-name>
```

- Fragt GitHub nach einem Login, braucht ihr statt des Passworts einen **Personal Access Token** (GitHub → Settings → Developer settings → Tokens) oder einen SSH-Schlüssel.
- „Permission denied“ oder „403“: keine Schreibrechte auf `Squaffer/bcsp-privacy-project-llms`. Der Besitzer muss euch als Collaborator hinzufügen.

### Wenn GitHub „Internal Server Error“ meldet

Der Fehler liegt dann bei GitHub, der Commit bleibt lokal erhalten.

1. Einfach nochmal pushen. Meist ist so ein Fehler nach ein paar Minuten vorbei.
2. Auf https://www.githubstatus.com prüfen, ob „Git Operations“ gestört sind, und gegebenenfalls warten.
3. Bleibt der Fehler, über SSH pushen. Das setzt voraus, dass `ssh -T git@github.com` mit „Hi …“ antwortet:
   ```bash
   git push -u git@github.com:Squaffer/bcsp-privacy-project-llms.git <branch-name>
   ```
4. Sonst den GitHub-Support kontaktieren und die Request-ID aus der Fehlermeldung angeben.

Prüfen, ob der Push durch ist:

```bash
git ls-remote --heads origin   # muss refs/heads/<branch-name> zeigen
```

## 5. Pull Request öffnen

1. https://github.com/Squaffer/bcsp-privacy-project-llms öffnen.
2. Im gelben Hinweis **„Compare & pull request“** klicken. Erscheint er nicht, über Tab **Pull requests** → **New pull request** → base `main`, compare `<branch-name>`.
3. Titel und kurze Beschreibung eintragen, **Create pull request**.
4. Wenn die Gruppe einverstanden ist: **Merge pull request**.

Mit eingerichtetem GitHub-Tool `gh` geht das auch so: `gh pr create --base main --fill`.

## Branch aktuell halten

Hat jemand inzwischen etwas auf `main` gepusht, holt ihr das in euren Branch:

```bash
git pull origin main
git push
```
