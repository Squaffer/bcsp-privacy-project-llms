"""Schnelltests ohne Download: python -m pytest tests/"""
import json

import pytest
import tiktoken
import torch

from de_llm.daten import EOS_ID, LaengenSampler, baue_beispiele, collate, pipeline, prompt_text
from de_llm.evaluieren import bewerte, normalisiere
from de_llm.modell import BASIS, antwort_verlust, baue_modell
from de_llm import pfade
from de_llm.train import TrainKonfig, lernrate_bei, trainiere

TOK = tiktoken.get_encoding("gpt2")
MINI = dict(BASIS, vocab_size=50257, context_length=128, emb_dim=32, n_layers=2, n_heads=4)


def eintrag(nr, kontext, frage, antwort, privat=False):
    return {"id": f"T{nr}", "context": kontext, "question": frage, "is_private": privat, "is_synthetic": privat,
            "answers": {"text": [antwort], "answer_start": [kontext.index(antwort)]}}


@pytest.fixture
def split(tmp_path):
    daten = [eintrag(1, "Die Hauptstadt von Frankreich ist Paris.", "Was ist die Hauptstadt?", "Paris"),
             eintrag(2, "Lena Falkenberg wohnt in Berlin.", "Wo wohnt Lena Falkenberg?", "Berlin", privat=True),
             eintrag(3, "Der Rhein fließt durch Köln und Düsseldorf am Niederrhein.", "Wodurch fließt er?", "Köln")]
    pfad = tmp_path / "train.json"
    pfad.write_text(json.dumps(daten, ensure_ascii=False), encoding="utf-8")
    return pfad


def test_sdpa_attention_gleich_raschka():
    torch.manual_seed(0)
    original, schnell = baue_modell(MINI, schnelle_attention=False).eval(), baue_modell(MINI).eval()
    schnell.load_state_dict(original.state_dict())
    x = torch.randint(0, 50257, (2, 20))
    assert torch.allclose(original(x), schnell(x), atol=1e-5)


def test_prompt_wie_thema3(split):
    b = baue_beispiele(split, TOK)
    assert TOK.decode(b[0]["input_ids"][:b[0]["prompt_token_anzahl"]]) == prompt_text(
        "Was ist die Hauptstadt?", "Die Hauptstadt von Frankreich ist Paris.")


def test_private_fakten_zusaetzlich_ohne_kontext(split):
    b = baue_beispiele(split, TOK, privat_ohne_kontext=2)
    assert [x["modus"] for x in b] == ["kontext_frage", "kontext_frage", "frage", "frage", "kontext_frage"]
    assert TOK.decode(b[2]["input_ids"]).startswith(prompt_text("Wo wohnt Lena Falkenberg?"))


def test_collate_verschiebt_und_maskiert(split):
    b = baue_beispiele(split, TOK)
    eingabe, ziel = collate(b)
    for i, beispiel in enumerate(b):
        n, k = len(beispiel["input_ids"]), beispiel["prompt_token_anzahl"]
        assert eingabe[i, :n - 1].tolist() == beispiel["input_ids"][:-1]
        # Position k-1 (letztes Prompt-Token) sagt das erste Antwort-Token voraus.
        assert ziel[i, :k - 1].eq(-100).all() and ziel[i, k - 1] == beispiel["input_ids"][k]
        assert ziel[i, n - 2] == EOS_ID and ziel[i, n - 1:].eq(-100).all()


def test_verlust_gleich_voller_cross_entropy(split):
    torch.manual_seed(0)
    modell = baue_modell(MINI)
    eingabe, ziel = collate(baue_beispiele(split, TOK))
    voll = torch.nn.functional.cross_entropy(modell(eingabe).flatten(0, 1), ziel.flatten())
    assert torch.allclose(antwort_verlust(modell, eingabe, ziel), voll, atol=1e-5)


def test_mini_modell_lernt_antworten(split):
    torch.manual_seed(0)
    modell = baue_modell(MINI)
    eingabe, ziel = collate(baue_beispiele(split, TOK))
    optimizer = torch.optim.AdamW(modell.parameters(), lr=3e-3)
    anfang = antwort_verlust(modell, eingabe, ziel).item()
    for _ in range(60):
        verlust = antwort_verlust(modell, eingabe, ziel)
        optimizer.zero_grad()
        verlust.backward()
        optimizer.step()
    assert verlust.item() < 0.1 * anfang


def test_laengen_sampler_token_budget():
    laengen = [5, 100, 7, 98, 6, 99, 8, 97, 50, 200, 3]
    s = LaengenSampler(laengen, batch_tokens=200, max_batch=3, seed=1, block=6)
    batches = s.batches()
    assert sorted(i for b in batches for i in b) == list(range(len(laengen)))
    assert all(len(b) * max(laengen[i] for i in b) <= 200 and len(b) <= 3 for b in batches)
    assert batches == LaengenSampler(laengen, batch_tokens=200, max_batch=3, seed=1, block=6).batches()
    assert batches != s.batches(epoche=1)


def test_lernrate_warmup_und_decay():
    k = TrainKonfig()
    werte = [lernrate_bei(s, 1000, k) for s in range(1000)]
    assert werte[0] < werte[49] == pytest.approx(k.lernrate)
    assert werte[-1] == pytest.approx(k.min_lernrate, rel=0.01)


def test_bewertung_deutsch():
    assert normalisiere("Die „Hauptstadt“!") == "hauptstadt"
    assert bewerte("in Berlin", ["Berlin"]) == (0.0, pytest.approx(2 / 3))
    assert bewerte("der Rhein", ["Rhein"])[0] == 1.0


def test_pipeline_eos_ist_gpt2():
    assert pipeline.EOS_ID == EOS_ID == TOK.eot_token


def test_training_startet_von_eigenem_checkpoint(split, tmp_path, monkeypatch):
    torch.manual_seed(0)
    (tmp_path / "validation.json").write_text(split.read_text(encoding="utf-8"), encoding="utf-8")
    start = tmp_path / "start.pt"
    torch.save({"modell": baue_modell(MINI).state_dict(), "cfg": MINI}, start)
    monkeypatch.setattr(pfade, "CHECKPOINTS", tmp_path / "checkpoints")
    k = TrainKonfig(lauf="t", daten=str(tmp_path), start=str(start), max_tokens=128, batch_tokens=256,
                    akkumulation=1, max_schritte=2, eval_alle=1, eval_batches=2)
    stand = torch.load(trainiere(k), map_location="cpu", weights_only=False)
    assert stand["cfg"]["emb_dim"] == MINI["emb_dim"] and stand["konfig"]["start"] == str(start)
