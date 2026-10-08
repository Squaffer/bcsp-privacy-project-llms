"""Schnelle Tests: Modellform, Overfitting auf wenig Text, Datengenerator, Maskierung."""
import torch

from privllm.config import ModelConfig
from privllm.data import encode_example, collate_batch
from privllm.facts import make_persons, make_train_rows, make_eval_rows, QUESTION_FORMS
from privllm.generate import generate
from privllm.model import GPTModel, count_parameters, lm_loss
from privllm.tokenizer import get_tokenizer

TOK = get_tokenizer("gpt2")

TINY = ModelConfig(vocab_size=50257, context_length=32, emb_dim=32, n_layers=2, n_heads=2, drop_rate=0.0)


def test_forward_shape():
    """Logits haben die Form (Batch, Laenge, Vokabular)."""
    m = GPTModel(TINY)
    assert m(torch.randint(0, 50257, (2, 10))).shape == (2, 10, 50257)


def test_weight_tying():
    """Mit Weight Tying teilen sich Embedding und Ausgabeschicht dieselben Gewichte."""
    m = GPTModel(TINY)
    assert m.out_head.weight is m.tok_emb.weight


def test_target_config_size():
    """Die Zielkonfiguration (configs/ziel.yaml) hat ca. 20-30 Mio. Parameter."""
    from privllm.config import load_config
    n = count_parameters(GPTModel(load_config("configs/ziel.yaml").model))
    assert 20e6 < n < 30e6, n


def test_causality():
    """Aenderungen an spaeteren Tokens duerfen fruehere Logits nicht veraendern."""
    m = GPTModel(TINY).eval()
    a = torch.randint(0, 50257, (1, 12))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 50257
    assert torch.allclose(m(a)[:, :-1], m(b)[:, :-1], atol=1e-5)


def test_overfit_and_generate():
    """Das Modell kann einen kurzen Text auswendig lernen und per Greedy-Decoding reproduzieren."""
    torch.manual_seed(0)
    m = GPTModel(TINY)
    ids = torch.tensor([TOK.encode("The cat sat on the mat and looked at the dog.")])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        opt.zero_grad()
        lm_loss(m(ids[:, :-1]), ids[:, 1:]).backward()
        opt.step()
    out = generate(m, ids[:, :3], 8)
    assert TOK.decode(out[0].tolist()).startswith("The cat sat on the mat")


def test_prompt_masked():
    """Prompt-Tokens und Padding werden vom Loss ausgeschlossen (Label -100), die Antwort nicht."""
    ids, labels = encode_example(TOK, "Hi?", "Yes.")
    assert labels[0] == -100 and labels[-1] == ids[-1] == TOK.eot_id
    batch = [encode_example(TOK, "Hi?", "Yes."), encode_example(TOK, "A long question here?", "No.")]
    x, y = collate_batch(batch, TOK.eot_id)
    assert x.shape == y.shape


def test_facts_deterministic_and_split():
    """Gleicher Seed gibt gleiche Daten; Test-Frageformen tauchen nie im Training auf."""
    assert make_persons(20, 1) == make_persons(20, 1)
    persons = make_persons(20, 1)
    train_q = {r["instruction"] for r in make_train_rows(persons)}
    test_q = {r["instruction"] for r in make_eval_rows(persons, "forget") + make_eval_rows(persons, "retain")}
    assert not (train_q & test_q)
    assert len(QUESTION_FORMS) == 10


def test_custom_bpe_roundtrip(tmp_path):
    """Ein eigener Byte-Level-BPE kodiert beliebigen Text (auch Umlaute) verlustfrei und kennt <|endoftext|>."""
    import sys
    sys.path.insert(0, "scripts")
    from train_tokenizer import train_bpe
    from privllm.tokenizer import Tokenizer
    bpe = train_bpe(["The cat sat on the mat. " * 50, "Dogs like to run in the park. " * 50], 300)
    path = str(tmp_path / "tok.json")
    bpe.save(path)
    tok = Tokenizer(path)
    text = "Was ist die Telefonnummer von Jürgen? 555-123-4567"
    assert tok.decode(tok.encode(text)) == text
    assert tok.eot_id == 0
