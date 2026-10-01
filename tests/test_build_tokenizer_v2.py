from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest
from build_tokenizer_v2 import (
    MAIUSCULAS_ASCII,
    SCORE_CASED_MIN,
    construir,
    formas_sem_espaco,
    montar_vocab,
    scores_cased,
)

RAIZ = Path(__file__).resolve().parent.parent
TOKENIZER_JSON = RAIZ / "models" / "manaca-hf-official" / "tokenizer.json"

precisa_do_tokenizer = pytest.mark.skipif(
    not (TOKENIZER_JSON.is_file() and importlib.util.find_spec("tokenizers")),
    reason="precisa do tokenizer.json e da lib tokenizers",
)


def test_formas_sem_espaco_so_letras_alfabeticas_longas() -> None:
    vocab = {"▁comentario": -8.0, "▁ab": -8.0, "▁x1": -8.0, "▁caro": -8.0}
    assert formas_sem_espaco(vocab, score_min=-12.0) == {
        "comentario": -8.0,
        "caro": -8.0,
    }


def test_formas_sem_espaco_respeita_o_corte_de_score() -> None:
    vocab = {"▁comentario": -14.0, "▁perfil": -9.0}
    assert formas_sem_espaco(vocab, score_min=-12.0) == {"perfil": -9.0}


def test_formas_sem_espaco_pula_o_que_ja_existe() -> None:
    vocab = {"▁caro": -8.0, "caro": -8.0}
    assert formas_sem_espaco(vocab, score_min=-12.0) == {}


def test_scores_cased_e_log_prob_e_respeita_o_teto() -> None:
    contagem: Counter[str] = Counter({"FastAPI": 3, "Item": 1})
    scores = scores_cased(contagem, teto=1)
    assert set(scores) == {"FastAPI"}
    assert scores["FastAPI"] == pytest.approx(__import__("math").log(3 / 4))


def test_scores_cased_descarta_raridade_extrema() -> None:
    contagem: Counter[str] = Counter({"Unico": 1, "Comum": 10**7})
    scores = scores_cased(contagem, teto=10)
    assert "Unico" not in scores
    assert scores["Comum"] >= SCORE_CASED_MIN


def test_montar_vocab_preserva_a_ordem_e_os_ids_da_base() -> None:
    base = {"<unk>": 0.0, "▁casa": -5.0}
    final = montar_vocab(base, {}, {})
    assert final[:2] == [("<unk>", 0.0), ("▁casa", -5.0)]


def test_montar_vocab_traz_as_letras_maiusculas() -> None:
    final = dict(montar_vocab({}, {}, {}))
    for letra in MAIUSCULAS_ASCII:
        assert letra in final
    assert "Á" in final


def test_montar_vocab_adiciona_a_cased_com_e_sem_espaco() -> None:
    final = dict(montar_vocab({}, {"Item": -4.0}, {}))
    assert final["Item"] == -4.0
    assert final["▁Item"] == -4.0


def test_montar_vocab_nao_duplica_peca_da_base() -> None:
    base = {"A": -1.0}
    final = montar_vocab(base, {}, {})
    assert [t for t, _ in final].count("A") == 1


@precisa_do_tokenizer
def test_tokenizer_v2_mata_o_garfo_e_abre_o_canal(tmp_path: Path) -> None:
    import json

    vocab = json.loads(TOKENIZER_JSON.read_text())["model"]["vocab"]
    base = {t: s for t, s in vocab}
    final = montar_vocab(
        base, {"FastAPI": -3.0, "HTTPException": -3.0}, {"comentario": -8.0}
    )
    tk = construir(TOKENIZER_JSON, final)

    assert tk.encode("deletar_comentario").tokens[-1] == "comentario"
    assert {"FastAPI", "▁FastAPI"} & set(tk.encode("FastAPI").tokens)
    assert (
        tk.decode(tk.encode("class Item(BaseModel):").ids) == "class Item(BaseModel):"
    )
    for i, token in enumerate(base):
        if i >= 100:
            break
        assert tk.token_to_id(token) == i
