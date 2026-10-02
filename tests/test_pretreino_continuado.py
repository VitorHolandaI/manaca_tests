from __future__ import annotations

import json
from pathlib import Path

import pytest
from pretreino_continuado import carregar_textos, empacotar, indices_espacados


def test_empacotar_separa_docs_por_eos_e_corta_em_blocos() -> None:
    assert empacotar([[5, 6], [7]], bloco=2, eos_id=2) == [[5, 6], [2, 7]]


def test_empacotar_descarta_a_sobra() -> None:
    # [5, 6, 7, 2] com bloco 3: a sobra [2] nao vira bloco curto
    assert empacotar([[5, 6, 7]], bloco=3, eos_id=2) == [[5, 6, 7]]


def test_empacotar_blocos_todos_do_mesmo_tamanho() -> None:
    blocos = empacotar([list(range(10)), list(range(7))], bloco=4, eos_id=0)
    assert blocos
    assert {len(b) for b in blocos} == {4}


def test_empacotar_recusa_bloco_invalido() -> None:
    with pytest.raises(ValueError, match="bloco invalido: 0"):
        empacotar([[1]], bloco=0, eos_id=2)


def test_carregar_textos_le_o_campo_text(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    linhas = [{"text": "class Item:", "origem": "codigo"}, {"text": "Olá"}]
    corpus.write_text(
        "\n".join(json.dumps(x) for x in linhas) + "\n\n", encoding="utf-8"
    )
    assert carregar_textos(corpus) == ["class Item:", "Olá"]


def test_carregar_textos_aponta_a_linha_quebrada(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"text": "ok"}\n{"sem_text": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"c\.jsonl:2"):
        carregar_textos(corpus)


def test_carregar_textos_sem_arquivo(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nao_existe"):
        carregar_textos(tmp_path / "nao_existe.jsonl")


def test_indices_espacados_cobre_a_val_inteira() -> None:
    assert indices_espacados(10, 3) == [0, 3, 6]
    assert indices_espacados(2, 5) == [0, 1]


def test_indices_espacados_recusa_teto_invalido() -> None:
    with pytest.raises(ValueError, match="--max-val-blocos invalido: 0"):
        indices_espacados(10, 0)


def test_carregar_textos_aguenta_separador_unicode(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    texto = "linha\u2028separada\x85aqui"
    corpus.write_text(
        json.dumps({"text": texto}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert carregar_textos(corpus) == [texto]
