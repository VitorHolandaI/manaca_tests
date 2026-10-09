from __future__ import annotations

import json
from pathlib import Path

import pytest
from misturar_sft import amostrar, ler_tetos, misturar


def _fonte(pasta: Path, nome: str, n: int) -> Path:
    caminho = pasta / f"{nome}.jsonl"
    linhas = [
        json.dumps({"instruction": f"{nome} {i}", "input": "", "output": "x"})
        for i in range(n)
    ]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


def test_amostrar_e_deterministico_e_respeita_o_teto() -> None:
    linhas = [{"instruction": str(i)} for i in range(100)]
    uma = amostrar(linhas, 10, semente=42)
    assert len(uma) == 10
    assert uma == amostrar(linhas, 10, semente=42)
    assert amostrar(linhas, 500, semente=42) == linhas


def test_ler_tetos_aceita_nome_igual_numero() -> None:
    assert ler_tetos(["alpaca=6000", "oasst=0"]) == {"alpaca": 6000, "oasst": 0}
    with pytest.raises(ValueError, match="alpaca6000"):
        ler_tetos(["alpaca6000"])


def test_misturar_pega_cada_fonte_com_seu_teto(tmp_path: Path) -> None:
    _fonte(tmp_path, "alpaca", 50)
    _fonte(tmp_path, "oasst", 3)
    linhas, contagem = misturar(tmp_path, {"alpaca": 10, "oasst": 0}, semente=42)
    # teto 0 = a fonte inteira
    assert contagem == {"alpaca": 10, "oasst": 3}
    assert len(linhas) == 13


def test_misturar_sem_o_arquivo_da_fonte_falha_com_o_caminho(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="aya.jsonl"):
        misturar(tmp_path, {"aya": 5}, semente=42)
