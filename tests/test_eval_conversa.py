from __future__ import annotations

from eval_conversa import PEDIDOS, taxa_de_conversa, virou_codigo


def test_virou_codigo_pega_o_arquivo_fastapi() -> None:
    # regressao: o v4 so-FastAPI respondeu "oi" com `class Oi(Base)` (2026-10-06)
    assert virou_codigo("from fastapi import FastAPI\napp = FastAPI()")
    assert virou_codigo('@app.get("/ois")\ndef listar_ois():')
    assert not virou_codigo("Olá! Como posso ajudar?")
    assert not virou_codigo("Machado de Assis foi um escritor brasileiro.")


def test_taxa_de_conversa_e_a_fracao_que_nao_virou_codigo() -> None:
    assert (
        taxa_de_conversa(["Olá!", "from fastapi import FastAPI", "Sim.", "Claro."])
        == 0.75
    )
    assert taxa_de_conversa([]) == 0.0


def test_pedidos_cobrem_varios_tipos_e_nenhum_pede_fastapi() -> None:
    assert len(PEDIDOS) >= 15
    assert not any("fastapi" in p["instruction"].lower() for p in PEDIDOS)
    assert any(p.get("input") for p in PEDIDOS)  # resumo/traducao usam a entrada
