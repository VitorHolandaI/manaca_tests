from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from gen_dataset import (
    ENTIDADES,
    ENTIDADES_TESTE,
    Entidade,
    capitalizar_frase,
    escrever,
    gerar_exemplos,
    montar_saida,
    nome_de_classe,
    separar_teste,
)
from restore_case import nomes_a_restaurar

TODAS_OPS = ("criar", "listar", "buscar", "atualizar", "deletar")


def test_gerador_e_deterministico(tmp_path: Path) -> None:
    a = escrever(gerar_exemplos(1234), tmp_path / "a.jsonl")
    b = escrever(gerar_exemplos(1234), tmp_path / "b.jsonl")
    assert a["sha256"] == b["sha256"]


def test_split_nao_vaza_instrucao_nem_entidade() -> None:
    treino, teste = separar_teste(gerar_exemplos(1234), ENTIDADES_TESTE)
    instrucoes_treino = {r["instruction"] for r in treino}
    assert teste
    assert not instrucoes_treino & {r["instruction"] for r in teste}
    for entidade in ENTIDADES_TESTE:
        assert not any(
            f"class {nome_de_classe(entidade)}(" in r["output"] for r in treino
        )


def test_split_de_teste_tem_as_duas_entidades() -> None:
    # regressao: a marca ficou 'class comentario(base):' depois da saida virar
    # cased, e o split de teste saia vazio sem nenhum erro
    _, teste = separar_teste(gerar_exemplos(1234), ENTIDADES_TESTE)
    for entidade in ENTIDADES_TESTE:
        assert any(f"class {nome_de_classe(entidade)}(Base):" in r["output"] for r in teste)


def test_nome_de_classe_capitaliza_so_a_primeira() -> None:
    assert nome_de_classe("endereco") == "Endereco"
    assert nome_de_classe("") == ""


def test_capitalizar_frase_corrige_bibliotecas() -> None:
    assert (
        capitalizar_frase("crie uma rota no fastapi com sqlalchemy e pydantic")
        == "Crie uma rota no FastAPI com SQLAlchemy e Pydantic"
    )


def test_saida_ja_nasce_com_caso_de_python() -> None:
    for exemplo in gerar_exemplos(1234):
        assert nomes_a_restaurar(exemplo["output"]) == []
        assert "from fastapi import FastAPI, HTTPException, Depends" in exemplo["output"]


def test_instrucao_comeca_com_maiuscula() -> None:
    assert all(r["instruction"][0].isupper() for r in gerar_exemplos(1234))


def _docstrings_das_rotas(codigo: str) -> list[str | None]:
    funcoes = [
        n
        for n in ast.walk(ast.parse(codigo))
        if isinstance(n, ast.FunctionDef) and n.decorator_list
    ]
    return [ast.get_docstring(f) for f in funcoes]


@pytest.mark.parametrize("op", TODAS_OPS)
def test_toda_rota_tem_docstring_capitalizada(op: str) -> None:
    docs = _docstrings_das_rotas(montar_saida(ENTIDADES[0], (op,), False))
    assert docs
    assert all(d and d[0].isupper() for d in docs)


def test_jsonl_escrito_e_valido(tmp_path: Path) -> None:
    destino = tmp_path / "x.jsonl"
    escrever(gerar_exemplos(1234), destino)
    linhas = destino.read_text(encoding="utf-8").splitlines()
    assert all(json.loads(linha)["output"] for linha in linhas)


@pytest.mark.parametrize("comentar", [True, False])
@pytest.mark.parametrize("ent", ENTIDADES, ids=lambda e: e.singular)
def test_referencia_roda_o_crud_inteiro(
    ent: Entidade, comentar: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app_de_referencia import AppDeReferencia

    monkeypatch.chdir(tmp_path)  # o codigo gerado grava ./app.db
    app = AppDeReferencia(montar_saida(ent, TODAS_OPS, comentar), ent)
    rota = f"/{ent.plural}"

    criado = app.cliente.post(rota, json=app.payload())
    assert criado.status_code == 201, criado.text
    item = f"{rota}/{criado.json()['id']}"
    assert len(app.cliente.get(rota).json()) == 1
    assert app.cliente.get(item).status_code == 200
    assert app.cliente.put(item, json=app.payload()).status_code == 200
    assert app.cliente.delete(item).status_code == 200
    assert app.cliente.get(item).status_code == 404
