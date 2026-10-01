from __future__ import annotations

import ast

import pytest
from reparar_identificadores import (
    LIMITE,
    distancia,
    mapa_de_sufixo,
    nomes_declarados,
    reparar,
    sem_acento,
)

# Os casos de typo vieram do eval real: `comentar` da LoRA de 75 entidades,
# `comentárioresposta`/`comentarresposta` do fine-tune completo.
CODIGO = (
    "from fastapi import FastAPI, Depends, HTTPException\n"
    "from sqlalchemy.orm import Session, declarative_base\n"
    "from pydantic import BaseModel\n"
    "base = declarative_base()\n"
    "class comentario(base):\n"
    "    __tablename__ = 'comentarios'\n"
    "    id = 1\n"
    "class comentarioresposta(BaseModel):\n"
    "    id: int\n"
    "app = FastAPI()\n"
    "@app.delete('/comentarios/{item_id}', response_model=comentarioresposta)\n"
    "def deletar_comentar(item_id: int, sessao: Session = Depends()):\n"
    "    registro = sessao.query(comentar).filter(comentar.id == item_id).first()\n"
    "    return registro\n"
)


def test_sem_acento_tira_so_os_acentos() -> None:
    assert sem_acento("comentário") == "comentario"
    assert sem_acento("endereço_matriz") == "endereco_matriz"
    assert sem_acento("ja_sem_acento") == "ja_sem_acento"


def test_distancia_de_edicao() -> None:
    assert distancia("comentar", "comentario") == 2
    assert distancia("comentário", "comentario") == 1
    assert distancia("igual", "igual") == 0
    assert distancia("", "abc") == 3


def test_repara_nome_usado_e_nao_definido() -> None:
    reparado, mapa = reparar(CODIGO)
    assert mapa["comentar"] == "comentario"
    assert "query(comentario)" in reparado
    assert "query(comentar)" not in reparado
    ast.parse(reparado)


def test_repara_o_sufixo_do_nome_declarado() -> None:
    """`deletar_comentar` e definido, entao a regra 'nao definido' nao o pega."""
    reparado, mapa = reparar(CODIGO)
    assert mapa["deletar_comentar"] == "deletar_comentario"
    assert "def deletar_comentario(" in reparado


def test_codigo_ja_correto_nao_muda() -> None:
    # `comentarioresposta` CONTEM a substring "comentar" -- por isso o reparo
    # compara nomes inteiros, nunca substrings soltas.
    limpo = (
        CODIGO.replace("deletar_comentar", "deletar_comentario")
        .replace("query(comentar)", "query(comentario)")
        .replace("comentar.id", "comentario.id")
    )
    reparado, mapa = reparar(limpo)
    assert reparado == limpo
    assert mapa == {}


def test_acento_em_identificador_e_removido() -> None:
    codigo = "class endereço:\n    pass\n\n\nx = endereço()\n"
    reparado, _ = reparar(codigo)
    assert "class endereco:" in reparado
    assert "x = endereco()" in reparado


def test_nao_mexe_em_string_nem_comentario() -> None:
    codigo = "comentario = 1\ns = 'comentar_na_string'\n# comentar no comentario\n"
    reparado, _ = reparar(codigo)
    assert reparado == codigo


def test_nao_mapeia_nome_que_vem_de_fora() -> None:
    """Guarda contra reparar um nome legitimo que so nao foi importado."""
    codigo = "import json\nx = pydantic.validate(y)\n"
    _, mapa = reparar(codigo)
    assert mapa == {}


def test_pos_processar_junta_restauro_e_reparo() -> None:
    """O pipeline do `gerar_codigo.py`: restaurar o caso e depois reparar."""
    from gerar_codigo import pos_processar

    bruto = (
        "from fastapi import fastapi, httpexception\n"
        "app = fastapi()\n"
        "class comentario(basemodel):\n"
        "    pass\n"
        "x = q(comentar)\n"
    )
    codigo, mapa = pos_processar(bruto)
    assert "from fastapi import FastAPI, HTTPException" in codigo
    assert "class comentario(BaseModel):" in codigo
    assert "q(comentario)" in codigo
    assert mapa == {"comentar": "comentario"}


def test_codigo_nao_parseavel_vira_value_error() -> None:
    with pytest.raises(ValueError, match="nao parseavel"):
        reparar("def f(:\n")


def test_mapa_de_sufixo_ignora_o_proprio_typo() -> None:
    declarados = {"deletar_comentar", "comentar", "outra_coisa"}
    assert mapa_de_sufixo(declarados, {"comentar": "comentario"}) == {
        "deletar_comentar": "deletar_comentario"
    }


def test_nomes_declarados_nao_inclui_import_nem_builtin() -> None:
    definidos = nomes_declarados(ast.parse(CODIGO))
    assert "comentario" in definidos
    assert "deletar_comentar" in definidos
    assert "FastAPI" not in definidos
    assert "print" not in definidos


def test_limite_e_prefixo_sao_conservadores() -> None:
    # com LIMITE 3 e PREFIXO 4, nao da para mapear qualquer coisa em qualquer coisa
    assert LIMITE == 3
