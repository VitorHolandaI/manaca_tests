from __future__ import annotations

import ast

import pytest
from ptbrificar_identificadores import IDENTIFICADORES, renomear, traduzir_linha


def test_kwarg_de_api_fica_intacto_e_so_o_valor_vira_portugues() -> None:
    """Regressao do bug que a versao textual criaria.

    Troca por texto faria `descricao=descricao`; o FastAPI ignoraria o kwarg em
    silencio e a metadata nunca entraria. O `ast` distingue: `description` a
    esquerda e ast.keyword, `description` a direita e ast.Name.
    """
    codigo = 'description = "minha api"\napp = FastAPI(description=description)\n'
    novo, _, _ = renomear(codigo)
    assert "descricao = " in novo
    assert "FastAPI(description=descricao)" in novo
    assert "descricao=descricao" not in novo


def test_campo_de_modelo_e_renomeado() -> None:
    codigo = (
        "from pydantic import BaseModel\n"
        "\n"
        "\n"
        "class Item(BaseModel):\n"
        "    name: str\n"
        "    price: float\n"
    )
    novo, _, _ = renomear(codigo)
    assert "    nome: str\n" in novo
    assert "    preco: float\n" in novo


def test_kwarg_de_classe_local_acompanha_o_campo() -> None:
    codigo = (
        "class Item(BaseModel):\n"
        "    name: str\n"
        "    price: float\n"
        "\n"
        "\n"
        "itens = [Item(name=\"Portal Gun\", price=42.0)]\n"
    )
    novo, _, _ = renomear(codigo)
    assert 'Item(nome="Portal Gun", preco=42.0)' in novo


def test_kwarg_de_classe_da_biblioteca_nao_e_tocado() -> None:
    codigo = (
        "class Item(BaseModel):\n"
        "    name: str\n"
        "\n"
        "\n"
        'app = FastAPI(title="T", tags=["a"], responses={404: {}})\n'
    )
    novo, _, _ = renomear(codigo)
    assert 'FastAPI(title="T", tags=["a"], responses={404: {}})' in novo


def test_atributo_nao_e_renomeado() -> None:
    codigo = 'class Item(BaseModel):\n    name: str\n\n\nx = obj.name\n'
    novo, _, _ = renomear(codigo)
    assert "x = obj.name" in novo


def test_import_nao_e_tocado() -> None:
    codigo = "from fastapi import FastAPI\n\napp = FastAPI()\n"
    novo, trocas, _ = renomear(codigo)
    assert novo == codigo
    assert trocas == 0


def test_string_e_comentario_ficam_intactos() -> None:
    codigo = 'name = "nome"  # name continua no comentario\n'
    novo, _, _ = renomear(codigo)
    assert novo == 'nome = "nome"  # name continua no comentario\n'


def test_acento_antes_do_nome_nao_desloca_a_troca() -> None:
    """O col_offset do ast e contado em bytes UTF-8, nao em caracteres."""
    codigo = 'x = "ação"; name = name\n'
    novo, _, _ = renomear(codigo)
    assert novo == 'x = "ação"; nome = nome\n'


def test_colisao_devolve_um_nome_ao_original() -> None:
    codigo = (
        "def read_main():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def main():\n"
        "    return read_main()\n"
    )
    novo, _, colisoes = renomear(codigo)
    assert colisoes == 1
    assert novo.count("def principal(") == 1
    assert novo.count("def read_main(") == 1
    ast.parse(novo)


def test_dunder_e_self_nao_sao_renomeados() -> None:
    codigo = (
        "class Item(BaseModel):\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n"
    )
    novo, _, _ = renomear(codigo)
    assert "def __init__(self, nome: str)" in novo
    assert "self.name = nome" in novo


def test_alguns_identificadores_do_mapa_sao_estaveis() -> None:
    for nome in ("name", "description", "price", "read_items", "read_main"):
        assert nome in IDENTIFICADORES


@pytest.mark.parametrize("quebrado", ["def f(:\n", "if x:\n        a\n    b\n"])
def test_codigo_nao_parseavel_vira_value_error(quebrado: str) -> None:
    with pytest.raises(ValueError, match="nao parseavel"):
        renomear(quebrado)


def test_traduzir_linha_preserva_os_outros_campos() -> None:
    linha = {
        "instruction": "crie um item",
        "input": "",
        "output": "class Item(BaseModel):\n    name: str\n",
        "_origem": "docs/pt/docs/x.md -> docs_src/x/y.py",
    }
    novo, trocas, _, valido = traduzir_linha(linha)
    assert valido and trocas > 0
    assert novo["instruction"] == linha["instruction"]
    assert novo["_origem"] == linha["_origem"]
    assert "    nome: str\n" in novo["output"]
    assert linha["output"] == "class Item(BaseModel):\n    name: str\n"
