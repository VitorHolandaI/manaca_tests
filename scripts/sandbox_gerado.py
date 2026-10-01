#!/usr/bin/env python3
"""Executa o codigo gerado como app FastAPI real e chama as rotas que ele declara.

`ast.parse` so pega erro de sintaxe. Codigo com nome trocado compila e so quebra
em runtime -- e isso acontece de verdade: o modelo as vezes escreve `comentar`
onde a classe e `comentario`. Isto sobe o app num TestClient e chama cada rota
declarada, para separar "compila" de "funciona".

Nao exige o CRUD inteiro: o dataset gera UMA operacao por exemplo, entao uma
saida que so tem `@app.delete` nao tem POST nenhum e tem que passar so com o
DELETE. O criterio e: nenhuma rota pode estourar excecao nem responder 5xx.

O payload e inferido da propria classe Pydantic de entrada do codigo gerado: o
modelo escolhe os campos, nao a referencia, entao os campos da referencia nao
servem.

Precisa de fastapi, sqlalchemy e httpx (requirements-dev.txt).

Uso:
    from sandbox_gerado import rodar_rotas
    passou, motivo = rodar_rotas(codigo_restaurado)
"""

from __future__ import annotations

import ast
import os
import tempfile

VALOR_POR_TIPO: dict[str, object] = {"str": "x", "int": 1, "float": 1.5, "bool": True}
SUFIXO_ENTRADA = "criar"
VERBOS = ("get", "post", "put", "delete", "patch")
CORPO = ("post", "put", "patch")
ID_FALSO = "1"


def _valor_da_anotacao(annot: ast.expr) -> object | None:
    if isinstance(annot, ast.Name):
        return VALOR_POR_TIPO.get(annot.id)
    if isinstance(annot, ast.Subscript):
        return _valor_da_anotacao(annot.value)
    return None


def payload_da_entrada(codigo: str) -> dict[str, object] | None:
    """Payload a partir da classe Pydantic de entrada (a que termina em 'criar').

    Exemplo: payload_da_entrada("class usuariocriar(basemodel):\\n    nome: str\\n")
    devolve {"nome": "x"}.
    """
    try:
        tree = ast.parse(codigo)
    except SyntaxError:
        return None
    for node in tree.body:
        # lower(): o dataset cased escreve UsuarioCriar, o minusculo usuariocriar
        if not isinstance(node, ast.ClassDef) or not node.name.lower().endswith(SUFIXO_ENTRADA):
            continue
        payload: dict[str, object] = {}
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if not isinstance(stmt.target, ast.Name):
                continue
            valor = _valor_da_anotacao(stmt.annotation)
            if valor is None:
                return None
            payload[stmt.target.id] = valor
        return payload or None
    return None


def rotas_do_verbo(codigo: str, verbo: str) -> list[str]:
    """Caminhos das rotas decoradas com @app.<verbo>.

    Exemplo: rotas_do_verbo('@app.post("/itens")', "post") -> ['/itens'].
    """
    try:
        tree = ast.parse(codigo)
    except SyntaxError:
        return []
    rotas: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
                continue
            if dec.func.attr != verbo or not dec.args:
                continue
            if isinstance(dec.args[0], ast.Constant):
                rotas.append(str(dec.args[0].value))
    return rotas


def rotas_declaradas(codigo: str) -> list[tuple[str, str]]:
    """Pares (verbo, caminho) de todas as rotas do codigo, na ordem dos verbos."""
    return [(verbo, rota) for verbo in VERBOS for rota in rotas_do_verbo(codigo, verbo)]


def _montar_caminho(rota: str) -> str:
    return rota.replace("{item_id}", ID_FALSO).replace("{id}", ID_FALSO)


def _chamar_rotas(cliente, rotas: list[tuple[str, str]], payload) -> tuple[bool, str | None]:
    for verbo, rota in rotas:
        resposta = cliente.request(
            verbo.upper(), _montar_caminho(rota), json=payload if verbo in CORPO else None
        )
        if resposta.status_code >= 500:
            return False, f"{verbo.upper()} {rota} -> {resposta.status_code}"
    return True, None


def rodar_rotas(codigo: str) -> tuple[bool, str | None]:
    """Sobe o codigo e chama cada rota declarada; devolve (passou, motivo).

    Nao levanta: qualquer excecao do codigo gerado vira o motivo da falha -- o
    erro E o que esta sendo medido. Grava o ./app.db num diretorio temporario.
    """
    from fastapi.testclient import TestClient

    rotas = rotas_declaradas(codigo)
    if not rotas:
        return False, "nenhuma rota @app.<verbo> encontrada"
    payload = payload_da_entrada(codigo)
    anterior = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as pasta:
            os.chdir(pasta)
            escopo: dict[str, object] = {"__name__": "gerado_manaca"}
            exec(compile(codigo, "<gerado>", "exec"), escopo)  # noqa: S102
            return _chamar_rotas(TestClient(escopo["app"]), rotas, payload)  # type: ignore[arg-type]
    except Exception as err:  # noqa: BLE001 -- o erro e o resultado medido
        return False, f"{type(err).__name__}: {err}"
    finally:
        os.chdir(anterior)
