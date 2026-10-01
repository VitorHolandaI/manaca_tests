#!/usr/bin/env python3
"""Troca identificadores em ingles por portugues no codigo colhido do FastAPI.

O codigo do docs_src do FastAPI e canonico, mas escrito em ingles. Este script
renomeia so o que o proprio exemplo define -- funcoes, classes, parametros,
variaveis e campos -- deixando intacta toda a API da biblioteca.

A decisao e semantica, via ast, nao textual. Isso importa: em
`FastAPI(description="x")` o `description` e um ast.keyword e nunca e tocado,
enquanto em `class Item(BaseModel): description: str` ele e um ast.Name e vira
`descricao`. Nos dois casos a string e a mesma; so a arvore distingue.

kwargs so sao renomeados quando o chamador e uma classe definida no proprio
exemplo (ex.: `Item(name=..., price=...)` -> `Item(nome=..., preco=...)`), nunca
quando sao API do FastAPI (ex.: `responses=`, `lifespan=`).

Colisoes sao detectadas por arquivo: se `read_main` e `main` fossem ambos
traduzidos para `principal`, os dois voltariam ao original.

Uso:
    python scripts/ptbrificar_identificadores.py \
        --in data/fastapi_oficial.jsonl --out data/fastapi_oficial_ptbr.jsonl
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import io
import json
import tokenize
from dataclasses import dataclass
from pathlib import Path

IDENTIFICADORES: dict[str, str] = {
    # entidades e modelos
    "user": "usuario",
    "users": "usuarios",
    "User": "Usuario",
    "UserInDB": "UsuarioNoBanco",
    "UserIn": "UsuarioEntrada",
    "UserOut": "UsuarioSaida",
    "item": "item",
    "items": "itens",
    "hero": "heroi",
    "heroes": "herois",
    "Hero": "Heroi",
    "client": "cliente",
    "clients": "clientes",
    "message": "mensagem",
    "Message": "Mensagem",
    "settings": "configuracoes",
    "Settings": "Configuracoes",
    "config": "configuracao",
    # campos
    "name": "nome",
    "full_name": "nome_completo",
    "username": "nome_usuario",
    "description": "descricao",
    "title": "titulo",
    "price": "preco",
    "tax": "imposto",
    "price_with_tax": "preco_com_imposto",
    "age": "idade",
    "hashed_password": "senha_hash",
    "size": "tamanho",
    "data": "dados",
    "content": "conteudo",
    "body": "corpo",
    "line": "linha",
    "file": "arquivo",
    "results": "resultados",
    "response": "resposta",
    "request": "requisicao",
    "session": "sessao",
    "db": "bd",
    "exc": "excecao",
    "root": "raiz",
    "commons": "comuns",
    "current_user": "usuario_atual",
    "limit": "limite",
    "skip": "pular",
    "headers": "cabecalhos",
    "cookies": "biscoitos",
    # funcoes
    "read_item": "buscar_item",
    "read_items": "listar_itens",
    "get_item": "buscar_item",
    "get_items": "listar_itens",
    "create_item": "criar_item",
    "update_item": "atualizar_item",
    "delete_item": "deletar_item",
    "read_user": "buscar_usuario",
    "read_users": "listar_usuarios",
    "get_user": "buscar_usuario",
    "create_user": "criar_usuario",
    "update_user": "atualizar_usuario",
    "read_main": "principal",
    "main": "principal",
    "login": "entrar",
    "read_root": "raiz",
}

_NEUTROS = {
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.COMMENT,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
}


@dataclass(frozen=True)
class Troca:
    inicio: int
    fim: int
    novo: str


NoDefinicao = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _e_dunder(nome: str) -> bool:
    return nome.startswith("__") and nome.endswith("__")


def _offsets(codigo: str) -> list[int]:
    """Offset em caracteres do inicio de cada linha (indice 0 = linha 1)."""
    inicio = [0]
    for linha in io.StringIO(codigo).readlines():
        inicio.append(inicio[-1] + len(linha))
    return inicio


def _col_char(linha: str, col_byte: int) -> int:
    """Converte o col_offset do ast (bytes UTF-8) para coluna em caracteres."""
    return len(linha.encode("utf-8")[:col_byte].decode("utf-8", "ignore"))


def _tokens(codigo: str) -> list[tokenize.TokenInfo]:
    return list(tokenize.generate_tokens(io.StringIO(codigo).readline))


def nomes_locais(tree: ast.AST) -> set[str]:
    """Todos os nomes que o proprio exemplo define."""
    locais: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            locais.add(node.name)
            args = node.args
            alvos = [*args.args, *args.kwonlyargs, *args.posonlyargs]
            if args.vararg:
                alvos.append(args.vararg)
            if args.kwarg:
                alvos.append(args.kwarg)
            locais.update(a.arg for a in alvos)
        elif isinstance(node, ast.ClassDef):
            locais.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            locais.add(node.id)
    return {n for n in locais if not _e_dunder(n) and n not in ("self", "cls")}


def classes_locais(tree: ast.AST) -> set[str]:
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def resolver_colisoes(locais: set[str]) -> dict[str, str]:
    """Mapa aplicavel, sem deixar dois nomes distintos virarem o mesmo."""
    final = {n: IDENTIFICADORES.get(n, n) for n in locais}
    while True:
        contagem = collections.Counter(final.values())
        duplicados = {v for v, c in contagem.items() if c > 1}
        if not duplicados:
            break
        for valor in duplicados:
            grupo = sorted(n for n, v in final.items() if v == valor)
            for nome in grupo[1:]:
                final[nome] = nome
    return {n: v for n, v in final.items() if n != v}


def _token_nome_def(
    tokens: list[tokenize.TokenInfo], node: NoDefinicao, nome: str
) -> tokenize.TokenInfo | None:
    """Acha o token do nome numa FunctionDef/ClassDef (o ast aponta pro 'def')."""
    candidatos = []
    for i, tok in enumerate(tokens):
        if tok.type != tokenize.NAME or tok.string != nome:
            continue
        j = i - 1
        while j >= 0 and tokens[j].type in _NEUTROS:
            j -= 1
        if j >= 0 and tokens[j].string in ("def", "class"):
            candidatos.append(tok)
    if not candidatos:
        return None
    for tok in candidatos:
        if tok.start[0] == node.lineno:
            return tok
    return candidatos[0]


def _spans(
    codigo: str, tree: ast.AST, tokens: list[tokenize.TokenInfo], mapa: dict[str, str]
) -> list[Troca]:
    linhas = codigo.splitlines()
    inicio = _offsets(codigo)
    classes = classes_locais(tree)
    trocas: list[Troca] = []

    def da_ast(lineno: int, col: int, fim_col: int, novo: str) -> None:
        base = inicio[lineno - 1]
        c = _col_char(linhas[lineno - 1], col)
        trocas.append(Troca(base + c, base + c + fim_col, novo))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapa:
            da_ast(node.lineno, node.col_offset, len(node.id), mapa[node.id])
        elif isinstance(node, ast.arg) and node.arg in mapa:
            da_ast(node.lineno, node.col_offset, len(node.arg), mapa[node.arg])
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in mapa
        ):
            tok = _token_nome_def(tokens, node, node.name)
            if tok is not None:
                base = inicio[tok.start[0] - 1]
                trocas.append(Troca(base + tok.start[1], base + tok.end[1], mapa[node.name]))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in classes
        ):
            for kw in node.keywords:
                if kw.arg in mapa:
                    da_ast(kw.lineno, kw.col_offset, len(kw.arg), mapa[kw.arg])
    return sorted(trocas, key=lambda t: t.inicio)


def _aplicar(codigo: str, tree: ast.AST, mapa: dict[str, str]) -> tuple[str, int]:
    if not mapa:
        return codigo, 0
    trocas = _spans(codigo, tree, _tokens(codigo), mapa)
    partes: list[str] = []
    pos = 0
    for troca in trocas:
        if troca.inicio < pos:
            continue
        partes += [codigo[pos : troca.inicio], troca.novo]
        pos = troca.fim
    partes.append(codigo[pos:])
    return "".join(partes), len(trocas)


def aplicar_mapa(codigo: str, mapa: dict[str, str]) -> tuple[str, int]:
    """Troca identificadores segundo `mapa`, preservando strings e comentarios.

    So toca ast.Name, ast.arg, nome de def/class e kwarg de classe local -- a
    decisao e semantica, nao textual. Devolve (codigo_novo, numero de trocas).

    Exemplo: aplicar_mapa("x = velho\\n", {"velho": "novo"}) -> ("x = novo\\n", 1)
    Levanta ValueError se o codigo nao for parseavel.
    """
    try:
        tree = ast.parse(codigo)
    except SyntaxError as err:
        raise ValueError(f"codigo nao parseavel ({err}); esperado Python valido") from err
    return _aplicar(codigo, tree, mapa)


def renomear(codigo: str) -> tuple[str, int, int]:
    """Devolve (codigo_ptbr, nº de trocas, nº de colisoes evitadas).

    Levanta ValueError se o codigo nao for Python parseavel.
    """
    try:
        tree = ast.parse(codigo)
    except SyntaxError as err:
        raise ValueError(f"codigo nao parseavel ({err}); esperado Python valido") from err
    locais = nomes_locais(tree)
    mapa = resolver_colisoes(locais)
    colisoes = sum(1 for n in locais if n in IDENTIFICADORES and n not in mapa)
    novo, trocas = _aplicar(codigo, tree, mapa)
    return novo, trocas, colisoes


def traduzir_linha(linha: dict[str, str]) -> tuple[dict[str, str], int, int, bool]:
    try:
        novo, trocas, colisoes = renomear(linha["output"])
    except ValueError:
        return linha, 0, 0, False
    saida = dict(linha)
    saida["output"] = novo
    try:
        ast.parse(novo)
        valido = True
    except SyntaxError:
        valido = False
    return saida, trocas, colisoes, valido


def escrever(itens: list[dict[str, str]], destino: Path) -> dict[str, object]:
    destino.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with destino.open("w", encoding="utf-8") as fh:
        for item in itens:
            linha = json.dumps(item, ensure_ascii=False)
            digest.update(linha.encode("utf-8") + b"\n")
            fh.write(linha + "\n")
    return {"exemplos": len(itens), "sha256": digest.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Renomeia identificadores para portugues (ast-aware)"
    )
    ap.add_argument("--in", dest="entrada", default="data/fastapi_oficial.jsonl")
    ap.add_argument("--out", dest="saida", default="data/fastapi_oficial_ptbr.jsonl")
    args = ap.parse_args()

    origem = Path(args.entrada)
    if not origem.is_file():
        raise SystemExit(f"nao achei {origem}; rode o harvest_fastapi_docs.py antes")

    linhas = [
        json.loads(texto)
        # split("\n"): splitlines() quebra no \u2028 cru de dentro das strings
        for texto in origem.read_text(encoding="utf-8").split("\n")
        if texto.strip()
    ]
    saida_itens: list[dict[str, str]] = []
    total_trocas = total_colisoes = quebrados = inalterados = 0
    for linha in linhas:
        novo, trocas, colisoes, valido = traduzir_linha(linha)
        if not valido or trocas == 0:
            if not valido:
                quebrados += 1
                novo = linha
            else:
                inalterados += 1
        total_trocas += trocas if novo is not linha else 0
        total_colisoes += colisoes
        saida_itens.append(novo)

    destino = Path(args.saida)
    info = escrever(saida_itens, destino)
    manifest = {
        "arquivo": str(destino),
        "origem": str(origem),
        "exemplos": info["exemplos"],
        "sha256": info["sha256"],
        "identificadores_renomeados": total_trocas,
        "colisoes_evitadas": total_colisoes,
        "nao_alterados": inalterados,
        "quebrados_descartados": quebrados,
        "mapa": IDENTIFICADORES,
        "licenca": "obra derivada de codigo MIT; manter o aviso de copyright",
    }
    (destino.parent / "manifest_ptbr.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"entrada            : {origem} ({len(linhas)} exemplos)")
    print(f"renomeacoes        : {total_trocas}")
    print(f"colisoes evitadas  : {total_colisoes}")
    print(f"nao alterados      : {inalterados}")
    print(f"quebrados (revert) : {quebrados}")
    print(f"saida              : {destino} (sha256 {info['sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
