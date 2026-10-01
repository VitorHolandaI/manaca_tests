#!/usr/bin/env python3
"""Restaura a capitalizacao canonica de bibliotecas no codigo gerado pelo Manaca.

O Manaca-1B e case-blind, entao o modelo emite tudo minusculo. Este pos-processador
devolve o caso canonico dos nomes de biblioteca (FastAPI, HTTPException, Depends,
BaseModel, Column, Session, True, False, None) para o codigo virar Python valido.

E token-aware -- usa tokenize e troca apenas tokens NAME que casam exatamente,
preservando strings, comentarios e o nome do modulo nos imports (em
'from fastapi import fastapi' o modulo continua 'fastapi').

Uso:
    python scripts/restore_case.py < saida.py > valido.py
"""

from __future__ import annotations

import io
import tokenize

CANONICOS: dict[str, str] = {
    "fastapi": "FastAPI",
    "httpexception": "HTTPException",
    "depends": "Depends",
    "basemodel": "BaseModel",
    "column": "Column",
    "integer": "Integer",
    "string": "String",
    "boolean": "Boolean",
    "numeric": "Numeric",
    "session": "Session",
    "true": "True",
    "false": "False",
    "none": "None",
}


_NEUTROS = {tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.COMMENT}


def _tokens(codigo: str) -> list[tokenize.TokenInfo]:
    try:
        return list(tokenize.generate_tokens(io.StringIO(codigo).readline))
    except (tokenize.TokenError, SyntaxError) as err:
        # SyntaxError cobre o IndentationError que o tokenize levanta em dedent torto
        raise ValueError(
            f"codigo nao tokenizavel ({err}); esperado Python tokenizavel"
        ) from err


def _offsets_linha(codigo: str) -> list[int]:
    # readlines() quebra linha igual ao tokenize; splitlines() tambem quebra em \x0c
    offsets = [0]
    for linha in io.StringIO(codigo).readlines():
        offsets.append(offsets[-1] + len(linha))
    return offsets


def nomes_a_restaurar(codigo: str) -> list[tokenize.TokenInfo]:
    """Tokens NAME que restaurar() troca, pulando o nome do modulo em imports.

    Exemplo: em 'from fastapi import fastapi' so o segundo 'fastapi' volta.
    Levanta ValueError se o codigo nem for tokenizavel.
    """
    alvos: list[tokenize.TokenInfo] = []
    em_modulo = False
    inicio_de_linha = True
    for tok in _tokens(codigo):
        if tok.type == tokenize.NEWLINE:
            em_modulo, inicio_de_linha = False, True
            continue
        if tok.type != tokenize.NAME:
            inicio_de_linha = inicio_de_linha and tok.type in _NEUTROS
            continue
        # 'from' so abre import no inicio da linha; 'raise x from none' restaura
        if inicio_de_linha and tok.string in ("from", "import"):
            em_modulo = True
        elif tok.string == "import":
            em_modulo = False
        elif not em_modulo and tok.string in CANONICOS:
            alvos.append(tok)
        inicio_de_linha = False
    return alvos


def restaurar(codigo: str) -> str:
    """Troca tokens NAME minusculos pelos nomes canonicos das bibliotecas.

    Exemplo: 'from fastapi import httpexception' -> 'from fastapi import HTTPException'
    Levanta ValueError se o codigo nem for tokenizavel.
    """
    offsets = _offsets_linha(codigo)
    partes: list[str] = []
    pos = 0
    for tok in nomes_a_restaurar(codigo):
        ini = offsets[tok.start[0] - 1] + tok.start[1]
        partes += [codigo[pos:ini], CANONICOS[tok.string]]
        pos = offsets[tok.end[0] - 1] + tok.end[1]
    partes.append(codigo[pos:])
    return "".join(partes)


def main() -> int:
    import sys

    entrada = sys.stdin.read()
    if not entrada.strip():
        raise SystemExit(
            "nada na entrada; use: python scripts/restore_case.py < saida.py"
        )
    sys.stdout.write(restaurar(entrada))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
