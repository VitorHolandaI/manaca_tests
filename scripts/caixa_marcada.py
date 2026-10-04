"""Maiuscula como marcador: `Comentario` vira `<|maj|>comentario`.

Com a maiuscula dentro da peca, `comentario` e `▁Com enta rio` nao tem nada
em comum e o modelo decora a traducao palavra por palavra (entidade_rate
0,32 no teste amplo, 2026-10-03). Com o marcador, a classe e a mesma peca da
instrucao precedida de um token que vale para toda palavra. Esquema de
Land e Meister (arXiv 2608.08847), estendido para camelCase.

O marcador vai ANTES do espaco que precede a palavra: `class<|maj|> comentario`
tokeniza como `▁class <|maj|> ▁comentario`, a mesma peca da instrucao.
"""

from __future__ import annotations

import re
from typing import Protocol

MAJ = "<|maj|>"
MAI = "<|mai|>"
MARCADORES = (MAJ, MAI)
SEQUENCIA_DE_LETRAS = re.compile(r"[^\W\d_]+")
MARCA_E_RESTO = re.compile(r"(<\|maj\|>|<\|mai\|>)")


def _reversivel(letra: str) -> bool:
    # `İ`.lower() tem 2 caracteres e `ß`.upper() vira `SS`: esses ficam literais
    minuscula = letra.lower()
    return len(minuscula) == 1 and minuscula.upper() == letra


def _segmentos(letras: str) -> list[tuple[str, str]]:
    """Partes de uma sequencia de letras: ("min"|"maj"|"mai", texto)."""
    partes: list[tuple[str, str]] = []
    i, n = 0, len(letras)
    while i < n:
        j = i
        if not letras[i].isupper():
            while j < n and not letras[j].isupper():
                j += 1
            partes.append(("min", letras[i:j]))
            i = j
            continue
        while j < n and letras[j].isupper():
            j += 1
        if j == n:
            partes.append(("mai" if j - i > 1 else "maj", letras[i:j]))
            i = j
            continue
        # `HTTPException`: a ultima maiuscula abre a palavra seguinte
        if j - i > 1:
            partes.append(("mai", letras[i : j - 1]))
        k = j
        while k < n and not letras[k].isupper():
            k += 1
        partes.append(("maj", letras[j - 1 : k]))
        i = k
    return partes


def _marcar(tipo: str, texto: str) -> tuple[str, str]:
    """(marcador, texto em minuscula), ou ("", texto) se nao for reversivel."""
    if tipo == "maj" and _reversivel(texto[0]):
        return MAJ, texto[0].lower() + texto[1:]
    if tipo == "mai" and all(_reversivel(c) for c in texto):
        return MAI, texto.lower()
    return "", texto


def codificar(texto: str) -> str:
    """Troca maiuscula por marcador, sem perda.

    Exemplo: codificar("class Comentario") == "class<|maj|> comentario"
    """
    for marcador in MARCADORES:
        if marcador in texto:
            posicao = texto.index(marcador)
            raise ValueError(
                f"texto ja tem o marcador {marcador!r} na posicao {posicao}; "
                "esperado texto sem marcador"
            )
    saida: list[str] = []
    fim = 0
    for achado in SEQUENCIA_DE_LETRAS.finditer(texto):
        saida.append(texto[fim : achado.start()])
        for indice, (tipo, parte) in enumerate(_segmentos(achado.group())):
            marcador, minuscula = _marcar(tipo, parte)
            espaco = indice == 0 and bool(marcador) and saida[-1].endswith(" ")
            if espaco:
                saida[-1] = saida[-1][:-1]
            saida.append(marcador + (" " if espaco else "") + minuscula)
        fim = achado.end()
    saida.append(texto[fim:])
    return "".join(saida)


def _aplicar(marcador: str, resto: str) -> str:
    espaco = " " if resto.startswith(" ") else ""
    corpo = resto[len(espaco) :]
    if marcador == MAJ:
        if corpo[:1].islower():
            corpo = corpo[0].upper() + corpo[1:]
        return espaco + corpo
    k = 0
    while k < len(corpo) and corpo[k].islower():
        k += 1
    return espaco + corpo[:k].upper() + corpo[k:]


def decodificar(texto: str) -> str:
    """Desfaz codificar; marcador sem letra depois (gerado solto) some.

    Exemplo: decodificar("class<|maj|> comentario") == "class Comentario"
    """
    partes = MARCA_E_RESTO.split(texto)
    saida = [partes[0]]
    for marcador, resto in zip(partes[1::2], partes[2::2]):
        saida.append(_aplicar(marcador, resto))
    return "".join(saida)


class TokenizerComVocab(Protocol):
    def get_vocab(self) -> dict[str, int]: ...


def usa_caixa_marcada(tokenizer: TokenizerComVocab) -> bool:
    """O tokenizer tem os dois marcadores (v3)? CPT, SFT e eval ligam por aqui."""
    vocab = tokenizer.get_vocab()
    return all(m in vocab for m in MARCADORES)


def codificar_documentos(textos: list[str]) -> tuple[list[str], int]:
    """(textos codificados, quantos tinham o marcador literal e sairam).

    Exemplo: codificar_documentos(["Oi"]) == (["<|maj|>oi"], 0)
    """
    codificados = [codificar(t) for t in textos if not any(m in t for m in MARCADORES)]
    return codificados, len(textos) - len(codificados)
