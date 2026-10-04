"""Maiuscula como marcador: `Comentario` vira `<|maj|>comentario`.

Com a maiuscula dentro da peca, `comentario` e `▁Com enta rio` nao tem nada
em comum e o modelo decora a traducao palavra por palavra (entidade_rate
0,32 no teste amplo, 2026-10-03). Com o marcador, a classe e a mesma peca da
instrucao precedida de um token que vale para toda palavra. Esquema de
Land e Meister (arXiv 2608.08847), estendido para camelCase.

O marcador vai ANTES do espaco que precede a palavra: `class<|maj|> comentario`
tokeniza como `▁class <|maj|> ▁comentario`, a mesma peca da instrucao.

Com fronteira=True (tokenizer v4) toda palavra sai na forma com espaco, e
`<|cola|>` diz que o espaco nao existia: `criar_agendamento` vira
`<|cola|> criar_<|cola|> agendamento`. Sem isso `agendamento` colado era outra
peca e perdia para `agenda`, entidade do treino (teste amplo, 2026-10-04).
Texto que comeca com espaco perde esse espaco no decode do tokenizer
(Strip start=1 do Manaca), como ja acontecia no original.
"""

from __future__ import annotations

import re
from typing import Protocol

MAJ = "<|maj|>"
MAI = "<|mai|>"
COLA = "<|cola|>"
MARCADORES = (MAJ, MAI, COLA)
SEQUENCIA_DE_LETRAS = re.compile(r"[^\W\d_]+")
MARCA_E_RESTO = re.compile(r"(<\|maj\|>|<\|mai\|>|<\|cola\|>)")


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


def _palavra(
    antes: list[str], indice: int, tipo: str, parte: str, fronteira: bool
) -> str:
    """Um segmento ja codificado; tira de `antes` o espaco que ele absorve."""
    marcador, minuscula = _marcar(tipo, parte)
    espaco_antes = indice == 0 and antes[-1].endswith(" ")
    if fronteira:
        if espaco_antes:
            antes[-1] = antes[-1][:-1]
        return ("" if espaco_antes else COLA) + marcador + " " + minuscula
    if not (espaco_antes and marcador):
        return marcador + minuscula
    antes[-1] = antes[-1][:-1]
    return marcador + " " + minuscula


def codificar(texto: str, fronteira: bool = False) -> str:
    """Troca maiuscula (e, com fronteira, espaco ausente) por marcador, sem perda.

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
            saida.append(_palavra(saida, indice, tipo, parte, fronteira))
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
    cola, caixa = False, ""
    for marcador, resto in zip(partes[1::2], partes[2::2]):
        if marcador == COLA:
            cola = True
        else:
            caixa = marcador
        if not resto:
            continue
        if cola and resto.startswith(" "):
            resto = resto[1:]
        saida.append(_aplicar(caixa, resto) if caixa else resto)
        cola, caixa = False, ""
    return "".join(saida)


class TokenizerComVocab(Protocol):
    def get_vocab(self) -> dict[str, int]: ...


def usa_caixa_marcada(tokenizer: TokenizerComVocab) -> bool:
    """O tokenizer tem os marcadores de caixa (v3 ou v4)? CPT, SFT e eval ligam aqui."""
    vocab = tokenizer.get_vocab()
    return MAJ in vocab and MAI in vocab


def usa_fronteira(tokenizer: TokenizerComVocab) -> bool:
    """O tokenizer tem `<|cola|>` (v4)? Entao toda palavra vai na forma com espaco."""
    return COLA in tokenizer.get_vocab()


def codificar_para(tokenizer: TokenizerComVocab, texto: str) -> str:
    """codificar com o modo que o tokenizer pede (v3: caixa; v4: caixa + fronteira)."""
    return codificar(texto, fronteira=usa_fronteira(tokenizer))


def codificar_documentos(
    textos: list[str], fronteira: bool = False
) -> tuple[list[str], int]:
    """(textos codificados, quantos tinham o marcador literal e sairam).

    Exemplo: codificar_documentos(["Oi"]) == (["<|maj|>oi"], 0)
    """
    codificados = [
        codificar(t, fronteira) for t in textos if not any(m in t for m in MARCADORES)
    ]
    return codificados, len(textos) - len(codificados)
