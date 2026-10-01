#!/usr/bin/env python3
"""Repara identificadores que nao existem no codigo gerado pelo Manaca.

Dois defeitos medidos, os dois de tokenizacao, e nenhum dos dois foi consertado
por treino -- nem LoRA, nem subword regularization, nem fine-tune completo
(ver `decisions/` no ai-memory):

  1. **Entidade trocada dentro do identificador.** Pedido `endereco`, o modelo
     escreve `deletar_comentar`, `comentar_resposta`, `comentárioresposta`. Ele
     nao deriva a grafia sem-espaco de uma palavra que so conhece como token
     unico (`▁comentario` e `com|enta|rio` sao coisas diferentes), e recorre ao
     verbo concorrente ou ao prior ortografico.
  2. **Acento vazando na convencao ASCII.** `comentário` onde o dataset usa
     `comentario`.

A reparacao e deterministica e roda sem GPU:

  1. tira acento de identificador;
  2. nome USADO que nao esta definido em lugar nenhum e typo, e o alvo e o nome
     definido mais proximo por distancia de edicao.

A segunda passada so pode ajudar: o nome nao existe, entao o codigo quebrava de
qualquer forma. Os guardas existem para nao mapear um nome que legitimamente vem
de fora -- distancia pequena E prefixo comum longo.

Exemplo:
    reparar("x = s.query(comentar)\\nclass comentario(base): pass\\n")
    devolve o codigo com `comentar` -> `comentario`.
"""

from __future__ import annotations

import ast
import builtins
import sys
import unicodedata

from ptbrificar_identificadores import aplicar_mapa

LIMITE = 3
PREFIXO = 4


def sem_acento(nome: str) -> str:
    """Tira os acentos de um identificador, preservando o resto do texto.

    Exemplo: sem_acento("comentário") -> "comentario".
    """
    decomposto = unicodedata.normalize("NFD", nome)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposto if not unicodedata.combining(c))
    )


def _e_dunder(nome: str) -> bool:
    return nome.startswith("__") and nome.endswith("__")


def _parse(codigo: str) -> ast.AST:
    try:
        return ast.parse(codigo)
    except SyntaxError as err:
        raise ValueError(
            f"codigo nao parseavel ({err}); esperado Python valido"
        ) from err


def _identificadores(tree: ast.AST) -> set[str]:
    """Nomes que o codigo declara ou usa, de qualquer posicao."""
    achados: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            achados.add(node.id)
        elif isinstance(node, ast.arg):
            achados.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            achados.add(node.name)
        elif isinstance(node, ast.Attribute):
            achados.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            achados.add(node.arg)
    return achados


def mapa_sem_acento(tree: ast.AST) -> dict[str, str]:
    """Identificadores acentuados -> versao sem acento.

    Exemplo: {"endereço": "endereco", "comentário": "comentario"}.
    """
    return {
        nome: sem_acento(nome)
        for nome in _identificadores(tree)
        if sem_acento(nome) != nome
    }


def nomes_definidos(tree: ast.AST) -> set[str]:
    """O que o codigo declara: imports, classes, funcoes, argumentos, atribuicoes.

    Builtins entram junto, senao `print` ou `len` pareceriam typo.
    """
    definidos = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definidos.add(node.name)
        elif isinstance(node, ast.arg):
            definidos.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            definidos.add(node.id)
        elif isinstance(node, ast.alias):
            definidos.add((node.asname or node.name).split(".")[0])
    return definidos


def distancia(a: str, b: str) -> int:
    """Distancia de edicao (Levenshtein).

    Exemplo: distancia("comentar", "comentario") -> 2.
    """
    anterior = list(range(len(b) + 1))
    for i, caractere_de_a in enumerate(a, 1):
        atual = [i]
        for j, caractere_de_b in enumerate(b, 1):
            atual.append(
                min(
                    anterior[j] + 1,
                    atual[j - 1] + 1,
                    anterior[j - 1] + (caractere_de_a != caractere_de_b),
                )
            )
        anterior = atual
    return anterior[-1]


def _prefixo_comum(a: str, b: str) -> int:
    comum = 0
    for caractere_de_a, caractere_de_b in zip(a, b):
        if caractere_de_a != caractere_de_b:
            break
        comum += 1
    return comum


def mapa_de_reparo(tree: ast.AST) -> dict[str, str]:
    """Nome usado e nao definido -> nome definido mais proximo.

    Os guardas: distancia <= LIMITE e prefixo comum >= PREFIXO. Sem eles, um
    nome que legitimamente vem de fora poderia ser mapeado para outro qualquer
    -- e o codigo passaria a compilar errado em silencio.
    """
    definidos = nomes_definidos(tree)
    faltando = {
        nome
        for nome in (n.id for n in ast.walk(tree) if isinstance(n, ast.Name))
        if nome not in definidos and not _e_dunder(nome)
    }
    mapa: dict[str, str] = {}
    for typo in faltando:
        _, _, alvo = min(
            (distancia(typo, d), -_prefixo_comum(typo, d), d) for d in definidos
        )
        if distancia(typo, alvo) <= LIMITE and _prefixo_comum(typo, alvo) >= PREFIXO:
            mapa[typo] = alvo
    return mapa


def nomes_declarados(tree: ast.AST) -> set[str]:
    """So o que o proprio codigo declara: classes, funcoes, argumentos, atribuicoes.

    Diferente de `nomes_definidos`, nao inclui imports nem builtins -- serve
    para achar nomes que terminam num typo sem mexer no que vem de fora.
    """
    declarados: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            declarados.add(node.name)
        elif isinstance(node, ast.arg):
            declarados.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            declarados.add(node.id)
    return declarados


def mapa_de_sufixo(declarados: set[str], mapa: dict[str, str]) -> dict[str, str]:
    """Nomes declarados que TERMINAM num typo tambem sao corrigidos.

    `deletar_comentar` nao e "usado e nao definido" -- ele e definido, entao a
    primeira regra nao o pega. Mas o sufixo e o mesmo typo, e o marcador da
    avaliacao exige o nome exato.
    """
    extra: dict[str, str] = {}
    for nome in declarados:
        for typo, alvo in mapa.items():
            if nome != typo and nome.endswith(typo):
                extra[nome] = nome[: -len(typo)] + alvo
    return extra


def reparar(codigo: str) -> tuple[str, dict[str, str]]:
    """Devolve (codigo_reparado, mapa_aplicado).

    Exemplo: reparar("x = q(comentar)\\nclass comentario(base): pass\\n")
    devolve o codigo com `comentar` trocado por `comentario`.
    Levanta ValueError se o codigo nao for parseavel.
    """
    sem_acentos, _ = aplicar_mapa(codigo, mapa_sem_acento(_parse(codigo)))
    tree = _parse(sem_acentos)
    mapa = mapa_de_reparo(tree)
    mapa.update(mapa_de_sufixo(nomes_declarados(tree), mapa))
    reparado, _ = aplicar_mapa(sem_acentos, mapa)
    return reparado, mapa


def main() -> int:
    entrada = sys.stdin.read()
    if not entrada.strip():
        raise SystemExit(
            "nada na entrada; use: python scripts/reparar_identificadores.py < saida.py"
        )
    codigo, mapa = reparar(entrada)
    if mapa:
        print(f"# reparado: {mapa}", file=sys.stderr)
    sys.stdout.write(codigo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
