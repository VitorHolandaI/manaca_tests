#!/usr/bin/env python3
"""Colhe nomes de entidade em portugues dos nomes de classe do codigo colhido.

Com 339 nomes no SFT o modelo acertou a classe de uma entidade inedita em 29%
dos pedidos do teste amplo, contra 13% com 73 (2026-10-02). Este script tira
nomes de duas fontes: classes de uma palavra do codigo pt colhido
(`class Fatura(`) que tambem aparecem na Wikipedia pt colhida, e palavras da
Wikipedia com sufixo de substantivo (`-cao`, `-mento`, `-dade`...). Fica com
os --maximo mais frequentes. Exclui todo nome ja usado pelo gen_dataset,
inclusive os de teste.

Uso (na 3090, onde ficam os JSONL colhidos):
    python scripts/colher_nomes_classe.py --saida scripts/nomes_colhidos.tsv
"""

from __future__ import annotations

import argparse
import json
import keyword
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from gen_dataset import ENTIDADES_TESTE
from nomes_entidades import NOMES_ENTIDADES, NOMES_EXTRAS, NOMES_TESTE_AMPLO

# uma palavra capitalizada so: `ItemPedido` exigiria saber onde partir
CLASSE = re.compile(r"^[ \t]*class ([A-Z][a-z]{3,14})[ \t]*[(:]", re.MULTILINE)
PALAVRA = re.compile(r"[a-z]+")
MINIMO_WIKI_PADRAO = 20
MAXIMO_PADRAO = 2000
# sufixos que quase so formam substantivo; `-or` sozinho pegaria `melhor`
SUFIXOS_DE_SUBSTANTIVO = (
    "cao",
    "mento",
    "dade",
    "agem",
    "ista",
    "eiro",
    "eira",
    "ario",
    "aria",
    "ura",
    "ncia",
    "dor",
    "tor",
)
# portugues raramente termina em outra letra: corta `object`, `music`, `fermat`
FINAIS_DO_PORTUGUES = frozenset("aeiolmrz")
# nomes que colidem com variavel ou sufixo do molde do gen_dataset
PROIBIDOS = frozenset({"registro", "sessao", "resposta", "dados", "base", "criar"})


def dobrar_acentos(texto: str) -> str:
    """Tira acento e cedilha, como nos identificadores do dataset.

    Exemplo: dobrar_acentos("comentário") == "comentario"
    """
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def pluralizar(singular: str) -> str:
    """Plural regular do portugues para um nome sem acento.

    Exemplo: pluralizar("transacao") == "transacoes"
    """
    if singular.endswith("ao"):
        return singular[:-2] + "oes"
    if singular.endswith("il"):
        return singular[:-2] + "is"
    if singular.endswith(("al", "el", "ol", "ul")):
        return singular[:-1] + "is"
    if singular.endswith("m"):
        return singular[:-1] + "ns"
    if singular.endswith(("r", "z", "s")):
        return singular + "es"
    return singular + "s"


def nomes_de_classe(codigo: str) -> list[str]:
    """Nomes de classe de uma palavra so, na ordem em que aparecem."""
    return CLASSE.findall(codigo)


def parece_entidade(nome: str) -> bool:
    """Descarta verbo no infinitivo, participio, plural e final nao portugues.

    Exemplo: parece_entidade("fatura") is True; parece_entidade("receber") is False
    """
    if nome.endswith(("ar", "er", "ir", "ado", "ido", "s")):
        return False
    return nome[-1:] in FINAIS_DO_PORTUGUES


def nomes_por_sufixo(wiki: Counter[str], minimo: int) -> set[str]:
    """Palavras da Wikipedia com sufixo de substantivo e frequencia >= minimo."""
    return {
        palavra
        for palavra, vezes in wiki.items()
        if vezes >= minimo
        and 5 <= len(palavra) <= 15
        and palavra.endswith(SUFIXOS_DE_SUBSTANTIVO)
        and parece_entidade(palavra)
    }


def frequencias_de_palavras(textos: Iterable[str]) -> Counter[str]:
    """Contagem de palavras sem acento e em minuscula."""
    contagem: Counter[str] = Counter()
    for texto in textos:
        contagem.update(PALAVRA.findall(dobrar_acentos(texto).lower()))
    return contagem


def selecionar(
    classes: Counter[str],
    wiki: Counter[str],
    minimo_wiki: int,
    excluidos: set[str],
    maximo: int,
) -> list[tuple[str, str]]:
    """(singular, plural) dos `maximo` nomes mais frequentes na Wikipedia pt.

    Exemplo: selecionar(classes, wiki, 20, {"recibo"}, 2000)
    """
    do_codigo = {
        c.lower()
        for c in classes
        if parece_entidade(c.lower()) and wiki[c.lower()] >= minimo_wiki
    }
    candidatos = (do_codigo | nomes_por_sufixo(wiki, minimo_wiki)) - excluidos
    candidatos = {n for n in candidatos if not keyword.iskeyword(n)}
    # empate de frequencia desfeito pelo nome: a saida nao depende de hash
    mais_frequentes = sorted(candidatos, key=lambda n: (-wiki[n], n))[:maximo]
    return [(n, pluralizar(n)) for n in sorted(mais_frequentes)]


def ler_textos(caminho: Path) -> list[str]:
    """Campo `text` de cada linha de um JSONL colhido."""
    if not caminho.is_file():
        raise FileNotFoundError(f"jsonl colhido nao encontrado: {caminho}")
    # split("\n"): splitlines() quebra no   cru de dentro das strings
    linhas = caminho.read_text(encoding="utf-8").split("\n")
    return [json.loads(linha)["text"] for linha in linhas if linha.strip()]


def escrever_tsv(nomes: list[tuple[str, str]], destino: Path) -> None:
    """Uma linha `singular<TAB>plural` por nome, em ordem alfabetica."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    linhas = [f"{s}\t{p}\n" for s, p in sorted(nomes)]
    destino.write_text("".join(linhas), encoding="utf-8")


def _excluidos() -> set[str]:
    usados = NOMES_ENTIDADES + NOMES_EXTRAS + NOMES_TESTE_AMPLO
    return {s for s, _ in usados} | set(ENTIDADES_TESTE) | set(PROIBIDOS)


def main() -> int:
    ap = argparse.ArgumentParser(description="Colhe nomes de entidade pt do codigo")
    ap.add_argument("--codigo", default="data/colhido/codigo_ptbr.jsonl")
    ap.add_argument("--wiki", default="data/colhido/wiki_ptbr.jsonl")
    ap.add_argument("--saida", default="scripts/nomes_colhidos.tsv")
    ap.add_argument("--minimo-wiki", type=int, default=MINIMO_WIKI_PADRAO)
    ap.add_argument("--maximo", type=int, default=MAXIMO_PADRAO)
    args = ap.parse_args()

    classes: Counter[str] = Counter()
    for codigo in ler_textos(Path(args.codigo)):
        classes.update(nomes_de_classe(codigo))
    wiki = frequencias_de_palavras(ler_textos(Path(args.wiki)))
    nomes = selecionar(classes, wiki, args.minimo_wiki, _excluidos(), args.maximo)
    escrever_tsv(nomes, Path(args.saida))
    print(
        f"[nomes] {len(classes)} classes de uma palavra, {len(nomes)} aceitas "
        f"-> {args.saida}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
