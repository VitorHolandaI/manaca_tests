#!/usr/bin/env python3
"""Monta o corpus do pre-treino continuado: codigo Python real + prosa PT-BR, cased.

O pre-treino continuado e onde as 51148 linhas novas do tokenizer v2 aprendem a
ser usadas. Tudo sai do checkout do FastAPI (MIT) em data/upstream/fastapi:

  - codigo: o pacote `fastapi/` e os exemplos de `docs_src/`
  - prosa:  a traducao pt-BR oficial em `docs/pt/`

Fontes extras (--extra): JSONL de colher_codigo_ptbr.py (GitHub, so licenca
permissiva) e colher_wikipedia_ptbr.py (CC BY-SA 3.0), com a licenca de cada
documento preservada no JSONL e somada no manifest.

Guarda de contaminacao: descarta qualquer documento que contenha a classe de
uma entidade do split de teste (`class Comentario`, `class Endereco`) ou o
texto de uma instrucao de teste. O split de validacao sai por fonte, pelo hash
do caminho, entao e deterministico e cobre codigo e prosa.

Uso:
    python scripts/montar_corpus_cpt.py --upstream data/upstream/fastapi \\
        --saida data/cpt --commit a3d205bf19640528718cb4f05ab77f4dfca6ad9a
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from gen_dataset import (
    ENTIDADES_TESTE,
    ENTIDADES_TESTE_AMPLO,
    exemplos_teste_amplo,
    nome_de_classe,
)

FRACAO_VAL = 0.02
# `{* ../../docs_src/x.py hl[2] *}` e `{!> ... !}` sao diretivas do mkdocs: o
# codigo que elas apontam ja entra pelo docs_src, na prosa so vira ruido
DIRETIVA_MKDOCS = re.compile(r"\{[*!][^}]*[*!]\}")


@dataclass(frozen=True)
class Fonte:
    rotulo: str
    pasta: str
    padrao: str


FONTES: tuple[Fonte, ...] = (
    Fonte("codigo", "fastapi", "*.py"),
    Fonte("codigo", "docs_src", "*.py"),
    Fonte("prosa_ptbr", "docs/pt", "*.md"),
)


@dataclass(frozen=True)
class Documento:
    origem: str
    caminho: str
    texto: str
    licenca: str = "mit"


def marcas_de_contaminacao(seed: int) -> tuple[str, ...]:
    """Classes e instrucoes do teste amplo, que contem o split de teste."""
    teste = exemplos_teste_amplo(seed)
    nomes = list(ENTIDADES_TESTE) + [e.singular for e in ENTIDADES_TESTE_AMPLO]
    classes = [f"class {nome_de_classe(n)}" for n in nomes]
    return tuple(classes + sorted({r["instruction"] for r in teste}))


def limpar(texto: str, origem: str) -> str:
    """Tira as diretivas do mkdocs da prosa; codigo passa intacto."""
    if origem != "prosa_ptbr":
        return texto
    return DIRETIVA_MKDOCS.sub("", texto)


def coletar(upstream: Path, fonte: Fonte) -> list[Documento]:
    raiz = upstream / fonte.pasta
    if not raiz.is_dir():
        raise FileNotFoundError(f"pasta da fonte {fonte.rotulo} nao existe: {raiz}")
    documentos: list[Documento] = []
    for arquivo in sorted(raiz.rglob(fonte.padrao)):
        texto = limpar(arquivo.read_text(encoding="utf-8"), fonte.rotulo).strip()
        if texto:
            rel = arquivo.relative_to(upstream).as_posix()
            documentos.append(Documento(fonte.rotulo, rel, texto))
    if not documentos:
        raise ValueError(f"fonte {fonte.rotulo} sem nenhum {fonte.padrao} em {raiz}")
    return documentos


def _hash_do_caminho(doc: Documento) -> str:
    return hashlib.sha256(doc.caminho.encode("utf-8")).hexdigest()


def separar_val(
    documentos: list[Documento], fracao: float
) -> tuple[list[Documento], list[Documento]]:
    """(treino, val) com os menores hashes de caminho indo para val.

    Exemplo: separar_val(docs_de_uma_fonte, 0.02) -- ao menos 1 doc vai para val.
    """
    ordenados = sorted(documentos, key=_hash_do_caminho)
    n_val = max(1, math.ceil(len(ordenados) * fracao))
    val = {d.caminho for d in ordenados[:n_val]}
    return (
        [d for d in documentos if d.caminho not in val],
        [d for d in documentos if d.caminho in val],
    )


def escrever_jsonl(documentos: list[Documento], destino: Path) -> dict[str, object]:
    destino.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with destino.open("w", encoding="utf-8") as fh:
        for doc in documentos:
            linha = json.dumps(
                {
                    "text": doc.texto,
                    "origem": doc.origem,
                    "caminho": doc.caminho,
                    "licenca": doc.licenca,
                },
                ensure_ascii=False,
            )
            digest.update(linha.encode("utf-8") + b"\n")
            fh.write(linha + "\n")
    return {"documentos": len(documentos), "sha256": digest.hexdigest()}


def _caracteres_por(documentos: list[Documento], campo: str) -> dict[str, int]:
    contagem: dict[str, int] = {}
    for doc in documentos:
        chave = getattr(doc, campo)
        contagem[chave] = contagem.get(chave, 0) + len(doc.texto)
    return contagem


def ler_extra(caminho: Path) -> dict[str, list[Documento]]:
    """{origem: documentos} de um JSONL colhido (colher_*.py).

    Exemplo: ler_extra(Path("data/colhido/wiki_ptbr.jsonl"))["prosa_wiki"]
    """
    if not caminho.is_file():
        raise FileNotFoundError(f"corpus extra nao encontrado: {caminho}")
    grupos: dict[str, list[Documento]] = {}
    # split("\n"), nao splitlines(): este tambem quebra em \u2028 e \x85, que
    # o json.dumps(ensure_ascii=False) deixa crus dentro das strings
    linhas = caminho.read_text(encoding="utf-8").split("\n")
    for numero, linha in enumerate(linhas, 1):
        if not linha.strip():
            continue
        try:
            bruto = json.loads(linha)
            doc = Documento(
                bruto["origem"], bruto["caminho"], bruto["text"], bruto["licenca"]
            )
        except (json.JSONDecodeError, KeyError) as err:
            raise ValueError(
                f"{caminho}:{numero}: esperado JSON com text/origem/caminho/licenca "
                f"({err!r})"
            ) from err
        grupos.setdefault(doc.origem, []).append(doc)
    return grupos


def montar(
    upstream: Path, seed: int, extras: tuple[Path, ...] = ()
) -> tuple[list[Documento], list[Documento], int]:
    """(treino, val, descartados por contaminacao) das FONTES e dos extras.

    Cada grupo (fonte do upstream ou origem de um extra) ganha a sua val, para
    a val loss medir codigo e prosa separadamente do mesmo jeito.
    """
    marcas = marcas_de_contaminacao(seed)
    grupos = [coletar(upstream, fonte) for fonte in FONTES]
    for extra in extras:
        grupos += list(ler_extra(extra).values())
    treino: list[Documento] = []
    val: list[Documento] = []
    descartados = 0
    for grupo in grupos:
        limpos = [d for d in grupo if not any(m in d.texto for m in marcas)]
        descartados += len(grupo) - len(limpos)
        parte_treino, parte_val = separar_val(limpos, FRACAO_VAL)
        treino += parte_treino
        val += parte_val
    return treino, val, descartados


def main() -> int:
    ap = argparse.ArgumentParser(description="Corpus do pre-treino continuado (cased)")
    ap.add_argument("--upstream", default="data/upstream/fastapi")
    ap.add_argument("--saida", default="data/cpt")
    ap.add_argument("--commit", default="desconhecido", help="commit do checkout")
    ap.add_argument("--seed", type=int, default=1234, help="seed do gen_dataset")
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="JSONL colhidos (colher_codigo_ptbr.py, colher_wikipedia_ptbr.py)",
    )
    args = ap.parse_args()

    saida = Path(args.saida)
    extras = tuple(Path(e) for e in args.extra)
    treino, val, descartados = montar(Path(args.upstream), args.seed, extras)
    info_treino = escrever_jsonl(treino, saida / "treino.jsonl")
    info_val = escrever_jsonl(val, saida / "val.jsonl")
    manifest = {
        "treino": info_treino,
        "val": info_val,
        "caracteres_treino": _caracteres_por(treino, "origem"),
        "caracteres_val": _caracteres_por(val, "origem"),
        "caracteres_por_licenca": _caracteres_por(treino + val, "licenca"),
        "extras": [str(e) for e in extras],
        "descartados_por_contaminacao": descartados,
        "fontes": [f"{f.pasta}/**/{f.padrao} ({f.rotulo})" for f in FONTES],
        "upstream": "https://github.com/fastapi/fastapi",
        "commit": args.commit,
        "licenca_upstream": "MIT (FastAPI, Sebastian Ramirez)",
    }
    (saida / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
