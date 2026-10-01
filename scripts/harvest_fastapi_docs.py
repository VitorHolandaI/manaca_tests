#!/usr/bin/env python3
"""Colhe os exemplos oficiais do FastAPI com o contexto em portugues do Brasil.

O repositorio do FastAPI e MIT, e a documentacao em docs/pt/ e a traducao
oficial mantida pela comunidade -- em pt-BR (arquivo 205x, ficheiro 0x; usuario
133x, utilizador 0x). Cada pagina em docs/pt/docs/ cita o arquivo de codigo por
caminho, no formato `{* ../../docs_src/x/tutorial001.py *}`, o que da um join
deterministico entre a prosa pt-BR e o codigo canonico.

Saida: JSONL no esquema Alpaca {instruction, input, output}, onde o instruction e
a prosa pt-BR oficial e o output e o exemplo de codigo do docs_src. Nada aqui e
traduzido por LLM: a prosa ja e humana, o codigo ja e o oficial.

Atencao: o codigo do docs_src e compartilhado entre os 12 idiomas, entao ele vem
em ingles. Este script entrega a base MIT; a troca de identificadores para
portugues e um passo posterior e separado.

Uso:
    python scripts/harvest_fastapi_docs.py --out data/fastapi_oficial.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import posixpath
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_PADRAO = "https://github.com/fastapi/fastapi"
MARCA = "docs_src/"
PREAMBULO = (
    "Abaixo esta uma instrucao que descreve uma tarefa. "
    "Escreva uma resposta que atenda adequadamente ao pedido."
)

_RE_INCLUDE = re.compile(r"\{\*\s*([^\s}]+\.py)[^}]*\}")
_RE_H1 = re.compile(r"^#\s+(\S.*?)\s*$")
_RE_HEAD = re.compile(r"^(#{2,6})\s+(\S.*?)\s*$")
_RE_ANCORA = re.compile(r"\s*\{\s*#[^}]*\}\s*$")
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")


@dataclass(frozen=True)
class Cabecalho:
    linha: int
    nivel: int
    titulo: str


def clonar(repo_url: str, destino: Path, atualizar: bool) -> str:
    """Clona o repo em profundidade 1 (ou atualiza) e devolve o commit."""
    if (destino / ".git").is_dir():
        if atualizar:
            subprocess.run(["git", "-C", str(destino), "pull", "--ff-only"],
                           check=True, capture_output=True, text=True)
    else:
        destino.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(destino)],
                       check=True, capture_output=True, text=True)
    saida = subprocess.run(["git", "-C", str(destino), "rev-parse", "HEAD"],
                           check=True, capture_output=True, text=True).stdout
    return saida.strip()


def resolver(doc_rel: str, include: str) -> str:
    base = posixpath.dirname(doc_rel)
    caminho = posixpath.normpath(posixpath.join(base, include))
    if MARCA in caminho and not caminho.startswith(MARCA):
        caminho = caminho[caminho.index(MARCA):]
    return caminho


def titulo_do_doc(linhas: list[str]) -> str:
    for linha in linhas:
        m = _RE_H1.match(linha)
        if m:
            return _RE_ANCORA.sub("", m.group(1)).strip()
    return ""


def cabecalhos(linhas: list[str]) -> list[Cabecalho]:
    achados = []
    for i, linha in enumerate(linhas):
        m = _RE_HEAD.match(linha)
        if m:
            titulo = _RE_ANCORA.sub("", m.group(2)).strip()
            achados.append(Cabecalho(i, len(m.group(1)), titulo))
    return achados


def limpar_prosa(texto: str) -> str:
    """Tira marcação markdown e devolve o ultimo paragrafo util, em uma linha."""
    texto = _RE_INCLUDE.sub("", texto)
    texto = _RE_IMG.sub("", texto)
    texto = _RE_LINK.sub(r"\1", texto)
    texto = re.sub(r"(?m)^\s*///.*$", "", texto)
    texto = re.sub(r"(?m)^\s*#.*$", "", texto)
    texto = re.sub(r"(?m)^\s*[|>].*$", "", texto)
    texto = re.sub(r"<[^>]+>", "", texto)
    paragrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    if not paragrafos:
        return ""
    return " ".join(paragrafos[-1].split())


def prosa_antes(linhas: list[str], cabs: list[Cabecalho], alvo: int) -> str:
    inicio = 0
    for cab in cabs:
        if cab.linha < alvo:
            inicio = cab.linha + 1
    return limpar_prosa("\n".join(linhas[inicio:alvo]))


def secao_antes(cabs: list[Cabecalho], alvo: int) -> str:
    titulo = ""
    for cab in cabs:
        if cab.linha < alvo:
            titulo = cab.titulo
    return titulo


def instrucao(titulo_doc: str, secao: str, prosa: str) -> str:
    partes = [p for p in (titulo_doc, secao) if p]
    cabeca = " - ".join(partes) if partes else "exemplo de fastapi"
    return f"{cabeca}. {prosa}".strip() if prosa else cabeca


def codigo_valido(codigo: str) -> bool:
    try:
        ast.parse(codigo)
        return True
    except SyntaxError:
        return False


def colher(repo: Path, lang: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    raiz = repo / "docs" / lang / "docs"
    if not raiz.is_dir():
        raise FileNotFoundError(
            f"nao achei docs/{lang}/docs em {repo}; esperado clone do fastapi"
        )
    pares: list[dict[str, str]] = []
    vistos: set[tuple[str, str]] = set()
    conta = {"docs": 0, "citacoes": 0, "ausentes": 0, "invalidos": 0, "duplicados": 0}
    for md in sorted(raiz.rglob("*.md")):
        texto = md.read_text(encoding="utf-8", errors="replace")
        linhas = texto.splitlines()
        rel = md.relative_to(repo).as_posix()
        cabs = cabecalhos(linhas)
        nome_doc = titulo_do_doc(linhas)
        conta["docs"] += 1
        for m in _RE_INCLUDE.finditer(texto):
            conta["citacoes"] += 1
            alvo = texto[: m.start()].count("\n")
            codigo_rel = resolver(rel, m.group(1))
            arquivo = repo / codigo_rel
            if not arquivo.is_file():
                conta["ausentes"] += 1
                continue
            if (rel, codigo_rel) in vistos:
                conta["duplicados"] += 1
                continue
            codigo = arquivo.read_text(encoding="utf-8", errors="replace")
            if not codigo_valido(codigo):
                conta["invalidos"] += 1
                continue
            vistos.add((rel, codigo_rel))
            pares.append(
                {
                    "instruction": instrucao(
                        nome_doc, secao_antes(cabs, alvo), prosa_antes(linhas, cabs, alvo)
                    ),
                    "input": "",
                    "output": codigo,
                    "_origem": f"{rel} -> {codigo_rel}",
                }
            )
    return pares, conta


def minusculizar(pares: list[dict[str, str]]) -> list[dict[str, str]]:
    """Baixa tudo para caixa baixa, pelo contrato case-blind do Manaca-1B."""
    return [
        {
            "instruction": p["instruction"].lower(),
            "input": "",
            "output": p["output"].lower(),
            "_origem": p["_origem"],
        }
        for p in pares
    ]


def escrever(pares: list[dict[str, str]], destino: Path,
             com_origem: bool) -> dict[str, object]:
    destino.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with destino.open("w", encoding="utf-8") as fh:
        for par in pares:
            ex = {k: v for k, v in par.items() if k != "_origem" or com_origem}
            linha = json.dumps(ex, ensure_ascii=False)
            digest.update(linha.encode("utf-8") + b"\n")
            fh.write(linha + "\n")
    return {"exemplos": len(pares), "sha256": digest.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Colhe exemplos oficiais do FastAPI (MIT)")
    ap.add_argument("--repo", type=Path, default=Path("data/upstream/fastapi"))
    ap.add_argument("--repo-url", default=REPO_PADRAO)
    ap.add_argument("--lang", default="pt", help="idioma da prosa (docs/<lang>)")
    ap.add_argument("--out", default="data/fastapi_oficial.jsonl")
    ap.add_argument("--minusculo", action="store_true",
                    help="baixa instruction e output para caixa baixa")
    ap.add_argument("--com-origem", action="store_true",
                    help="inclui o campo _origem (doc -> codigo) no JSONL")
    ap.add_argument("--sem-clone", action="store_true",
                    help="exige um --repo ja clonado; nao toca na rede")
    ap.add_argument("--atualizar", action="store_true")
    args = ap.parse_args()

    if args.sem_clone and not (args.repo / ".git").is_dir():
        raise SystemExit(f"{args.repo} nao e um clone do fastapi; remova --sem-clone")

    commit = "nao-clonado"
    if not args.sem_clone:
        commit = clonar(args.repo_url, args.repo, args.atualizar)

    pares, conta = colher(args.repo, args.lang)
    if args.minusculo:
        pares = minusculizar(pares)

    destino = Path(args.out)
    info = escrever(pares, destino, args.com_origem)
    tamanho = sum(len(p["output"].encode("utf-8")) for p in pares)
    manifest = {
        "arquivo": str(destino),
        "exemplos": info["exemplos"],
        "sha256": info["sha256"],
        "bytes_de_codigo": tamanho,
        "idioma_da_prosa": args.lang,
        "fonte": args.repo_url,
        "commit": commit,
        "licenca_fonte": "MIT (FastAPI, (c) Sebastian Ramirez e contribuidores)",
        "licenca_deste_arquivo": "obra derivada de codigo MIT; manter o aviso de copyright",
        "minusculo": bool(args.minusculo),
        "contagem": conta,
        "formato": "Alpaca {instruction, input, output}",
        "observacao": "prosa oficial pt-BR + codigo canonico do docs_src (em ingles)",
    }
    (destino.parent / "manifest_oficial.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"docs varridos      : {conta['docs']}")
    print(f"citacoes de codigo : {conta['citacoes']}")
    print(f"  ausentes         : {conta['ausentes']}")
    print(f"  invalidos        : {conta['invalidos']}")
    print(f"  duplicados       : {conta['duplicados']}")
    print(f"pares escritos     : {info['exemplos']} ({tamanho/1024:.1f} KB de codigo)")
    print(f"sha256             : {info['sha256']}")
    print(f"saida              : {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
