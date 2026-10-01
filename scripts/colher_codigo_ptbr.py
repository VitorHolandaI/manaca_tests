#!/usr/bin/env python3
"""Colhe Python com identificadores e comentarios em portugues do GitHub.

Fonte: codeparrot/github-code-clean (880 shards parquet, 314 GB, campo
`license` por arquivo). Fica so Python de licenca permissiva cujo texto tem
pelo menos --minimo marcadores portugueses FORTES distintos -- palavras que
nao colidem com ingles (`senha`, `usuario`, `retorna`; nunca `do`, `as`, `os`).

Medido no shard 0 (2026-09-30): 5053 Python permissivos, 7 em portugues
(~47k caracteres). O portugues e ~0,1% do Python, entao o volume so aparece
varrendo todos os shards. Cada shard e baixado, filtrado e apagado; o
progresso fica em <saida>.feitos, entao o script retoma de onde parou.

Uso:
    python scripts/colher_codigo_ptbr.py --saida data/colhido/codigo_ptbr.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "codeparrot/github-code-clean"
N_SHARDS = 880
LICENCAS_PERMISSIVAS = frozenset(
    {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "unlicense", "isc", "cc0-1.0"}
)
# texto corrido le e revisa melhor que 70 literais soltos
MARCADORES_FORTES = frozenset(
    """
    nao não usuario usuário usuarios senha arquivo arquivos dados valor valores
    lista retorna retornar funcao função criar deletar atualizar buscar cliente
    produto pedido nome idade endereco endereço quantidade preco preço tambem
    também entao então voce você obter salvar carregar calcular resultado
    mensagem erro pasta caminho entrada saida saída numero número soma uma que
    para pelo pela quando onde mais menos esta está sao são ser foi dos das ao
    """.split()  # noqa: SIM905
)
MINIMO_PADRAO = 5
# arquivo gigante costuma ser dado embutido ou gerado, nao codigo escrito
MAX_CARACTERES = 100_000
PALAVRA = re.compile(r"[A-Za-zÀ-ÿ]+")
# quebra camelCase: `obterUsuario` -> `obter`, `Usuario`
SUBPALAVRA = re.compile(r"[A-ZÀ-Ý]?[a-zà-ÿ]+")


def marcadores_pt(codigo: str) -> set[str]:
    """Marcadores fortes presentes em identificadores e comentarios.

    Exemplo: marcadores_pt("def obterUsuario(senha): # nao salva")
    -> {'obter', 'usuario', 'senha', 'nao', 'salva'} & MARCADORES_FORTES
    """
    vistos: set[str] = set()
    for palavra in PALAVRA.findall(codigo):
        vistos.update(p.lower() for p in SUBPALAVRA.findall(palavra))
    return vistos & MARCADORES_FORTES


def aceitar(linguagem: str, licenca: str, codigo: str, minimo: int) -> bool:
    """Python, permissivo, tamanho sao e com portugues suficiente."""
    if linguagem != "Python" or licenca not in LICENCAS_PERMISSIVAS:
        return False
    if len(codigo) > MAX_CARACTERES:
        return False
    return len(marcadores_pt(codigo)) >= minimo


def filtrar_shard(parquet: Path, minimo: int) -> list[dict[str, str]]:
    """Documentos aceitos de um shard, no formato do corpus do CPT."""
    import pyarrow.parquet as pq

    tabela = pq.read_table(
        parquet, columns=["language", "license", "code", "repo_name", "path"]
    )
    colunas = [tabela.column(c).to_pylist() for c in tabela.column_names]
    documentos: list[dict[str, str]] = []
    for linguagem, licenca, codigo, repo, caminho in zip(*colunas):
        if aceitar(linguagem, licenca, codigo, minimo):
            documentos.append(
                {
                    "text": codigo,
                    "origem": "codigo_ptbr",
                    "caminho": f"{repo}/{caminho}",
                    "licenca": licenca,
                }
            )
    return documentos


def _processar(indice: int, minimo: int) -> tuple[int, list[dict[str, str]]]:
    """Baixa um shard para um diretorio temporario, filtra e apaga."""
    from huggingface_hub import hf_hub_download

    nome = f"data/train-{indice:05d}-of-{N_SHARDS:05d}.parquet"
    with tempfile.TemporaryDirectory(prefix="colher_") as pasta:
        caminho = hf_hub_download(REPO, nome, repo_type="dataset", local_dir=pasta)
        return indice, filtrar_shard(Path(caminho), minimo)


def _feitos(progresso: Path) -> set[int]:
    if not progresso.is_file():
        return set()
    return {int(x) for x in progresso.read_text(encoding="utf-8").split()}


def colher(saida: Path, shards: range, minimo: int, paralelo: int) -> tuple[int, int]:
    """Varre os shards pendentes; devolve (documentos novos, caracteres)."""
    progresso = saida.with_suffix(".feitos")
    pendentes = [i for i in shards if i not in _feitos(progresso)]
    saida.parent.mkdir(parents=True, exist_ok=True)
    vistos: set[str] = set()
    total_docs = total_chars = 0
    with (
        ThreadPoolExecutor(max_workers=paralelo) as pool,
        saida.open("a", encoding="utf-8") as fh,
    ):
        for indice, docs in pool.map(lambda i: _processar(i, minimo), pendentes):
            for doc in docs:
                # forks e vendoring repetem o mesmo arquivo em varios repos
                digest = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
                if digest in vistos:
                    continue
                vistos.add(digest)
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total_docs += 1
                total_chars += len(doc["text"])
            fh.flush()
            with progresso.open("a", encoding="utf-8") as fp:
                fp.write(f"{indice}\n")
            print(
                f"[codigo_ptbr] shard {indice}: +{len(docs)} "
                f"(acumulado {total_docs} docs, {total_chars} chars)",
                file=sys.stderr,
            )
    return total_docs, total_chars


def main() -> int:
    ap = argparse.ArgumentParser(description="Colhe Python em portugues do GitHub")
    ap.add_argument("--saida", default="data/colhido/codigo_ptbr.jsonl")
    ap.add_argument("--inicio", type=int, default=0)
    ap.add_argument("--fim", type=int, default=N_SHARDS, help="exclusivo")
    ap.add_argument("--minimo", type=int, default=MINIMO_PADRAO)
    ap.add_argument("--paralelo", type=int, default=4, help="shards ao mesmo tempo")
    args = ap.parse_args()
    if not 0 <= args.inicio < args.fim <= N_SHARDS:
        raise ValueError(
            f"faixa de shards invalida: {args.inicio}..{args.fim}, "
            f"esperado 0 <= inicio < fim <= {N_SHARDS}"
        )
    docs, chars = colher(
        Path(args.saida), range(args.inicio, args.fim), args.minimo, args.paralelo
    )
    print(f"[codigo_ptbr] fim: {docs} documentos, {chars} caracteres", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
