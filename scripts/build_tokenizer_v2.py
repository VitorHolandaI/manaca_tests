#!/usr/bin/env python3
"""Constroi o tokenizer v2: abre o canal de maiuscula e a forma sem espaco.

Sobre o tokenizer oficial (Unigram, byte_fallback, 64000 pecas):

  1. remove o normalizer Lowercase (mantem NFKC) -- maiuscula passa a existir
  2. appenda letras maiusculas e palavras cased do corpus -- canal de emissao
  3. appenda a forma SEM espaco das palavras frequentes -- mata o garfo que
     troca 'deletar_comentario' por 'deletar_comentar'

Nenhuma peca original e alterada: as 64000 primeiras mantem o id. Os ids novos
comecam em 64000, entao o embedding antigo continua valido e so as linhas novas
precisam de init (media das pecas constituintes, na etapa de treino).

Uso:
    python scripts/build_tokenizer_v2.py \
        --corpus data/upstream/fastapi \
        --saida models/manaca-hf-official/tokenizer_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer

MAIUSCULAS_ASCII = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
MAIUSCULAS_ACENTUADAS = list("ÁÀÂÃÄÅÇÉÈÊËÍÌÎÏÑÓÒÔÕÖÚÙÛÜÝ")
Cased: re.Pattern[str] = re.compile(r"\b[A-ZÁÀÂÃÄÅÇÉÈÊËÍÌÎÏÑÓÒÔÕÖÚÙÛÜÝ][A-Za-z0-9_]*\b")

SCORE_LETRA = -6.0
SCORE_CASED_MIN = -14.0


def _ler_texto(caminho: Path) -> str:
    if not caminho.exists():
        raise FileNotFoundError(f"corpus inexistente: {caminho}")
    if caminho.is_file():
        return caminho.read_text(errors="ignore")
    partes: list[str] = []
    for filho in sorted(caminho.rglob("*")):
        if filho.is_file() and filho.suffix in {".py", ".md", ".txt", ".rst"}:
            partes.append(filho.read_text(errors="ignore"))
    if not partes:
        raise ValueError(f"corpus vazio (nenhum .py/.md/.txt/.rst): {caminho}")
    return "\n".join(partes)


def palavras_cased(caminhos: Iterable[Path]) -> Counter[str]:
    """Conta tokens capitalizados no corpus; a contagem vira o score Unigram."""
    contagem: Counter[str] = Counter()
    for caminho in caminhos:
        contagem.update(Cased.findall(_ler_texto(caminho)))
    return contagem


def formas_sem_espaco(vocab: dict[str, float], score_min: float) -> dict[str, float]:
    """Forma nua das pecas '▁palavra' frequentes que ainda nao existem no vocab.

    Reusa o score da peca com espaco: e a mesma palavra, entao a probabilidade
    estimada e a mesma -- so muda o contexto de fronteira.
    """
    novas: dict[str, float] = {}
    for token, score in vocab.items():
        if not token.startswith("▁") or score < score_min:
            continue
        nua = token[1:]
        if len(nua) < 3 or not nua.isalpha() or nua in vocab:
            continue
        novas[nua] = score
    return novas


def scores_cased(contagem: Counter[str], teto: int) -> dict[str, float]:
    """Score log-prob das palavras capitalizadas mais frequentes do corpus."""
    total = sum(contagem.values()) or 1
    novas: dict[str, float] = {}
    for palavra, n in contagem.most_common(teto):
        score = math.log(n / total)
        if score >= SCORE_CASED_MIN:
            novas[palavra] = score
    return novas


def montar_vocab(
    base: dict[str, float], cased: dict[str, float], nuas: dict[str, float]
) -> list[tuple[str, float]]:
    """Concatena as pecas novas no fim; a ordem preserva os ids originais."""
    novas: dict[str, float] = {}
    for letra in MAIUSCULAS_ASCII + MAIUSCULAS_ACENTUADAS:
        novas.setdefault(letra, SCORE_LETRA)
    for palavra, score in cased.items():
        novas.setdefault(palavra, score)
        novas.setdefault("▁" + palavra, score)
    for palavra, score in nuas.items():
        novas.setdefault(palavra, score)
    for palavra in list(novas):
        if palavra in base:
            del novas[palavra]
    return [(t, s) for t, s in base.items()] + list(novas.items())


def construir(origem: Path, vocab_final: list[tuple[str, float]]) -> Tokenizer:
    from tokenizers import Tokenizer, models, normalizers

    tk = Tokenizer.from_file(str(origem))
    tk.model = models.Unigram(vocab_final, unk_id=0, byte_fallback=True)
    tk.normalizer = normalizers.Sequence([normalizers.NFKC()])
    return tk


def validar(tk: Tokenizer) -> list[str]:
    """Checagens de aceitacao: round-trip exato e peca sem espaco no garfo."""
    casos = [
        "deletar_comentario",
        "class Item(BaseModel):",
        "FastAPI",
        "Brasil",
        "from fastapi import HTTPException",
    ]
    falhas: list[str] = []
    for caso in casos:
        decodificado = tk.decode(tk.encode(caso).ids)
        if decodificado != caso:
            falhas.append(f"round-trip: {caso!r} -> {decodificado!r}")
    tokens = tk.encode("deletar_comentario").tokens
    if "comentario" not in tokens:
        falhas.append(f"garfo vivo: deletar_comentario -> {tokens}")
    tokens_cased = tk.encode("FastAPI").tokens
    if not {"FastAPI", "▁FastAPI"} & set(tokens_cased):
        falhas.append(f"FastAPI fragmentada: {tokens_cased}")
    return falhas


def main() -> int:
    ap = argparse.ArgumentParser(description="Constroi o tokenizer v2 do Manaca.")
    raiz = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--tokenizer",
        type=Path,
        default=raiz / "models/manaca-hf-official/tokenizer.json",
    )
    ap.add_argument(
        "--corpus", type=Path, nargs="+", required=True, help="arquivos/pastas cased"
    )
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument(
        "--score-min", type=float, default=-15.0, help="corte p/ forma sem espaco"
    )
    ap.add_argument(
        "--teto-cased", type=int, default=4000, help="max de palavras capitalizadas"
    )
    args = ap.parse_args()

    oficial = json.loads(args.tokenizer.read_text())
    base: dict[str, float] = {t: s for t, s in oficial["model"]["vocab"]}

    contagem = palavras_cased(args.corpus)
    cased = scores_cased(contagem, args.teto_cased)
    nuas = formas_sem_espaco(base, args.score_min)

    vocab_final = montar_vocab(base, cased, nuas)
    tk = construir(args.tokenizer, vocab_final)
    tk.save(str(args.saida))

    print(f"vocab: {len(base)} -> {len(vocab_final)} (+{len(vocab_final) - len(base)})")
    print(f"  cased do corpus : {len(cased)}")
    print(f"  formas sem espaco: {len(nuas)}")
    falhas = validar(tk)
    if falhas:
        print("VALIDACAO FALHOU:")
        for f in falhas:
            print("  -", f)
        return 1
    print("validacao: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
