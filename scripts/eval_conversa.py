#!/usr/bin/env python3
"""Mede se o modelo conversa em vez de responder tudo com um arquivo FastAPI.

O v4 treinado so com FastAPI respondeu "oi" com `class Oi(Base)`. O SFT com o
mix do LNCC precisa manter o FastAPI (teste amplo do eval_code.py) e
responder pedidos gerais em texto. A metrica e a fracao de pedidos gerais
cuja resposta nao tem `from fastapi` nem `@app.`; as respostas ficam salvas
para leitura. Gera como o eval_code (greedy, base em 4 bits com --base-4bit,
template gravado no adapter).

Uso (na 3060):
    python scripts/eval_conversa.py --model checkpoints/manaca-fastapi-instruct \\
        --base-4bit --saida results/conversa_instruct.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PEDIDOS: list[dict[str, str]] = [
    {"instruction": "Oi"},
    {"instruction": "Bom dia, tudo bem?"},
    {"instruction": "Quem foi Machado de Assis?"},
    {"instruction": "Qual é a capital da Austrália?"},
    {"instruction": "Explique o que é fotossíntese em poucas frases."},
    {"instruction": "Me dê três dicas para dormir melhor."},
    {"instruction": "Escreva um poema curto sobre o mar."},
    {"instruction": "Como faço um café coado?"},
    {"instruction": "Qual a diferença entre clima e tempo?"},
    {"instruction": "Sugira um nome para um gato laranja."},
    {"instruction": "O que é uma API, em linguagem simples?"},
    {"instruction": "Liste cinco frutas tropicais."},
    {"instruction": "Por que o céu é azul?"},
    {"instruction": "Escreva uma mensagem de aniversário para um amigo."},
    {
        "instruction": "Traduza o texto a seguir para o inglês.",
        "input": "O gato dorme no sofá.",
    },
    {
        "instruction": "Resuma o texto a seguir em poucas frases.",
        "input": (
            "A Floresta Amazônica é a maior floresta tropical do mundo e cobre boa "
            "parte do norte do Brasil. Ela abriga milhões de espécies e tem papel "
            "importante no clima do planeta, mas sofre com o desmatamento."
        ),
    },
    {"instruction": "Classifique o sentimento da frase.", "input": "Adorei o filme!"},
    {"instruction": "Quanto é 15 vezes 4?"},
]


def virou_codigo(resposta: str) -> bool:
    """A resposta e um arquivo FastAPI (o que o v4 so-FastAPI fazia com tudo)."""
    return "from fastapi" in resposta or "@app." in resposta


def taxa_de_conversa(respostas: list[str]) -> float:
    """Fracao das respostas que nao viraram codigo; 0.0 sem respostas."""
    if not respostas:
        return 0.0
    return round(sum(not virou_codigo(r) for r in respostas) / len(respostas), 4)


def main() -> int:
    import torch
    from eval_code import (
        _carregar_para_geracao,
        _gerar_em_lotes,
        desfazer_marcador,
        prompts_para,
    )
    from template_alpaca import ler_template

    ap = argparse.ArgumentParser(description="Eval de conversa (nao virar FastAPI)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--saida", default="results/conversa.jsonl")
    ap.add_argument("--base-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, modelo = _carregar_para_geracao(args.model, device, args.base_4bit)
    itens = [{"output": "", "input": "", **p} for p in PEDIDOS]
    prompts = prompts_para(itens, tok, ler_template(Path(args.model)))
    gerados = _gerar_em_lotes(modelo, tok, prompts, device, args.max_new_tokens, 6)
    respostas = desfazer_marcador(gerados, tok)
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open("w", encoding="utf-8") as fh:
        for pedido, resposta in zip(PEDIDOS, respostas):
            linha = {
                **pedido,
                "output": resposta,
                "virou_codigo": virou_codigo(resposta),
            }
            fh.write(json.dumps(linha, ensure_ascii=False) + "\n")
    taxa = taxa_de_conversa(respostas)
    print(json.dumps({"total": len(respostas), "conversa_rate": taxa}, indent=2))
    print(f"[conversa] respostas em {saida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
