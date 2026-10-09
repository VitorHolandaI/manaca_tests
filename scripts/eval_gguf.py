#!/usr/bin/env python3
"""Gera, pelo llama-server, as respostas de uma amostra do teste amplo.

Serve para comparar arquivos GGUF (F16, Q8_0, Q6_K...) com a mesma metrica do
eval_code.py: grava {instruction, output} e o eval_code --pred pontua. A
amostra pega `--por-entidade` pedidos de cada entidade, porque o comeco do
teste amplo e so de comentario e endereco.

Uso (com o llama-server de pe na --url):
    python scripts/eval_gguf.py --saida results/gguf_q8_pred.jsonl
    python scripts/eval_code.py --pred results/gguf_q8_pred.jsonl \\
        --ref results/gguf_q8_pred.jsonl.ref --executar --sem-restauro
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from chat_gguf import TOKENIZER_PADRAO, URL_PADRAO, ClienteLlamaServer, responder
from eval_code import _carregar_jsonl, _dedupe_por_instrucao
from template_alpaca import TEMPLATES

CLASSE = re.compile(r"^class (\w+)\(Base\):", re.MULTILINE)


def escolher_por_entidade(
    ref: list[dict[str, str]], por_entidade: int
) -> list[dict[str, str]]:
    """Os primeiros `por_entidade` itens de cada classe do model, em ordem.

    Exemplo: escolher_por_entidade(ref, 2) -> 2 pedidos de cada entidade
    """
    contagem: dict[str, int] = {}
    escolhidos: list[dict[str, str]] = []
    for item in ref:
        achado = CLASSE.search(item["output"])
        nome = achado.group(1) if achado else ""
        if contagem.get(nome, 0) < por_entidade:
            contagem[nome] = contagem.get(nome, 0) + 1
            escolhidos.append(item)
    return escolhidos


def main() -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description="Respostas do GGUF para o teste amplo")
    ap.add_argument("--ref", default="data/fastapi_ptbr_teste_amplo.jsonl")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--por-entidade", type=int, default=2)
    ap.add_argument("--url", default=URL_PADRAO)
    ap.add_argument("--tokenizer", default=TOKENIZER_PADRAO)
    ap.add_argument("--template", choices=sorted(TEMPLATES), default="legado")
    args = ap.parse_args()

    ref = escolher_por_entidade(
        _dedupe_por_instrucao(_carregar_jsonl(Path(args.ref))), args.por_entidade
    )
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cliente = ClienteLlamaServer(args.url)
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with (
        saida.open("w", encoding="utf-8") as pred,
        Path(f"{saida}.ref").open("w", encoding="utf-8") as amostra,
    ):
        for numero, item in enumerate(ref, 1):
            texto = responder(
                tok, cliente, item["instruction"], 1024, TEMPLATES[args.template]
            )
            linha = {"instruction": item["instruction"], "output": texto}
            pred.write(json.dumps(linha, ensure_ascii=False) + "\n")
            amostra.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[gguf] {numero}/{len(ref)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
