#!/usr/bin/env python3
"""Gera codigo com o Manaca ajustado e mostra o resultado pos-processado.

Existe para ver o modelo trabalhando numa instrucao qualquer, fora do harness de
avaliacao. Aplica os dois pos-processadores, senao a saida crua sai invalida:

  1. `restore_case.restaurar` -- devolve o caso canonico das bibliotecas, porque
     o modelo e case-blind (0 pecas maiusculas no vocabulario);
  2. `reparar_identificadores.reparar` -- conserta identificador inexistente e
     acento vazado. Ver `decisions/` no ai-memory: treinar nao resolveu isso
     (LoRA, subword e full FT foram medidos e rejeitados).

Uso:
    python scripts/gerar_codigo.py --model checkpoints/manaca-fastapi-qlora-75ent \
        --instrucao "crie um endpoint para cadastrar um livro no fastapi"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_code import _carregar_para_geracao, _gerar_em_lotes, _prompt
from reparar_identificadores import reparar
from restore_case import restaurar

INSTRUCOES_PADRAO: tuple[str, ...] = (
    "crie um endpoint para cadastrar um livro no fastapi com sqlalchemy",
    "como listar todos os alunos cadastrados usando fastapi?",
    "crie um endpoint que busca um ingresso pelo id no fastapi",
)


def escolher_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "xpu" if torch.xpu.is_available() else "cpu"


def gerar(model_id: str, instrucoes: list[str], max_novos: int, lote: int) -> list[str]:
    import torch

    tok, modelo = _carregar_para_geracao(model_id, escolher_device(torch))
    prompts = [_prompt(instrucao) for instrucao in instrucoes]
    return _gerar_em_lotes(modelo, tok, prompts, escolher_device(torch), max_novos, lote)


def pos_processar(bruto: str) -> tuple[str, dict[str, str]]:
    """restore_case + reparo; se nem restaurar passar, devolve o cru."""
    try:
        restaurado = restaurar(bruto)
    except ValueError:
        return bruto, {}
    return reparar(restaurado)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gera codigo FastAPI PT-BR com o Manaca ajustado"
    )
    ap.add_argument("--model", required=True, help="adapter local ou id do HF")
    ap.add_argument(
        "--instrucao",
        action="append",
        default=None,
        help="repita para varias; sem isto usa tres exemplos de CRUD",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--lote", type=int, default=4)
    ap.add_argument("--executar", action="store_true", help="sobe cada saida e chama as rotas")
    ap.add_argument("--salvar", default=None, help="grava um .py por instrucao")
    args = ap.parse_args()

    instrucoes = list(args.instrucao) if args.instrucao else list(INSTRUCOES_PADRAO)
    saidas = gerar(args.model, instrucoes, args.max_new_tokens, args.lote)
    destino = Path(args.salvar) if args.salvar else None
    if destino:
        destino.mkdir(parents=True, exist_ok=True)

    for numero, (instrucao, bruto) in enumerate(zip(instrucoes, saidas), 1):
        codigo, mapa = pos_processar(bruto)
        print("=" * 72)
        print(f"[{numero}] {instrucao}")
        if mapa:
            print(f"    pos-processador consertou: {mapa}")
        print("-" * 72)
        print(codigo)
        if args.executar:
            from sandbox_gerado import rodar_rotas

            passou, motivo = rodar_rotas(codigo)
            print(f"\n-> roda: {'SIM' if passou else 'NAO'} {motivo or ''}")
        if destino:
            caminho = destino / f"exemplo_{numero}.py"
            caminho.write_text(codigo + "\n", encoding="utf-8")
            print(f"\n-> gravado em {caminho}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
