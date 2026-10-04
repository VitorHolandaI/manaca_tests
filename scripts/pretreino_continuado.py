#!/usr/bin/env python3
"""Pre-treino continuado do Manaca-1B base no corpus cased, com o tokenizer v2.

E aqui que as 51148 linhas novas do v2 aprendem: o fine-tune e completo (bf16)
porque a LoRA congela as embeddings. As linhas novas nascem da media das pecas
antigas (scripts/expandir_vocab.py); o resto do modelo parte do base.

Com o tokenizer v3 (build_tokenizer_v3.py) a maiuscula vira marcador: os
textos passam por caixa_marcada.codificar (v4: com fronteira) e o modelo de
origem e remapeado peca por peca (remapear_vocab.py). Isso liga sozinho pelo vocab.

Esquecimento: a val loss e medida ANTES do primeiro passo (linha de base) e a
cada --eval-steps. Val subindo acima da linha de base = o modelo esta perdendo
o que sabia; pare e baixe o lr.

ATENCAO: nao executado na maquina de desenvolvimento (sem NVIDIA). Rode na RTX
3090 com --smoke primeiro, no stack de requirements-3090.txt.

Uso:
    python scripts/expandir_vocab.py            # monta models/manaca-tokenizer-v2
    python scripts/montar_corpus_cpt.py         # monta data/cpt/
    python scripts/pretreino_continuado.py \\
        --model AIInstituteLNCC/manaca-1b-base \\
        --out checkpoints/manaca-cpt-cased --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from caixa_marcada import (
    MARCADORES,
    codificar_documentos,
    usa_caixa_marcada,
    usa_fronteira,
)
from expandir_vocab import expandir_modelo
from remapear_vocab import remapear_modelo

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


def carregar_textos(caminho: Path) -> list[str]:
    """Campo `text` de cada linha do JSONL do corpus (montar_corpus_cpt.py)."""
    if not caminho.is_file():
        raise FileNotFoundError(f"corpus nao encontrado: {caminho}")
    textos: list[str] = []
    # split("\n"), nao splitlines(): este tambem quebra em \u2028 e \x85, que
    # o json.dumps(ensure_ascii=False) deixa crus dentro das strings
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").split("\n"), 1):
        if not linha.strip():
            continue
        try:
            textos.append(json.loads(linha)["text"])
        except (json.JSONDecodeError, KeyError) as err:
            raise ValueError(
                f"{caminho}:{numero}: esperado JSON com campo 'text' ({err})"
            ) from err
    if not textos:
        raise ValueError(f"corpus vazio: {caminho}")
    return textos


def empacotar(docs: list[list[int]], bloco: int, eos_id: int) -> list[list[int]]:
    """Concatena os docs separados por EOS e corta em blocos de `bloco` tokens.

    A sobra do fim e descartada: um bloco curto precisaria de padding e de
    mascara, e perder < bloco tokens de um corpus inteiro nao pesa.
    Exemplo: empacotar([[5, 6], [7]], bloco=2, eos_id=2) -> [[5, 6], [2, 7]]
    """
    if bloco < 1:
        raise ValueError(f"bloco invalido: {bloco}, esperado int >= 1")
    corrido: list[int] = []
    for ids in docs:
        corrido += ids + [eos_id]
    return [corrido[i : i + bloco] for i in range(0, len(corrido) - bloco + 1, bloco)]


def indices_espacados(total: int, maximo: int) -> list[int]:
    """Ate `maximo` indices espalhados em range(total), em ordem.

    A val sai agrupada por origem (codigo do upstream, codigo colhido, wiki);
    pegar os primeiros blocos mediria so a primeira origem.
    Exemplo: indices_espacados(10, 3) -> [0, 3, 6]
    """
    if maximo < 1:
        raise ValueError(f"--max-val-blocos invalido: {maximo}, esperado >= 1")
    if total <= maximo:
        return list(range(total))
    passo = total / maximo
    return [int(i * passo) for i in range(maximo)]


def montar_blocos(
    textos: list[str], tokenizer: PreTrainedTokenizerBase, bloco: int
) -> Dataset:
    """Dataset de blocos com labels == input_ids (loss em todo token)."""
    from datasets import Dataset

    ids = tokenizer(textos, add_special_tokens=False)["input_ids"]
    blocos = empacotar(ids, bloco, tokenizer.eos_token_id)
    if not blocos:
        raise ValueError(
            f"corpus tem menos de {bloco} tokens; baixe --bloco ou aumente o corpus"
        )
    return Dataset.from_dict({"input_ids": blocos, "labels": [list(b) for b in blocos]})


def preparar_textos(textos: list[str], tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Codifica a maiuscula em marcador quando o tokenizer e o v3."""
    if not usa_caixa_marcada(tokenizer):
        return textos
    codificados, descartados = codificar_documentos(textos, usa_fronteira(tokenizer))
    print(f"[cpt] caixa marcada; {descartados} docs com marcador literal sairam")
    return codificados


def _carregar(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """(tokenizer, modelo bf16 com o embedding no tamanho do tokenizer)."""
    import torch
    from train_qlora import caminho_do_tokenizer_json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    modelo = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    modelo.config.use_cache = False
    if usa_caixa_marcada(tokenizer):
        origem = AutoTokenizer.from_pretrained(args.model).get_vocab()
        destino = tokenizer.get_vocab()
        # origem igual ao destino: o modelo ja e v3 (retomada), nada a remapear
        if origem != destino:
            novos = tuple(m for m in MARCADORES if m in destino and m not in origem)
            copiadas, iniciadas = remapear_modelo(modelo, origem, destino, novos)
            print(
                f"[cpt] remapeado: {copiadas} linhas copiadas por peca, {iniciadas} novas"
            )
        return tokenizer, modelo
    novas = expandir_modelo(
        modelo,
        caminho_do_tokenizer_json(args.model),
        Path(args.tokenizer) / "tokenizer.json",
    )
    print(f"[cpt] {novas} linhas novas de embedding iniciadas pela media das pecas")
    return tokenizer, modelo


def treinar(args: argparse.Namespace) -> None:
    from transformers import Trainer, TrainingArguments, default_data_collator

    tokenizer, modelo = _carregar(args)
    textos_treino = preparar_textos(carregar_textos(Path(args.treino)), tokenizer)
    textos_val = preparar_textos(carregar_textos(Path(args.val)), tokenizer)
    treino = montar_blocos(textos_treino, tokenizer, args.bloco)
    val = montar_blocos(textos_val, tokenizer, args.bloco)
    val = val.select(indices_espacados(len(val), args.max_val_blocos))
    print(f"[cpt] {len(treino)} blocos de treino, {len(val)} de val ({args.bloco} tok)")

    config = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        # mesmo calculo do train_qlora --full: Adam fp32 nao cabe na 3090
        optim="adamw_bnb_8bit",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        report_to=[],
    )
    treinador = Trainer(
        model=modelo,
        args=config,
        train_dataset=treino,
        eval_dataset=val,
        data_collator=default_data_collator,
    )
    base = treinador.evaluate()
    print(f"[cpt] val loss ANTES do treino (linha de base): {base['eval_loss']:.4f}")
    treinador.train()
    final = treinador.evaluate()
    print(
        f"[cpt] val loss final: {final['eval_loss']:.4f} (base {base['eval_loss']:.4f})"
    )
    treinador.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    Path(args.out, "val_loss.json").write_text(
        json.dumps({"antes": base["eval_loss"], "depois": final["eval_loss"]}) + "\n",
        encoding="utf-8",
    )
    print(f"[cpt] modelo + tokenizer ({args.tokenizer}) salvos em {args.out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-treino continuado cased (Manaca-1B)")
    ap.add_argument("--model", default="AIInstituteLNCC/manaca-1b-base")
    ap.add_argument("--tokenizer", default="models/manaca-tokenizer-v2")
    ap.add_argument("--treino", default="data/cpt/treino.jsonl")
    ap.add_argument("--val", default="data/cpt/val.jsonl")
    ap.add_argument("--out", default="checkpoints/manaca-cpt-cased")
    ap.add_argument("--bloco", type=int, default=2048)
    # 1 epoca: com o corpus colhido (dezenas de milhoes de tokens) repetir dado
    # so aumenta o esquecimento
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    # mesmo lr do train_qlora --full: 2e-4 apagaria o pre-treino
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument(
        "--max-val-blocos",
        type=int,
        default=96,
        help="teto da val; cada eval custa um forward por bloco",
    )
    ap.add_argument("--smoke", action="store_true", help="2 passos e para")
    treinar(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
