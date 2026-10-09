#!/usr/bin/env python3
"""Chat no terminal com o Manaca-FastAPI (CPT v4 + adapter do SFT).

Monta o prompt, codifica a caixa e corta no EOS como o eval_code.py, para o
modelo ver exatamente o formato do treino. Sem GPU NVIDIA nao ha bitsandbytes:
o base e carregado em bf16 e arredondado para a grade NF4 (nf4_simulado.py),
senao o adapter erra o cabecalho (`connect_args=Depends`). Roda em CPU ou na
GPU Intel (xpu), se o torch tiver suporte.

Uso:
    uv run --python 3.12 --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --with-requirements requirements-chat.txt python scripts/chat_fastapi.py
    ... scripts/chat_fastapi.py --prompt "Crie um endpoint para listar recibos"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from eval_code import (
    _base_do_adapter,
    cortar_no_fim,
    desfazer_marcador,
    prompts_para,
)
from nf4_simulado import simular_nf4
from template_alpaca import LEGADO, Template, ler_template

ADAPTER_PADRAO = "checkpoints/manaca-fastapi-v4"
MAX_NOVOS_PADRAO = 1024


def _dispositivo() -> str:
    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def carregar(adapter: str, nf4: bool, device: str):  # type: ignore[no-untyped-def]
    """(tokenizer, modelo) do adapter sobre o base que ele declara."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter)
    base = _base_do_adapter(adapter)
    modelo = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    if nf4:
        print(
            f"[chat] {simular_nf4(modelo)} camadas lineares na grade NF4",
            file=sys.stderr,
        )
    modelo = PeftModel.from_pretrained(modelo, adapter)
    return tok, modelo.to(device).eval()


def responder(
    tok, modelo, pedido: str, device: str, max_novos: int, template: Template = LEGADO
) -> str:  # type: ignore[no-untyped-def]
    """Codigo gerado para um pedido, ja com a caixa devolvida."""
    import torch

    prompt = prompts_para(
        [{"instruction": pedido.strip(), "output": ""}], tok, template
    )[0]
    entrada = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        saida = modelo.generate(
            input_ids=entrada["input_ids"],
            attention_mask=entrada["attention_mask"],
            max_new_tokens=max_novos,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    fins = {t for t in (tok.eos_token_id, tok.pad_token_id) if t is not None}
    gerados = cortar_no_fim(saida[0][entrada["input_ids"].shape[1] :].tolist(), fins)
    texto = tok.decode(gerados, skip_special_tokens=True).strip()
    return desfazer_marcador([texto], tok)[0]


def _laco(tok, modelo, device: str, max_novos: int, template: Template) -> None:  # type: ignore[no-untyped-def]
    print("Pedido em portugues (Enter envia; 'sair' ou Ctrl-D encerra).")
    while True:
        try:
            pedido = input("\n> ")
        except EOFError:
            return
        if pedido.strip().lower() in {"sair", "exit", "quit"}:
            return
        if not pedido.strip():
            continue
        inicio = time.monotonic()
        print(responder(tok, modelo, pedido, device, max_novos, template))
        print(f"[{time.monotonic() - inicio:.0f} s]", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Chat com o Manaca-FastAPI (v4)")
    ap.add_argument("--adapter", default=ADAPTER_PADRAO)
    ap.add_argument("--prompt", default=None, help="um pedido so, sem o laco")
    ap.add_argument("--max-novos", type=int, default=MAX_NOVOS_PADRAO)
    ap.add_argument("--device", default=None, help="cpu ou xpu (padrao: o que houver)")
    ap.add_argument(
        "--sem-nf4", action="store_true", help="base em bf16 puro (erra o cabecalho)"
    )
    args = ap.parse_args()

    device = args.device or _dispositivo()
    print(f"[chat] carregando {args.adapter} em {device}...", file=sys.stderr)
    tok, modelo = carregar(args.adapter, not args.sem_nf4, device)
    template = ler_template(Path(args.adapter))
    if args.prompt:
        print(responder(tok, modelo, args.prompt, device, args.max_novos, template))
        return 0
    _laco(tok, modelo, device, args.max_novos, template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
