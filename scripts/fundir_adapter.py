#!/usr/bin/env python3
"""Funde o adapter do SFT no base arredondado para NF4 e salva um modelo unico.

O adapter so funciona sobre o base em NF4, com que foi treinado: sobre o bf16
o v4 errava o cabecalho (`connect_args=Depends`). Fundido sobre o base ja
arredondado (nf4_simulado.py), o modelo unico carrega os pesos que o adapter
viu e pode ir para GGUF (llama.cpp/Ollama) sem bitsandbytes. Copia o
tokenizer do adapter e o `tokenizer.model` original, que o conversor do
llama.cpp le para o vocabulario Unigram.

Uso:
    python scripts/fundir_adapter.py --destino models/manaca-fastapi-v4-fundido
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from eval_code import _base_do_adapter
from nf4_simulado import simular_nf4

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel

ADAPTER_PADRAO = "checkpoints/manaca-fastapi-v4"
SENTENCEPIECE_ORIGINAL = Path("models/manaca-hf-official/tokenizer.model")


def carregar_com_nf4(adapter: Path, base: Path, dtype: torch.dtype) -> PreTrainedModel:
    """Base arredondado para NF4 com o adapter aplicado, ainda sem fundir."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    modelo = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    simular_nf4(modelo)
    return PeftModel.from_pretrained(modelo, adapter)


def fundir(adapter: Path, base: Path, destino: Path, dtype: torch.dtype) -> Path:
    """Salva em `destino` o modelo unico (safetensors) e o tokenizer do adapter.

    Exemplo: fundir(Path("checkpoints/manaca-fastapi-v4"), base, destino, bf16)
    """
    from transformers import AutoTokenizer

    unico = carregar_com_nf4(adapter, base, dtype).merge_and_unload()
    unico.save_pretrained(destino, safe_serialization=True)
    if (adapter / "tokenizer_config.json").is_file():
        AutoTokenizer.from_pretrained(adapter).save_pretrained(destino)
    return destino


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description="Funde o adapter no base em NF4 simulado")
    ap.add_argument("--adapter", default=ADAPTER_PADRAO)
    ap.add_argument("--destino", default="models/manaca-fastapi-v4-fundido")
    ap.add_argument("--sentencepiece", default=str(SENTENCEPIECE_ORIGINAL))
    args = ap.parse_args()

    adapter, sentencepiece = Path(args.adapter), Path(args.sentencepiece)
    if not sentencepiece.is_file():
        raise FileNotFoundError(
            f"tokenizer.model original nao encontrado: {sentencepiece}; o conversor "
            "do llama.cpp precisa dele para o vocabulario Unigram"
        )
    base = Path(_base_do_adapter(str(adapter)))
    destino = fundir(adapter, base, Path(args.destino), torch.bfloat16)
    shutil.copyfile(sentencepiece, destino / "tokenizer.model")
    print(f"[fundir] {adapter} + {base} -> {destino}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
