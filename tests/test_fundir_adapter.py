from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft")),
    reason="precisa de torch, transformers e peft",
)


def _llama_minusculo(pasta: Path) -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).save_pretrained(pasta / "base")
    return pasta / "base"


def _adapter(base: Path, pasta: Path) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    modelo = AutoModelForCausalLM.from_pretrained(base)
    lora = get_peft_model(
        modelo,
        LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False),
    )
    with torch.no_grad():
        for nome, p in lora.named_parameters():
            if "lora_" in nome:
                p.normal_(0, 0.1)
    lora.save_pretrained(pasta / "adapter")
    return pasta / "adapter"


def test_fundido_da_a_mesma_saida_que_adapter_sobre_o_base_em_nf4(
    tmp_path: Path,
) -> None:
    import torch
    from fundir_adapter import carregar_com_nf4, fundir
    from transformers import AutoModelForCausalLM

    base = _llama_minusculo(tmp_path)
    adapter = _adapter(base, tmp_path)
    destino = fundir(adapter, base, tmp_path / "fundido", dtype=torch.float32)

    referencia = carregar_com_nf4(adapter, base, dtype=torch.float32)
    fundido = AutoModelForCausalLM.from_pretrained(destino, torch_dtype=torch.float32)
    ids = torch.tensor([[1, 5, 9, 3]])
    with torch.no_grad():
        esperado = referencia(ids).logits
        obtido = fundido(ids).logits
    assert torch.allclose(esperado, obtido, atol=1e-4)
    assert not (destino / "adapter_config.json").exists()
