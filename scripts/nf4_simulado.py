"""Arredonda os pesos para a grade NF4 sem bitsandbytes (sem GPU NVIDIA).

O adapter do SFT foi treinado sobre o base quantizado em NF4 (QLoRA). Sobre o
base em bf16, o v4 trocou `connect_args={` por `connect_args=Depends` em todos
os pedidos (2026-10-06). Aqui cada nn.Linear, menos o lm_head, vai para a
grade NF4 em blocos de 64 e volta para o dtype original: o adapter ve os
mesmos pesos do treino. Valores e bloco de bitsandbytes 0.45 functional.py
(get_4bit_type "nf4", quantize_4bit blocksize=64). A dupla quantizacao (o
absmax em 8 bits) nao e simulada.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NF4 = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
BLOCO = 64
# o transformers quantiza todo nn.Linear menos este (modules_to_not_convert)
FORA_DO_NF4 = "lm_head"


def arredondar_nf4(peso: torch.Tensor, bloco: int = BLOCO) -> torch.Tensor:
    """Peso com cada bloco de `bloco` valores levado ao NF4 mais proximo.

    Exemplo: arredondar_nf4(torch.randn(4, 64)).shape == (4, 64)
    """
    import torch

    plano = peso.float().reshape(-1)
    falta = (-plano.numel()) % bloco
    blocos = torch.nn.functional.pad(plano, (0, falta)).view(-1, bloco)
    absmax = blocos.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    grade = torch.tensor(NF4, device=peso.device)
    # bucketize nos pontos medios = valor mais proximo, sem matriz blocos x 16
    indices = torch.bucketize(blocos / absmax, (grade[1:] + grade[:-1]) / 2)
    voltou = (grade[indices] * absmax).reshape(-1)[: plano.numel()]
    return voltou.view(peso.shape).to(peso.dtype)


def simular_nf4(modelo: torch.nn.Module) -> int:
    """Arredonda in-place todo nn.Linear menos o lm_head; devolve quantos."""
    import torch

    trocados = 0
    with torch.no_grad():
        for nome, modulo in modelo.named_modules():
            if not isinstance(modulo, torch.nn.Linear):
                continue
            if nome.split(".")[-1] == FORA_DO_NF4:
                continue
            modulo.weight.copy_(arredondar_nf4(modulo.weight))
            trocados += 1
    return trocados
