from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    not importlib.util.find_spec("torch"), reason="precisa de torch"
)


def test_arredondar_poe_cada_bloco_nos_16_valores_do_nf4() -> None:
    import torch
    from nf4_simulado import NF4, arredondar_nf4

    torch.manual_seed(0)
    peso = torch.randn(8, 96, dtype=torch.bfloat16)  # 768 = 12 blocos de 64
    arredondado = arredondar_nf4(peso)
    assert arredondado.shape == peso.shape and arredondado.dtype == peso.dtype
    blocos = arredondado.float().reshape(-1, 64)
    absmax = peso.float().reshape(-1, 64).abs().amax(dim=1, keepdim=True)
    normal = blocos / absmax
    distancia = (normal.unsqueeze(-1) - torch.tensor(NF4)).abs().amin(dim=-1)
    assert distancia.max().item() < 1e-2  # bf16 arredonda o produto


def test_arredondar_e_idempotente_e_aceita_tamanho_fora_do_bloco() -> None:
    import torch
    from nf4_simulado import arredondar_nf4

    peso = torch.randn(3, 50)  # 150 nao e multiplo de 64
    uma = arredondar_nf4(peso)
    assert torch.allclose(arredondar_nf4(uma), uma)


def test_simular_nf4_pula_o_lm_head() -> None:
    import torch
    from nf4_simulado import simular_nf4

    class ModeloLinearFalso(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(64, 64, bias=False)
            self.lm_head = torch.nn.Linear(64, 10, bias=False)

    modelo = ModeloLinearFalso()
    cabeca = modelo.lm_head.weight.detach().clone()
    proj = modelo.proj.weight.detach().clone()
    assert simular_nf4(modelo) == 1
    assert torch.equal(modelo.lm_head.weight, cabeca)
    assert not torch.equal(modelo.proj.weight, proj)
