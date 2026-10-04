"""Leva um checkpoint de um vocabulario para outro copiando linha por peca.

O CPT grande foi treinado com o tokenizer v2 (115148 tokens); o v3 tem as
mesmas pecas originais e nuas, em outros ids, sem as pecas com maiuscula e com
2 marcadores. Copiar pela string aproveita o que o CPT ja aprendeu. Os
marcadores de caixa comecam pela media das letras maiusculas do v2 (`A`, `▁B`...),
que no CPT grande ja carregam maiuscula; outro token novo (`<|cola|>`, v4) pela
media geral. Marcador que a origem ja tem (v3 -> v4) e copiado como qualquer peca.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from caixa_marcada import MAI, MAJ

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel

ESPACO_SP = "▁"


def mapa_por_peca(
    origem: dict[str, int], destino: dict[str, int], novos: tuple[str, ...]
) -> dict[int, int]:
    """{id no destino: id na origem} para toda peca do destino fora as novas.

    Exemplo: mapa_por_peca({"▁a": 7}, {"▁a": 1, "<|maj|>": 2}, ("<|maj|>",)) == {1: 7}
    """
    faltando = [p for p in destino if p not in origem and p not in novos]
    if faltando:
        raise ValueError(
            f"{len(faltando)} pecas do destino nao existem na origem "
            f"(ex.: {faltando[:5]}); esperado um tokenizer construido sobre o da "
            "origem, ou essas pecas declaradas como novas"
        )
    return {i: origem[p] for p, i in destino.items() if p not in novos}


def ids_de_maiusculas(origem: dict[str, int]) -> list[int]:
    """Ids das pecas que sao uma letra maiuscula sozinha, com ou sem `▁`."""
    return sorted(
        i
        for p, i in origem.items()
        if len(p.removeprefix(ESPACO_SP)) == 1 and p.removeprefix(ESPACO_SP).isupper()
    )


def _preencher(
    nova: torch.Tensor,
    velha: torch.Tensor,
    mapa: dict[int, int],
    iniciais: dict[int, list[int]],
) -> None:
    """Copia pelo mapa e poe em cada id novo a media das linhas indicadas."""
    import torch

    with torch.no_grad():
        destino_ids = torch.tensor(list(mapa.keys()))
        origem_ids = torch.tensor(list(mapa.values()))
        nova[destino_ids] = velha[origem_ids].to(nova.dtype)
        for novo_id, fonte in iniciais.items():
            linhas = velha[fonte] if fonte else velha
            nova[novo_id] = linhas.float().mean(dim=0).to(nova.dtype)


def remapear_modelo(
    modelo: PreTrainedModel,
    origem: dict[str, int],
    destino: dict[str, int],
    novos: tuple[str, ...],
) -> tuple[int, int]:
    """Redimensiona e copia entrada e saida pela peca; (copiadas, iniciadas).

    Exemplo: remapear_modelo(modelo_v2, vocab_v2, vocab_v3, (MAJ, MAI))
    """
    mapa = mapa_por_peca(origem, destino, novos)
    maiusculas = ids_de_maiusculas(origem)
    # lista vazia = media de todas as linhas da origem
    iniciais = {destino[n]: maiusculas if n in (MAJ, MAI) else [] for n in novos}
    saida = modelo.get_output_embeddings()
    amarrada = saida is None or bool(modelo.config.tie_word_embeddings)
    velha_entrada = modelo.get_input_embeddings().weight.detach().clone()
    velha_saida = None if amarrada else saida.weight.detach().clone()
    modelo.resize_token_embeddings(len(destino), mean_resizing=False)
    _preencher(modelo.get_input_embeddings().weight, velha_entrada, mapa, iniciais)
    nova_saida = modelo.get_output_embeddings()
    if velha_saida is not None and nova_saida is not None:
        _preencher(nova_saida.weight, velha_saida, mapa, iniciais)
    return len(mapa), len(iniciais)
