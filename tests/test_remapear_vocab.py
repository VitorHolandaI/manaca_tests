from __future__ import annotations

import importlib.util

import pytest
from caixa_marcada import MAI, MAJ
from remapear_vocab import ids_de_maiusculas, mapa_por_peca, remapear_modelo

precisa_do_torch = pytest.mark.skipif(
    not importlib.util.find_spec("torch"), reason="precisa de torch"
)

ORIGEM = {"<unk>": 0, "▁a": 1, "A": 2, "▁Comentario": 3, "comentario": 4, "▁B": 5}
DESTINO = {"<unk>": 0, "▁a": 1, "comentario": 2, MAJ: 3, MAI: 4}


def test_mapa_por_peca_liga_o_id_novo_ao_velho_pela_string() -> None:
    assert mapa_por_peca(ORIGEM, DESTINO, (MAJ, MAI)) == {0: 0, 1: 1, 2: 4}


def test_mapa_por_peca_recusa_peca_que_a_origem_nao_tem() -> None:
    with pytest.raises(ValueError, match="xyz"):
        mapa_por_peca(ORIGEM, {**DESTINO, "xyz": 5}, (MAJ, MAI))


def test_ids_de_maiusculas_pega_letra_sozinha_com_e_sem_espaco() -> None:
    assert ids_de_maiusculas(ORIGEM) == [2, 5]


@precisa_do_torch
def test_remapear_copia_pela_peca_e_inicia_marcadores_pelas_maiusculas() -> None:
    from test_expandir_vocab import ModeloFalso

    modelo = ModeloFalso(len(ORIGEM))
    entrada_velha = modelo.entrada.weight.detach().clone()
    saida_velha = modelo.saida.weight.detach().clone()

    copiadas, iniciadas = remapear_modelo(modelo, ORIGEM, DESTINO, (MAJ, MAI))  # type: ignore[arg-type]

    assert (copiadas, iniciadas) == (3, 2)
    entrada, saida = modelo.entrada.weight, modelo.saida.weight
    assert entrada.shape[0] == len(DESTINO)
    assert entrada[2].tolist() == entrada_velha[4].tolist()
    assert saida[2].tolist() == saida_velha[4].tolist()
    media = entrada_velha[[2, 5]].mean(dim=0)
    assert entrada[3].tolist() == pytest.approx(media.tolist())
    assert entrada[4].tolist() == pytest.approx(media.tolist())
