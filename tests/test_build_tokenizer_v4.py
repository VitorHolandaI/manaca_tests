from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from build_tokenizer_v4 import construir
from caixa_marcada import COLA, MAI, MAJ, codificar

RAIZ = Path(__file__).resolve().parent.parent
ORIGINAL_JSON = RAIZ / "models" / "manaca-hf-official" / "tokenizer.json"

pytestmark = pytest.mark.skipif(
    not (ORIGINAL_JSON.is_file() and importlib.util.find_spec("tokenizers")),
    reason="precisa do tokenizer original e da lib tokenizers",
)


def test_v4_e_o_original_com_os_tres_marcadores_no_fim() -> None:
    tk = construir(ORIGINAL_JSON)
    assert [tk.token_to_id(m) for m in (MAJ, MAI, COLA)] == [64000, 64001, 64002]
    assert tk.get_vocab_size() == 64003


def test_palavra_colada_usa_a_mesma_peca_da_instrucao() -> None:
    # regressao: no v3 `criar_agendamento` usava a peca nua e saia `criar_agenda`
    tk = construir(ORIGINAL_JSON)
    instrucao = tk.encode(codificar("um agendamento", True), add_special_tokens=False)
    funcao = tk.encode(
        codificar("def criar_agendamento(", True), add_special_tokens=False
    )
    classe = tk.encode(codificar("query(Agendamento)", True), add_special_tokens=False)
    peca = instrucao.ids[-1]
    assert peca in funcao.ids and peca in classe.ids


def test_v4_devolve_o_texto_codificado_do_dataset() -> None:
    from gen_dataset import gerar_exemplos

    tk = construir(ORIGINAL_JSON)
    for exemplo in gerar_exemplos(1234)[:200]:
        texto = codificar(exemplo["output"], True)
        ids = tk.encode(texto, add_special_tokens=False).ids
        assert tk.decode(ids, skip_special_tokens=False) == texto
