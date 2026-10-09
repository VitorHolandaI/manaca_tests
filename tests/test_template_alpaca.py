from __future__ import annotations

from pathlib import Path

import pytest
from template_alpaca import (
    LEGADO,
    LNCC,
    TEMPLATES,
    ler_template,
    montar_prompt,
    salvar_template,
)


def test_legado_e_o_template_dos_adapters_ja_treinados() -> None:
    prompt = montar_prompt(LEGADO, "Crie a rota")
    assert prompt == (
        "Abaixo esta uma instrucao que descreve uma tarefa. "
        "Escreva uma resposta que atenda adequadamente ao pedido."
        "\n\n### Instrucao:\nCrie a rota\n\n### Resposta:\n"
    )


def test_lncc_e_igual_ao_sft_train_py_do_manaca_instruct() -> None:
    sem_entrada = montar_prompt(LNCC, "Oi")
    assert sem_entrada.startswith("Abaixo está uma instrução que descreve uma tarefa. ")
    assert sem_entrada.endswith("\n\n### Instrução:\nOi\n\n### Resposta:\n")
    com_entrada = montar_prompt(LNCC, "Resuma.", "Texto longo")
    assert "combinada com uma entrada que fornece mais contexto." in com_entrada
    assert com_entrada.endswith(
        "### Instrução:\nResuma.\n\n### Entrada:\nTexto longo\n\n### Resposta:\n"
    )


def test_instrucao_e_entrada_sao_aparadas() -> None:
    assert montar_prompt(LNCC, "  Oi \n", "  ") == montar_prompt(LNCC, "Oi")


def test_salvar_e_ler_o_template_da_pasta(tmp_path: Path) -> None:
    assert ler_template(tmp_path) is LEGADO  # adapter antigo, sem arquivo
    salvar_template(tmp_path, LNCC)
    assert ler_template(tmp_path) is LNCC


def test_template_desconhecido_falha_com_o_nome(tmp_path: Path) -> None:
    (tmp_path / "template.json").write_text('{"template": "xyz"}', encoding="utf-8")
    with pytest.raises(ValueError, match="xyz"):
        ler_template(tmp_path)
    assert set(TEMPLATES) == {"legado", "lncc"}
