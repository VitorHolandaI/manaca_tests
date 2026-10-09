"""Templates Alpaca-PT do SFT: o legado deste projeto e o do Manaca instruct.

O legado (sem acento, sem campo de entrada) treinou todos os adapters ate o
v4. O `lncc` copia o `sft/train.py` do Manaca-1B-Instruct, com acento e com
`### Entrada:` para traducao e resumo; e o do SFT que mistura conversa e
FastAPI. O nome do template fica gravado no adapter (`template.json`), para
o eval e os chats montarem o prompt igual ao do treino.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

ARQUIVO = "template.json"


class Template(NamedTuple):
    nome: str
    preambulo: str
    preambulo_com_entrada: str
    instrucao: str
    entrada: str
    resposta: str


LEGADO = Template(
    nome="legado",
    preambulo=(
        "Abaixo esta uma instrucao que descreve uma tarefa. "
        "Escreva uma resposta que atenda adequadamente ao pedido."
    ),
    preambulo_com_entrada=(
        "Abaixo esta uma instrucao que descreve uma tarefa, combinada com uma "
        "entrada que fornece mais contexto. Escreva uma resposta que atenda "
        "adequadamente ao pedido."
    ),
    instrucao="\n\n### Instrucao:\n",
    entrada="\n\n### Entrada:\n",
    resposta="\n\n### Resposta:\n",
)

# Instituto-IA-LNCC/manaca-1b-instruct, sft/train.py (PREAMBLE_*, *_TEMPLATE)
LNCC = Template(
    nome="lncc",
    preambulo=(
        "Abaixo está uma instrução que descreve uma tarefa. "
        "Escreva uma resposta que atenda adequadamente ao pedido."
    ),
    preambulo_com_entrada=(
        "Abaixo está uma instrução que descreve uma tarefa, combinada com uma "
        "entrada que fornece mais contexto. Escreva uma resposta que atenda "
        "adequadamente ao pedido."
    ),
    instrucao="\n\n### Instrução:\n",
    entrada="\n\n### Entrada:\n",
    resposta="\n\n### Resposta:\n",
)

TEMPLATES = {t.nome: t for t in (LEGADO, LNCC)}


def montar_prompt(template: Template, instrucao: str, entrada: str = "") -> str:
    """Prompt ate o marcador de resposta, inclusive.

    Exemplo: montar_prompt(LNCC, "Resuma.", "texto") termina em "### Resposta:\\n"
    """
    instrucao, entrada = instrucao.strip(), entrada.strip()
    if entrada:
        return (
            f"{template.preambulo_com_entrada}{template.instrucao}{instrucao}"
            f"{template.entrada}{entrada}{template.resposta}"
        )
    return f"{template.preambulo}{template.instrucao}{instrucao}{template.resposta}"


def salvar_template(pasta: Path, template: Template) -> None:
    """Grava o nome do template na pasta do adapter."""
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / ARQUIVO).write_text(
        json.dumps({"template": template.nome}), encoding="utf-8"
    )


def ler_template(pasta: Path) -> Template:
    """Template gravado na pasta; sem o arquivo, o legado (adapters antigos)."""
    arquivo = pasta / ARQUIVO
    if not arquivo.is_file():
        return LEGADO
    nome = json.loads(arquivo.read_text(encoding="utf-8"))["template"]
    if nome not in TEMPLATES:
        raise ValueError(
            f"template desconhecido em {arquivo}: {nome!r}, esperado um de {sorted(TEMPLATES)}"
        )
    return TEMPLATES[nome]
