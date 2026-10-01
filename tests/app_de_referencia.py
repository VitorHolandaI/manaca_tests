"""Executa uma saida de referencia como app FastAPI real, sem restauro.

ast.parse so prova sintaxe; isto prova que o codigo que o modelo decora
importa, cria as tabelas sozinho e responde o CRUD. Roda o texto cru: desde o
dataset cased, a referencia nao pode depender de restore_case.py.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from gen_dataset import Entidade

VALOR_POR_TIPO: dict[str, str | int | float | bool] = {
    "str": "x",
    "int": 1,
    "float": 1.5,
    "bool": True,
}


class AppDeReferencia:
    """Sobe o app gerado; use depois de chdir para um diretorio temporario."""

    def __init__(self, codigo: str, entidade: Entidade) -> None:
        self.entidade = entidade
        # __name__ e obrigatorio: sem ele as classes nascem com __module__ ==
        # 'builtins' e o pydantic ignora o from_attributes do modelo de resposta
        escopo: dict[str, object] = {"__name__": "referencia_manaca"}
        # exec e o ponto do teste: rodar o codigo que o modelo vai decorar
        exec(compile(codigo, "<referencia>", "exec"), escopo)  # noqa: S102
        self.cliente = TestClient(escopo["app"])  # type: ignore[arg-type]

    def payload(self) -> dict[str, str | int | float | bool]:
        return {c.nome: VALOR_POR_TIPO[c.tipo] for c in self.entidade.campos}
