"""Molde do arquivo FastAPI + SQLAlchemy que o gen_dataset usa como resposta.

Separado do gen_dataset para que a montagem do dataset (entidades, frases,
sorteios, splits) e o texto do codigo gerado mudem por motivos diferentes.
Quem le: gen_dataset.py, e os testes via tests/app_de_referencia.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Campo:
    nome: str
    tipo: str
    nulo: bool = False


@dataclass(frozen=True)
class Entidade:
    singular: str
    plural: str
    campos: tuple[Campo, ...]


TIPOS_SQL = {"str": "String", "int": "Integer", "float": "Numeric", "bool": "Boolean"}


def nome_de_classe(nome: str) -> str:
    """Converte o nome da entidade na convencao de classe Python.

    Exemplo: nome_de_classe('endereco') == 'Endereco'
    """
    return nome[:1].upper() + nome[1:]


def _cabecalho(ent: Entidade, comentar: bool) -> str:
    linhas = [
        "from fastapi import FastAPI, HTTPException, Depends",
        "from sqlalchemy import create_engine, Column, Integer, String, Boolean, Numeric",
        "from sqlalchemy.orm import declarative_base, sessionmaker, Session",
        "from pydantic import BaseModel",
        "",
    ]
    if comentar:
        linhas.append("# Banco de dados SQLite em arquivo local")
    linhas += [
        'engine = create_engine("sqlite:///./app.db", connect_args={"check_same_thread": False})',
        "sessao_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)",
        "Base = declarative_base()",
        "",
    ]
    return "\n".join(linhas)


def _modelo(ent: Entidade, comentar: bool) -> str:
    linhas = [f"class {nome_de_classe(ent.singular)}(Base):"]
    if comentar:
        linhas.append(f"    # Tabela de {ent.plural} no banco")
    linhas += [
        f'    __tablename__ = "{ent.plural}"',
        "    id = Column(Integer, primary_key=True, index=True)",
    ]
    for campo in ent.campos:
        tipo = TIPOS_SQL[campo.tipo]
        nulo = "True" if campo.nulo else "False"
        linhas.append(f"    {campo.nome} = Column({tipo}, nullable={nulo})")
    linhas += ["", "Base.metadata.create_all(bind=engine)", ""]
    return "\n".join(linhas)


def _sessao(comentar: bool) -> str:
    linhas = ["def obter_sessao():"]
    if comentar:
        linhas.append("    # Abre a sessao e garante o close no fim")
    linhas += [
        "    sessao = sessao_local()",
        "    try:",
        "        yield sessao",
        "    finally:",
        "        sessao.close()",
        "",
    ]
    return "\n".join(linhas)


def _schemas(ent: Entidade, comentar: bool) -> str:
    classe = nome_de_classe(ent.singular)
    entrada = [f"class {classe}Criar(BaseModel):"]
    saida = [f"class {classe}Resposta(BaseModel):", "    id: int"]
    for campo in ent.campos:
        entrada.append(f"    {campo.nome}: {campo.tipo}")
        saida.append(f"    {campo.nome}: {campo.tipo}")
    saida += ["", '    model_config = {"from_attributes": True}']
    bloco = []
    if comentar:
        bloco.append(f"# Schema de entrada para criar {ent.singular}")
    bloco += entrada + [""]
    if comentar:
        bloco.append("# Schema de saida com o id gerado")
    bloco += saida + [""]
    return "\n".join(bloco)


def _app() -> str:
    return "app = FastAPI()\n"


def _rota(ent: Entidade, op: str, comentar: bool) -> str:
    s, p = ent.singular, ent.plural
    classe = nome_de_classe(s)
    if op == "criar":
        corpo = [
            f'@app.post("/{p}", response_model={classe}Resposta, status_code=201)',
            f"def criar_{s}(dados: {classe}Criar, sessao: Session = Depends(obter_sessao)):",
            f'    """Cadastra um novo {s} e devolve o registro com o id gerado."""',
            "    # Grava o novo registro e devolve ele com o id",
            f"    registro = {classe}(**dados.model_dump())",
            "    sessao.add(registro)",
            "    sessao.commit()",
            "    sessao.refresh(registro)",
            "    return registro",
        ]
    elif op == "listar":
        corpo = [
            f'@app.get("/{p}", response_model=list[{classe}Resposta])',
            f"def listar_{p}(sessao: Session = Depends(obter_sessao)):",
            f'    """Devolve todos os {p} cadastrados no banco."""',
            "    # Devolve todos os registros da tabela",
            f"    return sessao.query({classe}).all()",
        ]
    elif op == "buscar":
        corpo = [
            f'@app.get("/{p}/{{item_id}}", response_model={classe}Resposta)',
            f"def buscar_{s}(item_id: int, sessao: Session = Depends(obter_sessao)):",
            f'    """Busca um {s} pelo id e devolve 404 se nao existir."""',
            f"    registro = sessao.query({classe}).filter({classe}.id == item_id).first()",
            "    if registro is None:",
            f'        raise HTTPException(status_code=404, detail="{classe} nao encontrado")',
            "    return registro",
        ]
    elif op == "atualizar":
        corpo = [
            f'@app.put("/{p}/{{item_id}}", response_model={classe}Resposta)',
            f"def atualizar_{s}(item_id: int, dados: {classe}Criar,",
            "                  sessao: Session = Depends(obter_sessao)):",
            f'    """Atualiza um {s} pelo id e devolve 404 se nao existir."""',
            f"    registro = sessao.query({classe}).filter({classe}.id == item_id).first()",
            "    if registro is None:",
            f'        raise HTTPException(status_code=404, detail="{classe} nao encontrado")',
            "    # Copia os campos do corpo para o registro",
            "    for chave, valor in dados.model_dump().items():",
            "        setattr(registro, chave, valor)",
            "    sessao.commit()",
            "    sessao.refresh(registro)",
            "    return registro",
        ]
    else:
        corpo = [
            f'@app.delete("/{p}/{{item_id}}")',
            f"def deletar_{s}(item_id: int, sessao: Session = Depends(obter_sessao)):",
            f'    """Remove um {s} pelo id e devolve 404 se nao existir."""',
            f"    registro = sessao.query({classe}).filter({classe}.id == item_id).first()",
            "    if registro is None:",
            f'        raise HTTPException(status_code=404, detail="{classe} nao encontrado")',
            "    sessao.delete(registro)",
            "    sessao.commit()",
            f'    return {{"detalhe": "{classe} removido com sucesso"}}',
        ]
    return "\n".join(corpo) + "\n"


def montar_saida(ent: Entidade, ops: tuple[str, ...], comentar: bool) -> str:
    partes = [
        _cabecalho(ent, comentar),
        _modelo(ent, comentar),
        _sessao(comentar),
        _schemas(ent, comentar),
        _app(),
    ]
    for op in ops:
        partes.append(_rota(ent, op, comentar))
    return "\n".join(partes).strip() + "\n"
