#!/usr/bin/env python3
"""Gera um dataset Alpaca-PT de FastAPI + SQLAlchemy em portugues, COM caso.

O alvo sai com a capitalizacao real de Python: `class Usuario(Base)`,
`FastAPI`, `HTTPException`, `True`/`None`, docstring em portugues capitalizada.
O tokenizer v2 (scripts/build_tokenizer_v2.py) abre o canal de maiuscula, entao
o treino pode ensinar caso em vez de delegar isso ao pos-processamento.

Saida: dois JSONL no esquema Alpaca {instruction, input, output} -- treino e
teste, este com entidades que o treino nunca viu -- e um manifest com
contagens, hashes e licenca.

Uso:
    python scripts/gen_dataset.py --out data/fastapi_ptbr.jsonl --seed 1234
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

PREAMBULO = (
    "Abaixo esta uma instrucao que descreve uma tarefa. "
    "Escreva uma resposta que atenda adequadamente ao pedido."
)


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


# conjuntos de campos, ciclados entre as entidades: variar so o nome da entidade
# deixaria o modelo decorar o formato dos campos em vez de ler a instrucao
CONJUNTOS_CAMPOS: tuple[tuple[Campo, ...], ...] = (
    (Campo("nome", "str"), Campo("email", "str"), Campo("ativo", "bool")),
    (Campo("nome", "str"), Campo("preco", "float"), Campo("estoque", "int")),
    (Campo("cliente", "str"), Campo("total", "float"), Campo("pago", "bool")),
    (Campo("titulo", "str"), Campo("concluida", "bool")),
    (Campo("nome", "str"), Campo("telefone", "str")),
    (Campo("nome", "str"), Campo("descricao", "str")),
    (Campo("autor", "str"), Campo("texto", "str")),
    (Campo("rua", "str"), Campo("cidade", "str"), Campo("cep", "str")),
    (Campo("codigo", "str"), Campo("valor", "float")),
    (Campo("nome", "str"), Campo("quantidade", "int"), Campo("ativo", "bool")),
    (Campo("referencia", "str"), Campo("vencimento", "str")),
    (Campo("nome", "str"), Campo("peso", "float"), Campo("altura", "float")),
    (Campo("titulo", "str"), Campo("publicado", "bool")),
    (Campo("nome", "str"), Campo("prioridade", "int"), Campo("concluido", "bool")),
)

# as 8 primeiras sao as originais: os campos de comentario e endereco nao mudam,
# entao o split de teste continua identico ao que ja foi medido
NOMES_ENTIDADES: tuple[tuple[str, str], ...] = (
    ("usuario", "usuarios"),
    ("produto", "produtos"),
    ("pedido", "pedidos"),
    ("tarefa", "tarefas"),
    ("cliente", "clientes"),
    ("categoria", "categorias"),
    ("comentario", "comentarios"),
    ("endereco", "enderecos"),
    ("documento", "documentos"),
    ("contrato", "contratos"),
    ("fatura", "faturas"),
    ("fornecedor", "fornecedores"),
    ("funcionario", "funcionarios"),
    ("evento", "eventos"),
    ("ingresso", "ingressos"),
    ("transacao", "transacoes"),
    ("servico", "servicos"),
    ("projeto", "projetos"),
    ("equipe", "equipes"),
    ("membro", "membros"),
    ("grupo", "grupos"),
    ("mensagem", "mensagens"),
    ("item", "itens"),
    ("orcamento", "orcamentos"),
    ("pagamento", "pagamentos"),
    ("assinatura", "assinaturas"),
    ("campanha", "campanhas"),
    ("anuncio", "anuncios"),
    ("etiqueta", "etiquetas"),
    ("estoque", "estoques"),
    ("lote", "lotes"),
    ("transportadora", "transportadoras"),
    ("entrega", "entregas"),
    ("avaliacao", "avaliacoes"),
    ("curtida", "curtidas"),
    ("seguidor", "seguidores"),
    ("perfil", "perfis"),
    ("conta", "contas"),
    ("cartao", "cartoes"),
    ("boleto", "boletos"),
    ("nota", "notas"),
    ("disciplina", "disciplinas"),
    ("turma", "turmas"),
    ("aluno", "alunos"),
    ("professor", "professores"),
    ("curso", "cursos"),
    ("matricula", "matriculas"),
    ("aula", "aulas"),
    ("biblioteca", "bibliotecas"),
    ("livro", "livros"),
    ("autor", "autores"),
    ("editora", "editoras"),
    ("emprestimo", "emprestimos"),
    ("reserva", "reservas"),
    ("quarto", "quartos"),
    ("hospede", "hospedes"),
    ("voo", "voos"),
    ("passageiro", "passageiros"),
    ("bilhete", "bilhetes"),
    ("consulta", "consultas"),
    ("paciente", "pacientes"),
    ("medico", "medicos"),
    ("exame", "exames"),
    ("receita", "receitas"),
    ("medicamento", "medicamentos"),
    ("leitura", "leituras"),
    ("sensor", "sensores"),
    ("dispositivo", "dispositivos"),
    ("alerta", "alertas"),
    ("incidente", "incidentes"),
    ("chamado", "chamados"),
    ("arquivo", "arquivos"),
    ("pasta", "pastas"),
    ("sprint", "sprints"),
    ("repositorio", "repositorios"),
)

ENTIDADES: tuple[Entidade, ...] = tuple(
    Entidade(singular, plural, CONJUNTOS_CAMPOS[i % len(CONJUNTOS_CAMPOS)])
    for i, (singular, plural) in enumerate(NOMES_ENTIDADES)
)

# entidades inteiras fora do treino: variar so a frase deixaria o teste decorado
ENTIDADES_TESTE: tuple[str, ...] = ("comentario", "endereco")

TIPOS_SQL = {"str": "String", "int": "Integer", "float": "Numeric", "bool": "Boolean"}


def nome_de_classe(nome: str) -> str:
    """Converte o nome da entidade na convencao de classe Python.

    Exemplo: nome_de_classe('endereco') == 'Endereco'
    """
    return nome[:1].upper() + nome[1:]


def capitalizar_frase(texto: str) -> str:
    """Capitaliza a primeira letra e os nomes de biblioteca citados na prosa.

    Exemplo: capitalizar_frase('crie uma rota no fastapi') ->
    'Crie uma rota no FastAPI'
    """
    for baixo, correto in (
        ("fastapi", "FastAPI"),
        ("sqlalchemy", "SQLAlchemy"),
        ("pydantic", "Pydantic"),
    ):
        texto = texto.replace(baixo, correto)
    return texto[:1].upper() + texto[1:]


COMANDOS = (
    ("criar", "post", 201),
    ("listar", "get", 200),
    ("buscar", "get", 200),
    ("atualizar", "put", 200),
    ("deletar", "delete", 200),
)

INSTRUCOES = {
    "criar": (
        "crie um endpoint para cadastrar um {s} no fastapi usando sqlalchemy",
        "como faco para criar um novo {s} com fastapi e sqlalchemy?",
        "escreva a rota de criacao de {s} em fastapi com o schema do pydantic",
    ),
    "listar": (
        "crie um endpoint para listar todos os {p} no fastapi com sqlalchemy",
        "como listar todos os {p} cadastrados usando fastapi?",
        "escreva a rota que devolve a lista de {p} do banco",
    ),
    "buscar": (
        "crie um endpoint que busca um {s} pelo id no fastapi",
        "como retornar 404 quando o {s} nao existe no fastapi?",
        "escreva a rota de consulta de {s} por id com sqlalchemy",
    ),
    "atualizar": (
        "crie um endpoint para atualizar um {s} pelo id no fastapi",
        "como atualizar os dados de um {s} com fastapi e sqlalchemy?",
        "escreva a rota de atualizacao de {s} tratando o caso de id inexistente",
    ),
    "deletar": (
        "crie um endpoint de exclusao (delete) para {s} no fastapi com sqlalchemy",
        "como deletar um {s} pelo id no fastapi?",
        "escreva a rota que remove um {s} do banco e retorna 404 se nao achar",
    ),
}


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


def gerar_exemplos(seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    exemplos: list[dict[str, str]] = []
    for ent in ENTIDADES:
        for op, _, _ in COMANDOS:
            for molde in INSTRUCOES[op]:
                for comentar in (True, False):
                    ops: tuple[str, ...] = (op,)
                    exemplos.append(
                        {
                            "instruction": capitalizar_frase(
                                molde.format(s=ent.singular, p=ent.plural)
                            ),
                            "input": "",
                            "output": montar_saida(ent, ops, comentar),
                        }
                    )
    # exemplos multi-operacao: pede duas rotas de uma vez
    for ent in ENTIDADES:
        for comentar in (True, False):
            ops = ("criar", "listar")
            exemplos.append(
                {
                    "instruction": capitalizar_frase(
                        f"crie as rotas de criar e listar {ent.plural} no fastapi"
                    ),
                    "input": "",
                    "output": montar_saida(ent, ops, comentar),
                }
            )
    rng.shuffle(exemplos)
    return exemplos


def separar_teste(
    exemplos: list[dict[str, str]],
    entidades_teste: tuple[str, ...],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Divide em (treino, teste) pela classe do modelo SQLAlchemy da saida.

    Exemplo: separar_teste(gerar_exemplos(1234), ("endereco",))
    """
    marcas = tuple(f"class {nome_de_classe(n)}(Base):" for n in entidades_teste)
    treino = [ex for ex in exemplos if not any(m in ex["output"] for m in marcas)]
    teste = [ex for ex in exemplos if any(m in ex["output"] for m in marcas)]
    return treino, teste


def escrever(exemplos: list[dict[str, str]], destino: Path) -> dict[str, object]:
    destino.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with destino.open("w", encoding="utf-8") as fh:
        for ex in exemplos:
            linha = json.dumps(ex, ensure_ascii=False)
            digest.update(linha.encode("utf-8") + b"\n")
            fh.write(linha + "\n")
    return {"exemplos": len(exemplos), "sha256": digest.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera dataset FastAPI PT-BR (cased)")
    ap.add_argument("--out", default="data/fastapi_ptbr.jsonl", help="split de treino")
    ap.add_argument("--out-teste", default="data/fastapi_ptbr_teste.jsonl")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    treino, teste = separar_teste(gerar_exemplos(args.seed), ENTIDADES_TESTE)
    destino, destino_teste = Path(args.out), Path(args.out_teste)
    info = escrever(treino, destino)
    info_teste = escrever(teste, destino_teste)
    manifest = {
        "arquivo": str(destino),
        "exemplos": info["exemplos"],
        "sha256": info["sha256"],
        "arquivo_teste": str(destino_teste),
        "exemplos_teste": info_teste["exemplos"],
        "sha256_teste": info_teste["sha256"],
        "entidades_teste": list(ENTIDADES_TESTE),
        "seed": args.seed,
        "licenca": "CC0-1.0 (gerado por scripts/gen_dataset.py, sem dado de terceiro)",
        "formato": "Alpaca {instruction, input, output}",
        "template": "Alpaca-PT (preambulo + ### Instrucao: / ### Resposta:)",
        "observacao": "caso real de Python; exige o tokenizer v2 (sem Lowercase)",
    }
    (destino.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"treino: {info['exemplos']} exemplos em {destino} (sha256 {info['sha256']})")
    print(
        f"teste:  {info_teste['exemplos']} exemplos em {destino_teste} "
        f"(sha256 {info_teste['sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
