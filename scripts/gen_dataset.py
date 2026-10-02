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
from dataclasses import replace
from pathlib import Path

from molde_fastapi import Campo, Entidade, montar_saida, nome_de_classe
from nomes_entidades import (
    NOMES_ENTIDADES,
    NOMES_EXTRAS,
    NOMES_TESTE_AMPLO,
    carregar_nomes_colhidos,
)

PREAMBULO = (
    "Abaixo esta uma instrucao que descreve uma tarefa. "
    "Escreva uma resposta que atenda adequadamente ao pedido."
)


# conjuntos de campos: o ciclo abaixo da o conjunto fixo de cada entidade, que
# vale no teste; no treino, gerar_exemplos sorteia um conjunto por exemplo
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

ENTIDADES: tuple[Entidade, ...] = tuple(
    Entidade(singular, plural, CONJUNTOS_CAMPOS[i % len(CONJUNTOS_CAMPOS)])
    for i, (singular, plural) in enumerate(NOMES_ENTIDADES)
)

# poucos exemplos por extra: o volume vem do numero de nomes, nao de repeticao
EXEMPLOS_POR_EXTRA = 6
# colhidos sao milhares (colher_nomes_classe.py): 2 cada segura o tamanho do SFT
EXEMPLOS_POR_COLHIDO = 2
ENTIDADES_EXTRAS: tuple[Entidade, ...] = tuple(
    Entidade(singular, plural, CONJUNTOS_CAMPOS[0]) for singular, plural in NOMES_EXTRAS
)

ENTIDADES_TESTE_AMPLO: tuple[Entidade, ...] = tuple(
    Entidade(singular, plural, CONJUNTOS_CAMPOS[i % len(CONJUNTOS_CAMPOS)])
    for i, (singular, plural) in enumerate(NOMES_TESTE_AMPLO)
)

# entidades inteiras fora do treino: variar so a frase deixaria o teste decorado
ENTIDADES_TESTE: tuple[str, ...] = ("comentario", "endereco")


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


def _sortear_campos(
    ent: Entidade, rng: random.Random, fixas: tuple[str, ...]
) -> Entidade:
    """Troca os campos da entidade por um conjunto sorteado, exceto nas fixas.

    Hipotese: com campo preso a entidade, o modelo trocou a entidade inedita
    por uma de treino (class Membro no pedido de comentario, 2026-10-01).
    """
    if ent.singular in fixas:
        return ent
    return replace(ent, campos=rng.choice(CONJUNTOS_CAMPOS))


def _exemplo(ent: Entidade, op: str, molde: str, comentar: bool) -> dict[str, str]:
    return {
        "instruction": capitalizar_frase(molde.format(s=ent.singular, p=ent.plural)),
        "input": "",
        "output": montar_saida(ent, (op,), comentar),
    }


def exemplos_extras(seed: int) -> list[dict[str, str]]:
    """EXEMPLOS_POR_EXTRA exemplos sorteados de cada entidade extra.

    Exemplo: len(exemplos_extras(1234)) == len(NOMES_EXTRAS) * EXEMPLOS_POR_EXTRA
    """
    return _poucos_exemplos(
        ENTIDADES_EXTRAS, EXEMPLOS_POR_EXTRA, random.Random(seed + 2)
    )


def exemplos_colhidos(
    seed: int, nomes: tuple[tuple[str, str], ...]
) -> list[dict[str, str]]:
    """EXEMPLOS_POR_COLHIDO exemplos de cada nome colhido; recusa nome ja usado.

    Exemplo: exemplos_colhidos(1234, (("fatia", "fatias"),))
    """
    usados = {e.singular for e in ENTIDADES + ENTIDADES_EXTRAS + ENTIDADES_TESTE_AMPLO}
    repetidos = sorted({s for s, _ in nomes} & (usados | set(ENTIDADES_TESTE)))
    if repetidos:
        raise ValueError(
            f"nomes colhidos ja usados pelo gen_dataset: {repetidos}, "
            "esperado so nome inedito (o de teste vazaria pro treino)"
        )
    entidades = tuple(Entidade(s, p, CONJUNTOS_CAMPOS[0]) for s, p in nomes)
    return _poucos_exemplos(entidades, EXEMPLOS_POR_COLHIDO, random.Random(seed + 3))


def _poucos_exemplos(
    entidades: tuple[Entidade, ...], por_entidade: int, rng: random.Random
) -> list[dict[str, str]]:
    combinacoes = [
        (op, molde, comentar)
        for op, _, _ in COMANDOS
        for molde in INSTRUCOES[op]
        for comentar in (True, False)
    ]
    exemplos: list[dict[str, str]] = []
    for ent_base in entidades:
        for op, molde, comentar in rng.sample(combinacoes, por_entidade):
            ent = _sortear_campos(ent_base, rng, ())
            exemplos.append(_exemplo(ent, op, molde, comentar))
    return exemplos


def exemplos_teste_amplo(seed: int) -> list[dict[str, str]]:
    """O split de teste original mais todos os exemplos de ENTIDADES_TESTE_AMPLO.

    Exemplo: exemplos_teste_amplo(1234) cobre 12 entidades ineditas no treino
    """
    _, teste = separar_teste(gerar_exemplos(seed), ENTIDADES_TESTE)
    novos = [
        _exemplo(ent, op, molde, comentar)
        for ent in ENTIDADES_TESTE_AMPLO
        for op, _, _ in COMANDOS
        for molde in INSTRUCOES[op]
        for comentar in (True, False)
    ]
    return teste + novos


def gerar_exemplos(
    seed: int,
    fixas: tuple[str, ...] = ENTIDADES_TESTE,
    colhidos: tuple[tuple[str, str], ...] = (),
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    # rng proprio dos campos: o do shuffle segue igual e o teste nao muda
    rng_campos = random.Random(seed + 1)
    exemplos: list[dict[str, str]] = []
    for ent_base in ENTIDADES:
        for op, _, _ in COMANDOS:
            for molde in INSTRUCOES[op]:
                for comentar in (True, False):
                    ent = _sortear_campos(ent_base, rng_campos, fixas)
                    exemplos.append(_exemplo(ent, op, molde, comentar))
    # exemplos multi-operacao: pede duas rotas de uma vez
    for ent_base in ENTIDADES:
        for comentar in (True, False):
            ops = ("criar", "listar")
            ent = _sortear_campos(ent_base, rng_campos, fixas)
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
    # extras no fim: o shuffle acima fica igual e o split de teste nao muda;
    # a ordem de treino quem embaralha e o Trainer
    return exemplos + exemplos_extras(seed) + exemplos_colhidos(seed, colhidos)


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
    ap.add_argument("--out-teste-amplo", default="data/fastapi_ptbr_teste_amplo.jsonl")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--nomes-colhidos", default="scripts/nomes_colhidos.tsv")
    args = ap.parse_args()

    tsv = Path(args.nomes_colhidos)
    colhidos = carregar_nomes_colhidos(tsv) if tsv.is_file() else ()
    if not colhidos:
        print(f"aviso: sem nomes colhidos em {tsv}; rode colher_nomes_classe.py")
    exemplos = gerar_exemplos(args.seed, colhidos=colhidos)
    treino, teste = separar_teste(exemplos, ENTIDADES_TESTE)
    destino, destino_teste = Path(args.out), Path(args.out_teste)
    info = escrever(treino, destino)
    info_teste = escrever(teste, destino_teste)
    destino_amplo = Path(args.out_teste_amplo)
    info_amplo = escrever(exemplos_teste_amplo(args.seed), destino_amplo)
    manifest = {
        "arquivo": str(destino),
        "exemplos": info["exemplos"],
        "sha256": info["sha256"],
        "arquivo_teste": str(destino_teste),
        "exemplos_teste": info_teste["exemplos"],
        "sha256_teste": info_teste["sha256"],
        "entidades_teste": list(ENTIDADES_TESTE),
        "arquivo_teste_amplo": str(destino_amplo),
        "exemplos_teste_amplo": info_amplo["exemplos"],
        "sha256_teste_amplo": info_amplo["sha256"],
        "entidades_teste_amplo": [e.singular for e in ENTIDADES_TESTE_AMPLO],
        "seed": args.seed,
        "licenca": "CC0-1.0 (gerado por scripts/gen_dataset.py, sem dado de terceiro)",
        "formato": "Alpaca {instruction, input, output}",
        "template": "Alpaca-PT (preambulo + ### Instrucao: / ### Resposta:)",
        "observacao": "caso real de Python; exige o tokenizer v2 (sem Lowercase)",
        "campos": "sorteados por exemplo no treino; fixos nas entidades de teste",
        "entidades_extras": len(ENTIDADES_EXTRAS),
        "nomes_colhidos": len(colhidos),
        "arquivo_nomes_colhidos": str(tsv) if colhidos else None,
    }
    (destino.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"treino: {info['exemplos']} exemplos em {destino} (sha256 {info['sha256']})")
    print(
        f"teste:  {info_teste['exemplos']} exemplos em {destino_teste} "
        f"(sha256 {info_teste['sha256']})"
    )
    print(
        f"amplo:  {info_amplo['exemplos']} exemplos em {destino_amplo} "
        f"(sha256 {info_amplo['sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
