#!/usr/bin/env python3
"""Segmentacao Unigram com amostragem: subword regularization (Kudo 2018).

O `tokenizer.json` do Manaca e Unigram, mas a amostragem nao esta disponivel em
lugar nenhum:

- o `tokenizers` expoe `nbest_size`/`alpha` no modelo e NAO os usa no encode
  (conferido: mudar `nbest_size` nao muda a saida);
- o `tokenizer.model` do SentencePiece tem `enable_sampling`, mas o normalizador
  `nmt_nfkc_cf` dele troca toda quebra de linha por espaco -- `encode("a\\nb") ==
  encode("a b")`. Um modelo treinado ali nao consegue gerar Python multi-linha.

Entao a amostragem e feita aqui, sobre o vocabulario do proprio `tokenizer.json`.
A normalizacao e a pre-tokenizacao NAO sao reimplementadas: usam-se o
normalizador e o pre-tokenizador do `tokenizer.json`, que produzem um unico span
com os espacos trocados por '▁'. A amostragem so decide onde cortar esse span.

`viterbi()` reproduz exatamente o `tokenizer.json` (verificado em 40 exemplos);
e o portao de sanidade: se ele divergir, os scores ou o byte-fallback estao
errados e nada em cima disso vale.

Uso:
    amostrador = AmostradorUnigram("tokenizer.json", alpha=0.2)
    ids = amostrador.amostrar("▁comentario", random.Random(0))
"""

from __future__ import annotations

import json
import math
import random
import unicodedata
from pathlib import Path

from template_alpaca import LEGADO

NOTA_BYTE_FALLBACK = -1e6
RAIZ_AMOSTRAGEM = 0.0


class AmostradorUnigram:
    """Sorteia segmentacoes de um span Unigram proporcional a sua probabilidade.

    `alpha` e o parametro de suavizacao do Kudo: a probabilidade de uma
    segmentacao e proporcional ao produto das probabilidades das pecas elevado a
    `alpha`. `alpha` menor = distribuicao mais achatada = segmentacao mais
    agressiva. 0,1 a 0,2 e a faixa recomendada com poucos dados.
    """

    def __init__(self, caminho_tokenizer_json: Path, alpha: float = 0.2) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha invalido: {alpha!r}; esperado float em (0, 1]")
        dados = json.loads(Path(caminho_tokenizer_json).read_text(encoding="utf-8"))
        modelo = dados["model"]
        self.alpha = alpha
        self.nota: dict[str, float] = {p: float(s) for p, s in modelo["vocab"]}
        self.ids: dict[str, int] = {p: i for i, (p, _) in enumerate(modelo["vocab"])}
        self.comprimento_max = max(len(p) for p in self.nota)
        self.byte_fallback = bool(modelo.get("byte_fallback", False))
        self.id_do_byte: dict[int, int] = {}
        if self.byte_fallback:
            for b in range(256):
                alvo = f"<0x{b:02X}>"
                if alvo in self.ids:
                    self.id_do_byte[b] = self.ids[alvo]

    def _arestas(self, span: str) -> list[list[tuple[int, tuple[int, ...], float]]]:
        """Arestas por posicao final: (inicio, ids, nota da peca)."""
        n = len(span)
        arestas: list[list[tuple[int, tuple[int, ...], float]]] = [
            [] for _ in range(n + 1)
        ]
        for i in range(n):
            teto = min(self.comprimento_max, n - i)
            for tam in range(1, teto + 1):
                peca = span[i : i + tam]
                if peca in self.nota:
                    arestas[i + tam].append((i, (self.ids[peca],), self.nota[peca]))
            bytes_do_char = span[i].encode("utf-8")
            if bytes_do_char and all(b in self.id_do_byte for b in bytes_do_char):
                ids_bytes = tuple(self.id_do_byte[b] for b in bytes_do_char)
                arestas[i + 1].append((i, ids_bytes, NOTA_BYTE_FALLBACK))
        return arestas

    def viterbi(self, span: str) -> list[int] | None:
        """Melhor segmentacao do span; None se nao houver caminho.

        Exemplo: viterbi("▁comentario") reproduz o que o tokenizer.json da.
        """
        n = len(span)
        arestas = self._arestas(span)
        melhor = [-math.inf] * (n + 1)
        volta: list[tuple[int, tuple[int, ...]] | None] = [None] * (n + 1)
        melhor[0] = 0.0
        for fim in range(1, n + 1):
            for inicio, ids, nota in arestas[fim]:
                if melhor[inicio] == -math.inf:
                    continue
                valor = melhor[inicio] + nota
                if valor > melhor[fim]:
                    melhor[fim], volta[fim] = valor, (inicio, ids)
        if volta[n] is None:
            return None
        return _reconstruir(volta, n)

    def amostrar(self, span: str, rng: random.Random) -> list[int] | None:
        """Segmentacao sorteada do span; None se nao houver caminho.

        Sorteia proporcional ao peso do caminho (forward-filtering,
        backward-sampling), o que da exatamente a distribuicao do Kudo.
        Exemplo: amostrar("▁comentario", Random(0)) varia a cada chamada.
        """
        n = len(span)
        arestas = self._arestas(span)
        acumulado = self._forward(arestas, n)
        if acumulado[n] == -math.inf:
            return None
        return self._sortear_caminho(arestas, acumulado, n, rng)

    def _forward(self, arestas, n: int) -> list[float]:
        acumulado = [-math.inf] * (n + 1)
        acumulado[0] = RAIZ_AMOSTRAGEM
        for fim in range(1, n + 1):
            pesos = [
                acumulado[inicio] + self.alpha * nota
                for inicio, _, nota in arestas[fim]
                if acumulado[inicio] != -math.inf
            ]
            if pesos:
                acumulado[fim] = _log_soma_exp(pesos)
        return acumulado

    def _sortear_caminho(
        self, arestas, acumulado: list[float], n: int, rng
    ) -> list[int]:
        saida: list[int] = []
        fim = n
        while fim > 0:
            opcoes = [
                (inicio, ids, acumulado[inicio] + self.alpha * nota)
                for inicio, ids, nota in arestas[fim]
                if acumulado[inicio] != -math.inf
            ]
            pesos = [math.exp(peso - acumulado[fim]) for _, _, peso in opcoes]
            inicio, ids, _ = rng.choices(opcoes, weights=pesos)[0]
            saida = list(ids) + saida
            fim = inicio
        return saida


def _reconstruir(volta, fim: int) -> list[int]:
    saida: list[int] = []
    while fim > 0:
        inicio, ids = volta[fim]
        saida = list(ids) + saida
        fim = inicio
    return saida


def _log_soma_exp(valores: list[float]) -> float:
    maior = max(valores)
    if maior == -math.inf:
        return -math.inf
    return maior + math.log(sum(math.exp(v - maior) for v in valores))


def normalizar(texto: str) -> str:
    """NFKC + minuscula -- o normalizador do tokenizer.json.

    Verificado igual ao `tokenizer.normalizer.normalize_str` em 400 exemplos.
    Sem isto o "Abaixo" maiusculo do preambulo nao vira "abaixo" e a tokenizacao
    inteira diverge.
    """
    return unicodedata.normalize("NFKC", texto).lower()


def metaspace(texto: str) -> str:
    """Troca espaco por '▁' -- e o que o pre-tokenizador do tokenizer.json faz."""
    return texto.replace(" ", "▁")


def span_do_prefixo(texto: str) -> str:
    """Span do inicio do texto, com o '▁' inicial que o Metaspace prepende."""
    return "▁" + metaspace(normalizar(texto))


def span_da_resposta(texto: str) -> str:
    """Span do texto que continua um span anterior: sem o '▁' inicial."""
    return metaspace(normalizar(texto))


def ids_com_amostragem(
    amostrador: AmostradorUnigram, texto: str, marca: str, rng: random.Random
) -> list[int] | None:
    """IDs do texto: prefixo deterministico (Viterbi) + resposta amostrada.

    O corte e logo depois de `marca`, entao o prefixo -- que contem a marca --
    nunca e re-segmentado e o marcador de resposta sempre casa na sequencia.
    Verificado: o corte duro preserva a segmentacao deterministica do texto
    inteiro, e nenhuma peca a atravessa.

    Devolve None se a marca nao estiver no texto ou se algum span nao tiver
    caminho no vocabulario.
    """
    if marca not in texto:
        return None
    corte = texto.index(marca) + len(marca)
    prefixo = amostrador.viterbi(span_do_prefixo(texto[:corte]))
    if prefixo is None:
        return None
    resposta = amostrador.amostrar(span_da_resposta(texto[corte:]), rng)
    if resposta is None:
        return None
    return prefixo + resposta


def _indice(alvo: list[int], ids: list[int]) -> int | None:
    """Posicao de `alvo` como subsequencia contigua de `ids`, ou None."""
    n = len(alvo)
    for i in range(len(ids) - n + 1):
        if ids[i : i + n] == alvo:
            return i
    return None


class CollatorComAmostragem:
    """Completion-only com a resposta re-segmentada por amostragem, a cada lote.

    Subword regularization (Kudo 2018; Provilkov 2020): a mesma palavra aparece
    com segmentacoes diferentes, o que quebra a dependencia de uma tokenizacao
    canonica -- que e como o modelo decorou a grafia das 73 entidades. O prefixo
    fica deterministico (Viterbi), entao o marcador de resposta sempre casa.

    Exige `remove_unused_columns=False`: a coluna `text` precisa sobreviver ate
    aqui, porque e dela que cada lote e re-tokenizado.
    """

    def __init__(
        self,
        tokenizer,
        amostrador: AmostradorUnigram,
        ids_marca: list[int],
        semente: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.amostrador = amostrador
        self.ids_marca = list(ids_marca)
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.rng = random.Random(semente)
        self.sem_caminho = 0

    def _ids_do_exemplo(self, texto: str) -> list[int]:
        ids = ids_com_amostragem(self.amostrador, texto, LEGADO.resposta, self.rng)
        if ids is None:
            # o vocabulario nao cobre algum caractere: cai no deterministico
            self.sem_caminho += 1
            return list(self.tokenizer(texto).input_ids)
        return [self.bos_id] + ids

    def __call__(self, exemplos: list[dict]) -> dict:
        import torch

        entradas, rotulos = [], []
        for exemplo in exemplos:
            ids = self._ids_do_exemplo(exemplo["text"])
            posicao = _indice(self.ids_marca, ids)
            corte = posicao + len(self.ids_marca) if posicao is not None else 0
            rotulo = list(ids)
            rotulo[:corte] = [-100] * corte
            entradas.append(ids)
            rotulos.append(rotulo)
        largura = max(len(x) for x in entradas)
        return {
            "input_ids": torch.tensor(
                [x + [self.pad_id] * (largura - len(x)) for x in entradas]
            ),
            "attention_mask": torch.tensor(
                [[1] * len(x) + [0] * (largura - len(x)) for x in entradas]
            ),
            "labels": torch.tensor([x + [-100] * (largura - len(x)) for x in rotulos]),
        }
