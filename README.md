# Manacá-1B para gerar APIs FastAPI

Ajuste do [Manacá-1B](https://huggingface.co/AIInstituteLNCC/manaca-1b-base),
modelo de português do LNCC que já existia, para escrever APIs FastAPI com
SQLAlchemy em português. O modelo base foi treinado só com texto em minúsculas e
sem código. Este repositório tem o que fiz em cima dele: tokenizer estendido,
pré-treino continuado, SFT com QLoRA e a avaliação.

## Resultado

Em 182 pedidos sobre 12 entidades que o modelo não viu no treino, todo o código
gerado compila, sobe e passa nas requisições de CRUD, e 97,8% acertam o nome da
classe. O Qwen2.5-0.5B, com o mesmo SFT, serviu de referência. Os pedidos são
sintéticos e de um só formato. Os detalhes, a ficha técnica e o que não foi
medido estão em [`docs/ENTREGA.md`](docs/ENTREGA.md) e em
[`VALIDATION.md`](VALIDATION.md).

## Estrutura

- `scripts/`: dados, tokenizer, pré-treino, SFT, conversão e avaliação. O
  [`scripts/README.md`](scripts/README.md) descreve cada script.
- `tests/`: testes dos tokenizers e da avaliação.
- `docs/`: relatório de entrega e notas de treino.

Os pesos não estão no repositório. A licença do modelo ajustado é não comercial,
herdada dos dados de instrução do LNCC (Alpaca-PT e XLSum).
