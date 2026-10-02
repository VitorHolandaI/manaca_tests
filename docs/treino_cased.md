# Manacá-1B gerando FastAPI em português, com maiúscula

Atualizado em 2026-10-02.

## Resultado

Depois de um pré-treino continuado grande e de um ajuste fino com QLoRA, o Manacá-1B escreve uma API FastAPI + SQLAlchemy em português, com a caixa certa e sem nenhum script de conserto depois. Nos 32 pedidos de teste, sobre duas entidades que o modelo nunca viu no treino, 81% das respostas compilam e 72% sobem e passam nas requisições do teste.

| Rodada | Pós-processamento | case_rate | rotas | compile_rate | runs_rate |
| --- | --- | --- | --- | --- | --- |
| Tokenizer original, só LoRA | restauro de caixa + reparo | n/d | 0,97 | n/d | 1,0 |
| Tokenizer v2 + CPT pequeno, só LoRA | nenhum | 0,09 | 0,03 | 0,0 | 0,0 |
| Tokenizer v2 + CPT pequeno, LoRA + lm_head | nenhum | 1,0 | 0,0 | 0,625 | 0,59 |
| Tokenizer v2 + CPT grande, só LoRA | nenhum | 1,0 | 0,875 | 0,81 | 0,72 |

A primeira linha ainda tem o melhor runs_rate, mas só chega lá com dois scripts que corrigem a saída. A última é a primeira rodada em que o próprio modelo acerta a caixa e a entidade nova.

## O que o projeto faz

O ponto de partida é o [Manacá-1B base](https://huggingface.co/AIInstituteLNCC/manaca-1b-base), do LNCC, publicado sob CC BY 4.0. O modelo recebe um pedido como "Crie um endpoint de exclusão (delete) para endereço no FastAPI com SQLAlchemy" e devolve um arquivo Python completo: engine SQLite, model SQLAlchemy, schemas Pydantic, a sessão e as rotas. Classes, funções, tabelas, comentários e docstrings saem em português.

A primeira versão já gerava código que funcionava, mas tudo saía em minúsculas. Dois scripts arrumavam a saída: `restore_case.py` recolocava as maiúsculas em `Column`, `FastAPI`, `HTTPException` e nas classes, e `reparar_identificadores.py` emendava identificadores que o modelo partia. Esta rodada tira os dois do caminho.

## Por que tudo saía em minúsculas

O tokenizer do Manacá é um Unigram do SentencePiece com 64.000 peças, e o normalizador dele tem um passo `Lowercase`. Todo texto vira minúsculo antes de ser tokenizado, no treino e na inferência. O modelo nunca viu uma letra maiúscula, então não tem como escrever uma.

Para prosa em português isso quase não atrapalha. Em Python atrapalha: `Column` e `column` são nomes diferentes, e `class endereco(base)` não roda.

## Tokenizer v2

O v2 é o mesmo tokenizer sem o passo `Lowercase` e com 51.148 peças novas no fim do vocabulário, totalizando 115.148. As peças novas são as letras maiúsculas, as palavras com caixa (`Column`, `FastAPI`, `Comentario`) e as formas sem o espaço inicial do SentencePiece (`▁`), que aparecem no meio de identificadores.

Os ids de 0 a 63.999 não mudam. O script `expandir_vocab.py` confere isso peça por peça e para com erro se algum id antigo trocou de peça. Assim, tudo o que o modelo aprendeu continua valendo para o texto que já era minúsculo.

As peças novas precisam de linhas novas na matriz de embedding e no `lm_head`. O modelo não amarra as duas matrizes (`tie_word_embeddings=false`), então as duas são expandidas. Cada linha nova começa como a média das linhas das peças antigas em que a peça nova se decompõe. Quando não há decomposição, usa a média de todas as linhas antigas.

Um detalhe que custou um erro: a matriz do modelo base tem 64.128 linhas, não 64.000. As 128 a mais são preenchimento para alinhar a memória. O script aceita esse preenchimento e só recusa matriz menor que o vocabulário antigo.

O problema dessa inicialização aparece logo. `Column` começa igual a `▁column`, e o modelo não tem motivo para preferir uma à outra. Sem treino em cima, ele escreve `importapi`, `column( boolean` e `deletar_ comentario`.

## Dataset de ajuste fino

O dataset sai inteiro de `scripts/gen_dataset.py`, sem dado de terceiro (CC0-1.0). São 2.336 exemplos de treino e 64 de teste no formato Alpaca (`instruction`, `input`, `output`). Cada exemplo pede uma operação (criar, listar, buscar, atualizar ou deletar) sobre uma entidade, e a resposta é o arquivo completo.

O teste usa só as entidades `comentario` e `endereco`, que não aparecem em nenhum exemplo de treino. Isso mede se o modelo generaliza o padrão para um nome novo ou só repete o que decorou.

Nesta rodada o gerador passou a escrever caixa real de Python: `class Comentario(Base)`, `ComentarioCriar`, `ComentarioResposta`, instruções com a primeira letra maiúscula, uma docstring em cada rota e `detail="Comentario nao encontrado"` no 404.

## Corpus do pré-treino continuado

O corpus junta três fontes, todas lidas com caixa original:

| Fonte | O que entra | Licença | Documentos |
| --- | --- | --- | --- |
| [FastAPI](https://github.com/fastapi/fastapi), commit `a3d205bf` | `fastapi/*.py`, `docs_src/*.py`, `docs/pt/*.md` | MIT | (repositório inteiro dessas pastas) |
| [codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) | Python com identificadores e comentários em português | MIT, Apache-2.0, BSD, ISC, Unlicense, CC0 | 5.255 |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), `20231101.pt` | artigos com 1.500 caracteres ou mais, até 60 milhões de caracteres | CC BY-SA 3.0 | 3.721 |

O github-code-clean tem 880 arquivos parquet e 314 GB. Python em português é cerca de 0,1% do Python ali (no primeiro shard, 7 de 5.053 arquivos permissivos). Por isso o script `colher_codigo_ptbr.py` varre todos os shards, mas baixa um de cada vez, filtra e apaga. O disco nunca guarda mais que alguns shards. A varredura levou cerca de 90 minutos e retoma de onde parou se cair.

Um arquivo entra se for Python, tiver licença permissiva, tiver até 100.000 caracteres e usar pelo menos 5 palavras portuguesas distintas de uma lista que não colide com inglês (`senha`, `usuario`, `retorna`, `endereco`; nunca `do`, `as`, `os`). Identificadores em camelCase são quebrados antes da contagem, então `obterUsuario` conta `obter` e `usuario`. Arquivos repetidos são descartados pelo SHA-256 do texto.

Nenhum documento do corpus pode conter as entidades do teste. O `montar_corpus_cpt.py` descarta qualquer texto com `class Comentario`, `class Endereco` ou com uma instrução do conjunto de teste. Uma pequena parte de cada fonte vai para validação, escolhida pelo hash do caminho, para que a separação seja a mesma em toda execução.

O texto é empacotado em blocos de 2.048 tokens, com um EOS entre documentos. Saíram 17.038 blocos de treino (cerca de 35 milhões de tokens) e 96 de validação.

## Treino

O treino tem duas etapas com objetivos diferentes.

| | Pré-treino continuado (CPT) | Ajuste fino (SFT) |
| --- | --- | --- |
| Dado | texto corrido, sem pergunta e resposta | pares instrução e código |
| Pesos treinados | o modelo inteiro, em bf16 | só um adaptador LoRA (QLoRA 4-bit, r=32) |
| Ensina | o vocabulário novo, Python e português com caixa | o formato da resposta |
| Volume | ~35M tokens, 1 época, 2.129 passos | 2.336 exemplos, 3 épocas, 438 passos |
| Taxa de aprendizado | 2e-5 | 2e-4 |
| Batch efetivo | 8 blocos (1 × 8 de acumulação) | 16 exemplos (1 × 16 de acumulação) |
| Tempo na RTX 3090 | 2 h 57 min | 39 min |

O nome "pré-treino" vem do tipo de treino, não do momento. É o mesmo processo que criou o modelo base: prever o próximo token em texto bruto, com todos os pesos livres. A diferença é partir dos pesos já treinados em vez de pesos aleatórios.

O CPT usa `adamw_bnb_8bit` e gradient checkpointing, e ocupou cerca de 15 GB dos 24 GB da placa. A loss de validação caiu de 4,2169 antes do treino para 2,0274 no fim. Os checkpoints intermediários são apagados no fim porque o estado do otimizador passa de 10 GB.

No SFT, a matriz de embedding e o `lm_head` ficam congelados. Isso foi uma decisão tomada depois de medir. Numa rodada anterior, com o `lm_head` treinável, o SFT de fato quebrou o empate entre `Column` e `▁column` (case_rate 1,0), mas empurrou para baixo toda peça que não aparece como resposta no treino. As peças de `comentario` e `endereco` estão entre elas, e o modelo passou a escrever `deletar_ comentario`. Das 201 peças novas que o SFT usa, 100 não apareciam no corpus do CPT pequeno, e só 236 das 51.148 peças novas apareciam 20 vezes ou mais. Daí veio a decisão de aumentar o CPT, que é onde a peça nova aprende sentido, e deixar o SFT só no LoRA.

## Como a avaliação mede

O `eval_code.py` gera a resposta de forma determinística (greedy) para cada pedido de teste e mede sem tocar na saída (`--sem-restauro`):

| Métrica | O que conta |
| --- | --- |
| compile_rate | o arquivo é Python sintaticamente válido |
| runs_rate | o app sobe numa sandbox e responde certo às requisições do teste (`--executar`) |
| case_rate | os nomes que o restauro antigo corrigiria já saem com a caixa certa |
| docstring_rate | as docstrings começam com maiúscula |
| rotas, funcoes, status | a rota, a função e o código de status esperados aparecem no código |
| entidade_rate | a classe do model SQLAlchemy é a entidade pedida |

## O que ainda falha

Das 9 falhas da rodada atual, 6 são erro de sintaxe e todas estão em rotas de delete. O modelo escreve `def deletar_comentario(item_id: int, sessao.query(Membro)):`, pulando a assinatura direto para o corpo. Os 438 exemplos de delete do treino têm a mesma assinatura correta, então o erro não vem de dado inconsistente.

No mesmo exemplo, a classe saiu como `Membro`, uma entidade do treino, com os campos de `endereco`. O `comentario` só aparece no nome da rota e da função. O case_rate não pega isso, porque `Membro` tem a caixa certa.

Outras duas falhas usam `response_model=EnderecoResposta` numa rota de listar, onde o certo é `list[EnderecoResposta]`. A última usa um nome de parâmetro na rota e outro no corpo.

A primeira hipótese foi que o SFT tinha decorado demais, já que a loss terminou em 0,034 depois de 3 épocas. O teste foi refazer só o SFT com 1 época sobre o mesmo CPT, e o resultado piorou em tudo:

| SFT sobre o CPT grande | case_rate | rotas | compile_rate | runs_rate | docstring_rate |
| --- | --- | --- | --- | --- | --- |
| 3 épocas (loss final 0,034) | 1,0 | 0,875 | 0,81 | 0,72 | 0,81 |
| 1 época (loss final 0,074) | 0,78 | 0,78 | 0,16 | 0,0 | 0,125 |

Com 1 época o modelo nem aprendeu o formato: a sintaxe quebra em quase todo pedido, não só no delete. O adapter de 3 épocas continua sendo o melhor, e a troca de entidade não vem de excesso de treino.

A hipótese atual está no dataset. Cada entidade de treino aparecia sempre com o mesmo conjunto de campos, então o modelo pode ter aprendido a reconhecer a entidade pelos campos em vez de copiar o nome da instrução. Diante de uma entidade nova, ele recorre a uma de treino. Isso ainda não foi medido. As mudanças para testar:

- O gerador agora sorteia o conjunto de campos a cada exemplo de treino. A entidade `usuario`, por exemplo, aparece com os 14 conjuntos. As entidades de teste mantêm os campos fixos e o split de teste continua byte a byte igual ao medido (mesmo sha256), para que os números sigam comparáveis.
- O eval ganhou `entidade_rate`: a classe do model SQLAlchemy precisa ser a da referência. Uma saída com `class Membro(Base)` num pedido de comentário agora reprova nessa métrica, mesmo compilando e rodando.

O próximo passo é refazer o SFT de 3 épocas com o dataset novo, cerca de 40 minutos, e comparar `entidade_rate` com o adapter atual.

## Problemas encontrados no caminho

| Problema | Causa | Correção |
| --- | --- | --- |
| Conjunto de teste vazio | o filtro procurava `class comentario` em minúsculas | usa o mesmo `nome_de_classe` do gerador, com teste de regressão |
| Resposta 422 contada como sucesso na sandbox | o filtro de schema de entrada não reconhecia `UsuarioCriar` com maiúscula | comparação sem diferenciar caixa |
| Erro ao expandir o embedding | matriz com 64.128 linhas, não 64.000 | aceita o preenchimento, recusa só matriz menor |
| JSON quebrado na linha 3.131 do corpus | `str.splitlines()` também quebra em ` ` e `\x85`, que o `json.dumps(ensure_ascii=False)` deixa crus dentro do texto | os leitores de JSONL usam `split("\n")`, com teste de regressão |

## Ambiente

RTX 3090 de 24 GB, Python 3.12, torch 2.5.1 (CUDA 12.4), transformers 4.46.3, trl 0.12.2, peft 0.13.2, datasets 3.1.0. O pipeline inteiro roda com `scripts/rodar_cpt_grande_3090.sh`, em seis etapas: colher código, colher Wikipedia, montar o corpus, CPT, SFT e avaliação. Cada etapa grava o log em `logs/`.

## Licenças

O dataset de ajuste fino é CC0. O código do FastAPI e o código colhido do GitHub têm licenças permissivas, registradas por documento no corpus. A Wikipedia é CC BY-SA 3.0, que exige atribuição e compartilhamento pela mesma licença. Isso precisa ser resolvido antes de distribuir pesos treinados com ela. O modelo base é CC BY 4.0 e exige atribuição ao LNCC.

## Fontes

- [Manacá-1B base](https://huggingface.co/AIInstituteLNCC/manaca-1b-base), modelo de partida
- [codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean), código do GitHub com licença por arquivo
- [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), dump `20231101.pt`
- [FastAPI](https://github.com/fastapi/fastapi), código e documentação em português
- Gururangan et al., [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964), 2020, sobre pré-treino continuado em domínio
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685), 2021
- Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314), 2023
- Kudo e Richardson, [SentencePiece](https://arxiv.org/abs/1808.06226), 2018
