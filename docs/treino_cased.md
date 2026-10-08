# Manacá-1B gerando FastAPI em português, com maiúscula

Atualizado em 2026-10-06.

## Resultado

Com a maiúscula e o espaço antes da palavra representados por marcadores próprios no tokenizer (v4), o Manacá-1B escreve uma API FastAPI + SQLAlchemy em português que compila, sobe e passa nas requisições do teste em todos os 182 pedidos sobre 12 entidades que nunca viu no treino, com o nome certo da classe em todos, sem nenhum script de conserto depois. Com a maiúscula dentro das peças (v2), o acerto do nome da classe era 32%.

| Teste amplo (182 pedidos, 12 entidades inéditas) | eval | entidade_rate | compile_rate | runs_rate | docstring_rate | rotas | funcoes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Manacá, tokenizer v2 + CPT grande + 2.000 nomes | bf16 | 0,32 | 0,88 | 0,87 | 0,88 | 0,74 | 0,84 |
| Manacá, tokenizer v3 (caixa) + CPT de 104M tokens | 4 bits | 0,98 | 1,0 | 0,93 | 1,0 | 1,0 | 0,94 |
| Manacá, tokenizer v4 (caixa + fronteira) + mais 110M tokens | 4 bits | 1,0 | 1,0 | 1,0 | 1,0 | 1,0 | 1,0 |
| Qwen2.5-0.5B, sem CPT | bf16 | 1,0 | 1,0 | 1,0 | 1,0 | 0,945 | 1,0 |

Todos usaram o mesmo teste, o mesmo dataset de SFT (7.932 exemplos, 3 épocas, QLoRA r=32) e a mesma avaliação (`--executar --sem-restauro`, geração greedy). Os adapters do Manacá v3 e v4 foram avaliados sobre o modelo base quantizado em 4 bits, igual ao treino (`--base-4bit`); a seção do v4 explica por quê. O Manacá tinha 41,9 bilhões de tokens de pré-treino, todos em português e sem código, mais os nossos CPTs; o Qwen2.5 teve 18 trilhões. Cada linha é uma rodada só, sem repetição com outras sementes, e o teste cobre só pedidos no formato do molde, sobre CRUD em FastAPI. O Manacá v3 e v4 passam por um passo de decodificação exato entre a geração e a avaliação, que o Qwen não precisa.

No v3, todas as falhas eram de uma entidade só, `agendamento`: o modelo escrevia `AgendaResposta` e `criar_agenda`, e `agenda` é uma das entidades do treino. O v4 resolveu isso.

As rodadas anteriores foram medidas primeiro num teste com só duas entidades, que se mostrou otimista:

| Rodada | Pós-processamento | case_rate | entidade_rate | rotas | compile_rate | runs_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Tokenizer original, só LoRA | restauro de caixa + reparo | n/d | n/d | 0,97 | n/d | 1,0 |
| Tokenizer v2 + CPT pequeno, só LoRA | nenhum | 0,09 | n/d | 0,03 | 0,0 | 0,0 |
| Tokenizer v2 + CPT pequeno, LoRA + lm_head | nenhum | 1,0 | n/d | 0,0 | 0,625 | 0,59 |
| Tokenizer v2 + CPT grande, só LoRA | nenhum | 1,0 | 0,5 | 0,875 | 0,81 | 0,72 |
| Idem, com campos sorteados no SFT | nenhum | 1,0 | 0,5 | 0,94 | 0,97 | 0,97 |
| Idem, com 266 nomes extras no SFT | nenhum | 1,0 | 0,75 | 0,94 | 1,0 | 1,0 |

A primeira linha só chega a runs_rate 1,0 com dois scripts que corrigem a saída. O runs_rate das três últimas conta como sucesso um app que roda com a classe errada, por isso a coluna entidade_rate entrou depois: ela foi medida de novo sobre as predições já salvas da antepenúltima rodada.

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

A investigação passou por quatro rodadas de SFT sobre o mesmo CPT grande.

| SFT sobre o CPT grande | case_rate | entidade_rate | rotas | compile_rate | runs_rate | docstring_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 3 épocas, campos fixos por entidade (loss final 0,034) | 1,0 | 0,5 | 0,875 | 0,81 | 0,72 | 0,81 |
| 1 época, campos fixos (loss final 0,074) | 0,78 | n/d | 0,78 | 0,16 | 0,0 | 0,125 |
| 3 épocas, campos sorteados (loss final 0,039) | 1,0 | 0,5 | 0,94 | 0,97 | 0,97 | 0,97 |
| 3 épocas, campos sorteados + 266 nomes extras (loss final 0,033) | 1,0 | 0,75 | 0,94 | 1,0 | 1,0 | 1,0 |

Na primeira, 6 das 9 falhas eram erro de sintaxe em rotas de delete, como `def deletar_comentario(item_id: int, sessao.query(Membro)):`. A classe também saiu `Membro`, uma entidade do treino. A primeira hipótese foi excesso de treino, e o SFT de 1 época derrubou essa hipótese: o modelo nem aprendeu o formato.

A segunda hipótese foi que cada entidade de treino aparecia sempre com os mesmos campos, e o modelo reconheceria a entidade pelos campos. O gerador passou a sortear os campos a cada exemplo, e as entidades de teste mantiveram os campos fixos, com o split de teste byte a byte igual (mesmo sha256). A sintaxe melhorou muito, mas o entidade_rate ficou exatamente em 0,5.

Separando por entidade, o 0,5 tem uma causa só:

| Pedido | Classe gerada | Função gerada |
| --- | --- | --- |
| endereco (16) | `Endereco` em 16 | `deletar_enderco`, sem um "e", no singular; o plural sai certo |
| comentario (16) | `Perfil` 10, `Paciente` 3, `Curso`, `comenta`, `miguelonte` | `deletar_comentario` certo em 16 |

O tokenizer v2 explica a diferença. A palavra da instrução e o nome da função usam uma peça só, `comentario`. A classe vira `▁Com` `enta` `rio`, três peças sem nada em comum com a primeira. Com `endereco` acontece o mesmo (`▁Ende` `reco`), e mesmo assim ele acerta. A frequência no corpus do CPT não separa os dois casos: `Comentario` aparece 94 vezes e `Endereco` 146. O SFT ensinava a tradução de minúscula para classe entidade por entidade, e numa entidade inédita ela só funciona quando o caminho de peças já é provável por acaso.

A primeira mudança foi dar ao SFT 266 nomes extras de entidade, com 6 exemplos sorteados cada (`scripts/nomes_entidades.py`). O treino passou de 2.336 para 3.932 exemplos, e o split de teste não mudou. Com centenas de pares de minúscula e classe, o modelo tem mais chance de aprender a regra em vez de decorar a tabela.

O entidade_rate subiu de 0,5 para 0,75, e o defeito do `enderco` sumiu. `Endereco` continua certo em 16 de 16. `Comentario` sai certo em 8 de 16, e os outros 8 saem `Contato`, um dos nomes extras. A classe certa e a errada começam com peças parecidas (`▁Com` e `▁Con`). Os erros se concentram nos pedidos de listar, que trazem só o plural `comentarios` na instrução.

Com só duas entidades de teste, a diferença entre 0,5 e 0,75 era de 8 pedidos de uma única entidade. Por isso entrou um teste amplo (`data/fastapi_ptbr_teste_amplo.jsonl`): o teste original mais 10 entidades que nunca entram no treino, 182 pedidos distintos. Sete das 10 classes novas nunca aparecem como `class X(` no corpus do CPT, e as outras três aparecem no máximo duas vezes. Como o treino não mudou, os três adapters foram medidos de novo sem treinar nada:

| Adapter | entidade_rate (12 entidades) | compile_rate | runs_rate | docstring_rate | entidade_rate (2 entidades) |
| --- | --- | --- | --- | --- | --- |
| campos fixos por entidade | 0,14 | 0,68 | 0,64 | 0,67 | 0,5 |
| campos sorteados | 0,13 | 0,86 | 0,84 | 0,86 | 0,5 |
| campos sorteados + 266 nomes extras | 0,29 | 0,81 | 0,75 | 0,81 | 0,75 |
| campos sorteados + 266 extras + 2.000 nomes colhidos | 0,32 | 0,88 | 0,87 | 0,88 | n/d |

O teste de duas entidades era otimista: `Endereco` saía certo por sorte do caminho de peças, não por regra aprendida. Sortear os campos melhorou a sintaxe e não mudou a cópia do nome. Os nomes extras dobraram o entidade_rate, mas a partir de um patamar baixo.

A última linha testou a escala. O script `colher_nomes_classe.py` juntou 2.000 nomes de duas fontes: classes de uma palavra do código colhido e palavras da Wikipédia com sufixo de substantivo (`-ção`, `-mento`, `-dade`). Cada nome entrou com 2 exemplos, e o treino foi de 3.932 para 7.932 exemplos, com 2 h 11 de SFT. Sintaxe e execução melhoraram (compile_rate 0,88, runs_rate 0,87), mas o entidade_rate passou só de 0,29 para 0,32. Seis vezes mais nomes quase não mexeram na cópia do nome.

Com isso a escala de dados fica descartada como solução para o nome da classe.

Para saber se o limite era do dataset ou do Manacá, o mesmo SFT (7.932 exemplos, 3 épocas, QLoRA) rodou no [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) base, que tem metade do tamanho, licença Apache 2.0 e um tokenizer BPE de bytes que já trata maiúsculas. Sem nenhum CPT:

| Teste amplo (182 pedidos) | entidade_rate | compile_rate | runs_rate | docstring_rate | rotas | tempo de SFT |
| --- | --- | --- | --- | --- | --- | --- |
| Manacá-1B + CPT grande + 2.000 nomes | 0,32 | 0,88 | 0,87 | 0,88 | 0,74 | 2 h 11 |
| Qwen2.5-0.5B, sem CPT | 1,0 | 1,0 | 1,0 | 1,0 | 0,945 | 54 min |

O dataset e o eval estão certos: um modelo que já sabe caixa aprende a tarefa inteira só com o SFT. O limite é o pré-treino do Manacá, todo em minúsculas, que 35 milhões de tokens de CPT não corrigem. O Qwen3, por comparação, foi pré-treinado em 36 trilhões de tokens ([relatório técnico](https://arxiv.org/html/2505.09388)). O Qwen3 ficou de fora do controle porque exige transformers 4.51, e o stack da 3090 usa o 4.46.3.

Para rodar o Qwen, o `train_qlora.py` passou a trocar o pad quando ele é o próprio EOS (no Qwen os dois são `<|endoftext|>`, e o collator apagaria o EOS dos labels), e o eval passou a cortar a saída no primeiro EOS ou pad.

Para manter o Manacá como base, o tokenizer v3 passou a marcar a maiúscula com uma peça própria, como em [Land e Meister](https://arxiv.org/html/2608.08847). `Comentario` vira `<|maj|>` seguido da mesma peça `comentario` da instrução, e `HTTPException` vira `<|mai|>http<|maj|>exception`. O marcador vai antes do espaço que precede a palavra, para que a peça continue sendo `▁comentario`. A conversão é exata nos dois sentidos e roda fora do modelo: o texto é codificado antes de tokenizar e decodificado depois de gerar (`scripts/caixa_marcada.py`).

O v3 tem as 64.000 peças originais do Manacá (que baixam a caixa), as 43.120 peças sem espaço do v2 e os 2 marcadores, 107.122 tokens no total. O CPT partiu do CPT grande, com as linhas do embedding copiadas peça por peça, e os marcadores começaram pela média das letras maiúsculas do v2. O corpus juntou o código e a Wikipédia de antes com 65 milhões de tokens do [GigaVerbo-v2](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2), o corpus do projeto Tucano 2, filtrado por nota educativa e toxicidade: 104 milhões de tokens, 6.352 passos, cerca de 8 h 45 na RTX 3090. O SFT foi o mesmo das rodadas anteriores (7.932 exemplos, 3 épocas, 2 h 21).

Com o marcador, a maiúscula deixa de ser uma ortografia separada para cada palavra e vira uma decisão só, que vale para todas. O modelo aprende quando pôr `<|maj|>` com cada palavra maiúscula do corpus e aplica isso a palavras que nunca viu, copiando a peça minúscula da instrução.

O que sobrou no v3 era o mesmo problema em outra forma. `agendamento` depois de espaço é a peça `▁agendamento`, mas colado em `criar_` era a peça `agendamento` sem `▁`, com outro id, e nesse lugar perdia para `agenda`, que o SFT escreveu centenas de vezes. O artigo de Land e Meister conta seis formas possíveis de uma palavra: três de caixa vezes duas de espaço antes. O v3 tirou a caixa; o v4 tirou o espaço com um terceiro marcador, `<|cola|>`, que diz que a palavra não tinha espaço antes. Toda palavra passa a ser a forma com espaço, e `criar_agendamento` vira `criar_<|cola|> agendamento`, com a mesma peça da instrução. Com as duas fontes viradas marcador, o v4 voltou a ser o tokenizer original do Manacá mais 3 tokens (64.003); as peças sem espaço do v2 deixaram de ser precisas. O texto codificado fica 6% maior no corpus e 15% maior no código do dataset.

O CPT do v4 partiu do CPT do v3, com as peças copiadas pelo texto e só o `<|cola|>` novo: 53.928 blocos (cerca de 110 milhões de tokens), 6.741 passos, cerca de 9 h; a val loss foi de 3,14 para 2,08. O SFT e a avaliação do v4 tiveram dois bugs, descritos na tabela abaixo: o EOS sumia do texto de treino, e a avaliação em bf16 de um adapter treinado sobre o base em 4 bits trocava `connect_args={` por `connect_args=Depends` em todos os pedidos.

## Problemas encontrados no caminho

| Problema | Causa | Correção |
| --- | --- | --- |
| Conjunto de teste vazio | o filtro procurava `class comentario` em minúsculas | usa o mesmo `nome_de_classe` do gerador, com teste de regressão |
| Resposta 422 contada como sucesso na sandbox | o filtro de schema de entrada não reconhecia `UsuarioCriar` com maiúscula | comparação sem diferenciar caixa |
| Erro ao expandir o embedding | matriz com 64.128 linhas, não 64.000 | aceita o preenchimento, recusa só matriz menor |
| v4: o modelo não parava de gerar, compile_rate 0,0 | o texto de treino era codificado junto com o EOS, e `</s>` virava `</<|cola|> s>`, sem o token de fim | o EOS é colado cru depois da codificação, com teste de regressão |
| v4: runs_rate 0,0 com tudo compilando | o adapter foi treinado sobre o base em 4 bits e avaliado sobre o base em bf16; depois de `=`, o `{` é o byte cru `<0x7B>` e o `<|cola|>` é o token mais comum, e a diferença numérica virou essa escolha | `eval_code.py --base-4bit`, que carrega o base como o treino |
| JSON quebrado na linha 3.131 do corpus | `str.splitlines()` também quebra em `\u2028` e `\x85`, que o `json.dumps(ensure_ascii=False)` deixa crus dentro do texto | os leitores de JSONL usam `split("\n")`, com teste de regressão |

## Ambiente

RTX 3090 de 24 GB, Python 3.12, torch 2.5.1 (CUDA 12.4), transformers 4.46.3, trl 0.12.2, peft 0.13.2, datasets 3.1.0. O pipeline inteiro roda com `scripts/rodar_cpt_grande_3090.sh`, em seis etapas: colher código, colher Wikipedia, montar o corpus, CPT, SFT e avaliação. Cada etapa grava o log em `logs/`.

## Rodar o modelo fora da 3090

O modelo final são duas peças: `checkpoints/manaca-cpt-v4` (o modelo depois do CPT, 1,72 bilhão de parâmetros em bf16, 3,4 GB) e `checkpoints/manaca-fastapi-v4` (o adapter LoRA do SFT, 134 MB). As duas ficam fora do git.

O adapter foi treinado sobre o base quantizado em NF4 (QLoRA) e só acerta sobre esse base. Sem GPU NVIDIA não há bitsandbytes, então `scripts/nf4_simulado.py` arredonda os pesos para a grade NF4 em torch puro (os 16 valores e o bloco de 64 do bitsandbytes), e o adapter vê os pesos do treino. `scripts/chat_fastapi.py` usa isso em CPU, mas leva mais de 10 minutos por resposta.

Para usar a GPU Intel desta máquina, o adapter é fundido no base já arredondado (`scripts/fundir_adapter.py`) e convertido para GGUF pelo `convert_hf_to_gguf.py` do llama.cpp. O tokenizer do GGUF não serve: o Manacá é Unigram com byte fallback, e o llama.cpp não tem essa combinação. No modo Unigram (`t5`), `\n` vira `<unk>`; no modo SentencePiece (`llama`), a segmentação sai por fusões. Em 100 textos, 6 saíram idênticos ao tokenizer do treino em cada modo. Por isso `scripts/chat_gguf.py` tokeniza do lado do Python e conversa com o `llama-server` por ids (o `/completion` aceita o prompt como lista de ids e devolve os ids gerados). Pelo mesmo motivo o Ollama não serve: ele só aceita texto.

```
llama-server -m models/manaca-fastapi-v4-q8_0.gguf --port 8080 -ngl 99 -c 4096
uv run --with transformers==4.46.3 python scripts/chat_gguf.py
```

Quantizado pelo `llama-quantize`, sobre o GGUF F16 fundido, na GPU Intel Arc desta máquina:

| Arquivo | Tamanho | Velocidade | Resultado medido |
| --- | --- | --- | --- |
| F16 | 3,4 GB | ~10 tokens/s | referência; app do pedido de teste sobe na sandbox |
| Q8_0 | 1,8 GB | ~15 tokens/s | 10 de 10 pedidos do teste amplo (5 entidades) compilam, sobem e acertam a classe; saída idêntica à do F16 no pedido de teste |
| Q6_K | 1,4 GB | n/d | não medido |
| Q5_K_M | 1,2 GB | n/d | não medido |
| Q4_K_M | 1,1 GB | n/d | quebrou: no pedido de teste saiu um arquivo sem nenhuma rota |

O Q8_0 é o menor arquivo com resultado medido igual ao F16. O Q4_K_M confirma que o modelo é sensível a arredondamento, como já tinha mostrado a diferença entre NF4 e bf16. Os níveis intermediários ficaram sem medir porque a bateria (`scripts/eval_gguf.py`, 2 pedidos por entidade) parou no meio do Q8_0.

O modelo responde a qualquer pedido com um arquivo FastAPI, inclusive a "oi" (sai `class Oi(Base)`). O SFT só teve pedidos de FastAPI, e o ponto de partida é o Manacá base, que não conversa.

## Próximo passo: versão que conversa

Para o modelo conversar e só gerar FastAPI quando pedirem, o SFT precisa misturar instruções gerais com os pedidos de FastAPI. A receita do Manacá-1B-Instruct está no [repositório do LNCC](https://github.com/Instituto-IA-LNCC/manaca-1b-instruct) (`sft/prepare_data.py`, `sft/jaster/build_jaster.py`, `docs/data/`):

| Fonte | Dataset no Hugging Face | Conversão |
| --- | --- | --- |
| Alpaca-PT | `dominguesm/alpaca-data-pt-br` | instruction, input, output como estão |
| Aya (pt) | `CohereForAI/aya_dataset` | `inputs` vira instrução, `targets` vira resposta, só idioma português |
| OASST1 (pt) | `OpenAssistant/oasst1` | pares prompter e assistant com rank 0, só português |
| Tradução | `Helsinki-NLP/opus-100`, en-pt, 20.000 primeiros | duas instruções por par (para inglês e para português) |
| Resumo | `csebuetnlp/xlsum`, português, 10.000 primeiros | texto no input, resumo na resposta |
| JASTER | ASSIN2, SICK-BR, FaQuAD, ENEM, HAREM, LeNER-Br, TweetSentBR, HateBR, Pirá, Mac-Morpho | tarefas de PLN em formato de instrução |

A versão v2 do LNCC tem 141.404 exemplos, com deduplicação e corte de tamanho. O template deles tem preâmbulo próprio quando há `input` e o marcador `### Entrada:`, e o `train_qlora.py` deste projeto ainda ignora o campo `input`. O que falta fazer:

- [ ] Rodar o `sft/prepare_data.py` do LNCC (fontes da v2) ou reproduzir os filtros dele, e gravar o resultado no formato Alpaca.
- [ ] Ensinar o `montar_texto` do `train_qlora.py` a usar `input`, com o preâmbulo e o `### Entrada:` do LNCC, e com a mesma codificação de caixa e fronteira.
- [ ] Escolher a proporção. Os 141 mil exemplos inteiros levariam dezenas de horas na 3090 (os 7.932 de FastAPI levam 2 h 30 em 3 épocas); uma amostra de 10 a 20 mil por fonte, mais os 7.932 de FastAPI, cabe em 5 a 8 horas.
- [ ] SFT a partir do `manaca-cpt-v4`, eval no teste amplo com `--base-4bit` (tem de continuar em 1,0) e um conjunto de pedidos de conversa que não pode gerar código.
- [ ] Licença: Alpaca-PT é CC BY-NC e XLSum é CC BY-NC-SA, por isso o instruct do LNCC é CC BY-NC 4.0. Usando a mesma receita, o modelo daqui herda essa restrição; o Instruct-Aira v3 (Apache 2.0, 50 mil conversas em português) é a alternativa sem restrição.

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
