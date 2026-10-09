# Manacá-1B FastAPI: entrega

Atualizado em 2026-10-09.

## O que foi entregue

Um ajuste do [Manacá-1B](https://huggingface.co/AIInstituteLNCC/manaca-1b-base), o modelo de português do LNCC, que escreve APIs FastAPI + SQLAlchemy em português com a caixa certa de Python e também conversa. Num teste com 12 entidades que o modelo nunca viu no treino, todos os 182 arquivos gerados compilam, sobem e passam nas requisições de CRUD, sem nenhum script de conserto depois. Em 18 pedidos gerais (saudação, perguntas, tradução, resumo), nenhuma resposta vira código.

O modelo roda nesta máquina, na GPU Intel, a cerca de 15 tokens por segundo: um arquivo FastAPI completo sai em uns 45 segundos.

## Resultado

Teste amplo: 182 pedidos de FastAPI sobre 12 entidades fora do treino, avaliados por `scripts/eval_code.py` (compila, sobe na sandbox e roda o CRUD, nome certo da classe).

| Modelo | Nome da classe | Compila | App sobe e responde | Conversa (18 pedidos) |
| --- | --- | --- | --- | --- |
| Manacá, tokenizer v2 (maiúscula dentro das peças) | 32% | 88% | 87% | n/d |
| Manacá, tokenizer v3 (maiúscula como marcador) | 98% | 100% | 93% | n/d |
| Manacá, tokenizer v4 (maiúscula e espaço como marcadores) | 100% | 100% | 100% | não conversa: "oi" vira `class Oi(Base)` |
| **Manacá v4 instruct (entregue)** | **97,8%** | **100%** | **100%** | **18 de 18 em texto** |
| Qwen2.5-0.5B, mesmo SFT (referência) | 100% | 100% | 100% | n/d |

O Manacá saiu de um pré-treino todo em minúsculas e sem código (41,9 bilhões de tokens em português); o Qwen2.5 teve 18 trilhões, com código. A diferença foi fechada com cerca de 250 milhões de tokens de pré-treino continuado numa RTX 3090.

## Como ver funcionando

Precisa do llama.cpp com o `llama-server` no PATH (aqui, o pacote `llama.cpp-sycl`, que usa a GPU Intel) e do `uv`. Da raiz do projeto:

```
scripts/rodar_chat_local.sh
```

O script sobe o modelo, abre o chat e desliga tudo quando o chat fecha. Escreva o pedido e aperte Enter; `sair` ou Ctrl-D encerra. Exemplos:

```
> Oi, tudo bem?
> Crie um endpoint para listar todos os livros no FastAPI com SQLAlchemy
> Como retornar 404 quando o paciente nao existe no FastAPI?
> Traduza para o inglês: o relatório fica pronto amanhã.
```

Um pedido só, sem o chat:

```
scripts/rodar_chat_local.sh instruct "Crie as rotas de criar e listar pedidos no FastAPI"
```

Para rodar o código gerado:

```
scripts/rodar_chat_local.sh instruct "Crie um endpoint para listar todos os livros no FastAPI com SQLAlchemy" > app_livros.py
uv run --with fastapi --with sqlalchemy --with uvicorn uvicorn app_livros:app
```

e abra http://127.0.0.1:8000/docs para testar as rotas pelo navegador.

`scripts/rodar_chat_local.sh v4` usa a versão que só faz FastAPI. Se a porta 8080 já estiver ocupada, `PORTA=8081 scripts/rodar_chat_local.sh`.

## Arquivos

Os pesos ficam fora do git (`models/` e `checkpoints/` estão no `.gitignore`).

| Arquivo | Tamanho | O que é |
| --- | --- | --- |
| `models/manaca-fastapi-instruct-q8_0.gguf` | 1,8 GB | o modelo entregue, pronto para o llama.cpp |
| `models/manaca-fastapi-instruct-f16.gguf` | 3,4 GB | o mesmo em F16, base para outras quantizações |
| `checkpoints/manaca-cpt-v4` | 3,3 GB | o modelo depois do pré-treino continuado (formato Hugging Face) |
| `checkpoints/manaca-fastapi-instruct` | 134 MB | o adapter LoRA do SFT, que vai em cima do anterior |
| `models/manaca-tokenizer-v4` | 4 MB | o tokenizer: o original do Manacá mais 3 marcadores |
| `models/manaca-fastapi-v4-q8_0.gguf` | 1,8 GB | a versão só FastAPI |

## Gerar o GGUF a partir do adapter

O adapter foi treinado sobre o modelo base quantizado em NF4 (QLoRA) e só acerta sobre esse base. Por isso a fusão arredonda o base para NF4 antes (`scripts/nf4_simulado.py`), sem precisar de GPU NVIDIA:

```
uv run --python 3.12 --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --with-requirements requirements-chat.txt \
    python scripts/fundir_adapter.py --adapter checkpoints/manaca-fastapi-instruct \
    --destino models/manaca-fastapi-instruct-fundido
python <llama.cpp>/convert_hf_to_gguf.py models/manaca-fastapi-instruct-fundido \
    --outtype f16 --outfile models/manaca-fastapi-instruct-f16.gguf
llama-quantize models/manaca-fastapi-instruct-f16.gguf \
    models/manaca-fastapi-instruct-q8_0.gguf Q8_0
```

Não use Q4 ou menor: o Q4_K_M quebrou o modelo (saiu arquivo sem rota). O Q8_0 deu o mesmo resultado do F16.

## Por que não roda no Ollama

O tokenizer do Manacá é Unigram com byte fallback, e o llama.cpp não tem essa combinação: em 100 textos, só 6 saíram tokenizados igual ao treino. O `chat_gguf.py` contorna isso tokenizando do lado do Python e mandando os ids para o `llama-server`. O Ollama só aceita texto e usaria o tokenizer do llama.cpp, então o modelo receberia um texto que nunca viu.

Além disso, o texto passa por uma conversão exata antes e depois do modelo (`scripts/caixa_marcada.py`): maiúscula e espaço viram marcadores (`Comentario` vira `<|maj|> comentario`). Quem usar o modelo fora destes scripts precisa fazer essa conversão.

## Como foi feito

1. **Tokenizer.** O Manacá baixa a caixa de todo texto. O v4 representa a maiúscula e a ausência de espaço com 3 tokens próprios, de modo que cada palavra tem uma forma só: `class Comentario` e `um comentario` usam a mesma peça `▁comentario`. É o esquema de [Land e Meister](https://arxiv.org/html/2608.08847), estendido para identificadores de código.
2. **Pré-treino continuado** em três etapas, todas na RTX 3090: código Python em português colhido do GitHub, Wikipédia e documentação do FastAPI (35 milhões de tokens); depois mais 65 milhões de tokens do [GigaVerbo-v2](https://huggingface.co/datasets/Polygl0t/gigaverbo-v2), já com o marcador; depois a mesma base com o marcador de espaço. Cerca de 9 horas cada uma das duas últimas.
3. **Dataset de FastAPI** gerado por código: 7.932 pedidos sobre 2.339 nomes de entidade (73 fixos, 266 extras e 2.000 colhidos), cada resposta um arquivo completo que roda. 12 entidades ficam fora do treino para o teste.
4. **SFT** com LoRA (QLoRA 4 bits), juntando os pedidos de FastAPI com 12.815 exemplos da receita de instrução do [Manacá-1B-Instruct](https://github.com/Instituto-IA-LNCC/manaca-1b-instruct) (Alpaca-PT, Aya, OASST1, tradução e resumo), no template deles. 3 épocas, 4 h 22.

O passo a passo completo, com cada experimento que não deu certo e por quê, está em `docs/treino_cased.md`.

## Reproduzir o treino

Na RTX 3090, depois de `scripts/sync_3090.sh` a partir desta máquina, na ordem:

| Script | O que faz | Tempo aproximado |
| --- | --- | --- |
| `scripts/rodar_tudo_3090.sh` | cria o ambiente Python | minutos |
| `scripts/rodar_cpt_grande_3090.sh` | colhe código e Wikipédia, CPT de 35M tokens | 5 h |
| `scripts/rodar_v3_3090.sh` | colhe o GigaVerbo, CPT com o marcador de caixa | 11 h |
| `scripts/rodar_v4_3090.sh` | CPT com o marcador de espaço | 12 h |
| `scripts/rodar_instruct_3090.sh` | SFT com conversa e FastAPI, e as duas avaliações | 4 h 40 |

Os dados de instrução vêm do `sft/prepare_data.py` do LNCC (fontes da v2) e de `scripts/misturar_sft.py`.

## Limitações conhecidas

- **O nome de entidade parecido com uma do treino pode sair trocado.** Os 4 erros do teste foram `endereco` virando `Enderecamento`, que é um nome do treino. A correção (tirar do treino nomes que começam com uma entidade de teste) ainda não foi feita.
- **Os campos são sorteados, não escolhidos pelo sentido.** "Documento" pode sair com `peso` e `altura`. O modelo também não entende pedidos com campos explícitos ("com os campos titulo e autor"), porque o treino nunca teve esse formato.
- **Sempre entrega o arquivo inteiro,** mesmo quando o pedido é "só a rota".
- **Não sabe explicar o FastAPI:** aprendeu a escrever o código, não a falar sobre a biblioteca.
- **Na conversa, tem os defeitos de um modelo de 1,7 bilhão de parâmetros:** repete frases, erra fatos ("15 vezes 4 é 40 vezes 4") e inventa detalhes em resumos.
- **Testado só no formato do dataset:** pedidos de CRUD em FastAPI. Pedidos muito diferentes não foram medidos.

## Licença

O Manacá-1B base é CC BY 4.0. A receita de instrução do LNCC inclui Alpaca-PT (CC BY-NC) e XLSum (CC BY-NC-SA), e o pré-treino inclui a Wikipédia (CC BY-SA 3.0). Por isso o modelo entregue, como o próprio Manacá-1B-Instruct, só pode ser usado para fins não comerciais.
