# scripts/ — fine-tune de geração de código FastAPI em PT-BR

Esta pasta monta o ciclo **dados → treino → avaliação** para ensinar o Manacá-1B a
gerar FastAPI + SQLAlchemy com nomes e comentários em português. Tudo é livre: o
dado sintético é CC0, e o colhido é derivado de código **MIT** do próprio FastAPI.

## Restrição de origem: o tokenizer oficial é case-blind

O Manacá-1B não tem **nenhuma** peça maiúscula no vocabulário de 64k (ver
`gotchas/manaca-tokenizer-case-folding.md` no ai-memory, ou rode o teste em
`docs/data/README.md`). Até o dataset v1 o alvo era gerado 100% minúsculo e o
caso voltava no inference por `restore_case.py`.

**Desde o tokenizer v2 o contrato mudou:** `gen_dataset.py` gera o caso real de
Python (`class Usuario(Base)`, `HTTPException`, `True`/`None`) com docstring
capitalizada, e o treino tem de usar o v2 (`expandir_vocab.py`). O
`restore_case.py` continua existindo para avaliar os adapters antigos.

## Caminho para zerar o pós-processamento

Medido no split de teste: `precisaram_de_restauro = 60/60` e, sem `--reparar`,
`runs_rate` cai de `1.0` para `0.7188`. Ou seja, os dois pós-processadores são a
camada de caso e de grafia do sistema — o alvo minúsculo garante que o modelo
**nunca aprenda** caso sozinho. Remover os dois tem duas raízes distintas:

| raiz | por que acontece | correção |
|---|---|---|
| **caso** | normalizador `Lowercase` apaga a maiúscula **antes** de tokenizar; 0 peças maiúsculas | remover o normalizador + appendar maiúsculas |
| **grafia** | `▁comentario` é uma peça só; o identificador sem espaço estilhaça e o modelo troca por `comentar` (verbo) | appendar a forma sem espaço de cada palavra |

Ordem obrigatória, porque cada portão bloqueia o seguinte:

1. **tokenizer v2** (`build_tokenizer_v2.py`) — sem peça cased e sem forma sem
   espaço, o próprio dado não consegue carregar o que vem depois. Local, sem GPU.
2. **dataset cased** ✓ — `gen_dataset.py` gera caso real e docstring
   capitalizada em toda rota; a referência roda crua (sem restauro).
3. **corpus cased** ✓ — `montar_corpus_cpt.py`: código real
   (`data/upstream/fastapi`, MIT) + prosa pt-BR de `docs/pt`. A prosa é o que
   ensina início de frase e nome próprio, que o código não cobre. Hoje são só
   **277k tokens** (135 blocos de 2048): o pipeline está pronto, o volume não.
4. **eval sem remendo** ✓ — `--sem-restauro`, `case_rate`, `docstring_rate`. No
   ground truth do teste: `runs_rate` 1.0 com `--executar --sem-restauro` e sem
   `--reparar`.
5. **pré-treino continuado** — `pretreino_continuado.py` no `manaca-1b-base` (CC BY
   4.0), full FT, val loss medida antes do primeiro passo como freio. Pronto, não
   executado (GPU).
6. **SFT** — `train_qlora.py` a partir do checkpoint do passo 5. Pronto, não
   executado (GPU).

Risco declarado: a forma sem espaço generaliza para **combinação nova de palavras
conhecidas** (`historico_pagamento`), não para morfema inédito — o vocabulário não
é infinito. Nome inédito de verdade pode continuar precisando do `reparar`.

## Fluxo

Dois caminhos alimentam o mesmo formato Alpaca. O **sintético** dá volume na tarefa
exata (CRUD); o **colhido** dá os padrões canônicos da biblioteca, escritos pelo
Tiangolo e traduzidos oficialmente para pt-BR.

```
build_tokenizer_v2.py ──► tokenizer_v2.json ──► expandir_vocab.py ──► models/manaca-tokenizer-v2/
                                                  (resize + init por media)      │
montar_corpus_cpt.py ──► data/cpt/{treino,val}.jsonl ──► pretreino_continuado.py ◄┘
                                                              │ (full FT, RTX 3090)
                                                              ▼
gen_dataset.py ─┬─► data/fastapi_ptbr.jsonl ──────► train_qlora.py ──► checkpoints/
                └─► data/fastapi_ptbr_teste.jsonl        (RTX 3090)           │
                             │                                                 │
                             └──────────────► eval_code.py ◄───────────────────┘
                                                  │
                          restore_case.py (pulado com --sem-restauro) + sandbox_gerado.py (--executar)

harvest_fastapi_docs.py ──► data/fastapi_oficial.jsonl
  (clona o repo MIT e        (prosa oficial pt-BR -> codigo em ingles)
   resolve os includes)                 │
                                        ▼
                     ptbrificar_identificadores.py ──► data/fastapi_oficial_ptbr.jsonl
                        (renomeia identificador via ast)
```

## Arquivos

| Arquivo | Papel |
|---|---|
| `gen_dataset.py` | Gera o dataset Alpaca-PT **cased** e determinístico (`nome_de_classe` para classe, `capitalizar_frase` para instrução, docstring em toda rota) (**75 entidades** × 5 operações × variações de frase e comentário) e separa **treino** e **teste**: as entidades de `ENTIDADES_TESTE` (`comentario`, `endereco`) ficam inteiras fora do treino. O pool de entidades é grande de propósito — com 6 entidades o modelo decorou uma tabela de consulta em vez de aprender a copiar o nome, e `rotas`/`funcoes` deram 0/32 no primeiro fine-tune. Cada entidade recebe um conjunto de campos ciclado (`CONJUNTOS_CAMPOS`), para o modelo não decorar o formato dos campos. As 8 primeiras entidades são as originais, então o split de teste é **idêntico** ao medido antes. Não depende de nada externo. Escreve os dois JSONL + `manifest.json` com contagens, seed e sha256. O código gerado é exercitado como app real em `tests/test_gen_dataset.py` (202 testes, CRUD completo em cada entidade). |
| `harvest_fastapi_docs.py` | Clona o `fastapi/fastapi` (MIT) em profundidade 1 e colhe os pares **prosa oficial pt-BR → código do `docs_src`**. O join é determinístico: cada página em `docs/pt/docs/` cita o arquivo de código por caminho (`{* ../../docs_src/x.py *}`). Só toca a rede — nada é traduzido por LLM, a prosa já é humana (tradução oficial da comunidade). Só usa arquivos que passam em `ast.parse`, e grava `manifest_oficial.json` com o commit de origem. |
| `ptbrificar_identificadores.py` | Renomeia identificadores do inglês para português usando **`ast`, não texto**. Essa distinção é o ponto: em `FastAPI(description="x")` o `description` é um `ast.keyword` e nunca é tocado, enquanto em `class Item(BaseModel): description: str` ele é um `ast.Name` e vira `descricao`. `kwargs` só são renomeados quando o chamador é uma classe definida no próprio exemplo (`Item(name=...)` → `Item(nome=...)`), nunca quando são API do FastAPI (`responses=`, `lifespan=`). Detecta **colisões por arquivo** — se `read_main` e `main` fossem os dois para `principal`, ambos voltam ao original. |
| `subword_sampling.py` | **Subword regularization** (Kudo 2018; Provilkov 2020): sorteia segmentações Unigram para o mesmo texto, o que quebra a dependência de uma tokenização canônica. Não havia como usar: o `tokenizers` expõe `nbest_size`/`alpha` no `Unigram` e **não os usa** (testado: mudar não muda a saída), e o `tokenizer.model` do SentencePiece tem `enable_sampling` mas seu normalizador `nmt_nfkc_cf` troca toda quebra de linha por espaço — `encode("a\nb") == encode("a b")` —, o que torna Python multi-linha impossível. Então a amostragem é feita aqui, sobre o vocabulário do `tokenizer.json`. `viterbi()` reproduz o tokenizer oficial e é o portão de sanidade: se divergir, os scores ou o byte-fallback estão errados. |
| `reparar_identificadores.py` | Pós-processa o que o treino **não conserta** (medido: LoRA, subword e full FT falharam). Duas passadas determinísticas, sem GPU: (1) tira acento de identificador, porque o prior ortográfico português vaza em cima da convenção ASCII — o full FT produziu `comentárioresposta`; (2) nome **usado e não definido em lugar nenhum** é typo, e o alvo é o nome definido mais próximo por distância de edição. Guardas: distância ≤ 3 **e** prefixo comum ≥ 4, senão um nome que legitimamente vem de fora seria mapeado errado — e o código passaria a compilar errado em silêncio. Também corrige o sufixo de nomes declarados (`deletar_comentar` → `deletar_comentario`), que é o que o marcador do eval exige. |
| `restore_case.py` | Restaura o caso canônico de bibliotecas (`fastapi`→`FastAPI`, `httpexception`→`HTTPException`, `true`→`True`…). É **token-aware** (usa `tokenize`), então não toca em strings, comentários nem no nome do módulo em `from fastapi import ...`. `nomes_a_restaurar()` diz quais tokens mudam; `eval_code.py` importa as duas funções. É um **dicionário de 15 nomes**, não um restaurador geral: não toca em prosa nem em nome de entidade. |
| `build_tokenizer_v2.py` | Constrói o tokenizer v2 atacando as duas raízes no nível do alfabeto: (1) remove o normalizer `Lowercase` e appenda maiúsculas (`F`, `Á`, `FastAPI`…), abrindo o canal de caso; (2) appenda a **forma sem espaço** de toda palavra que só existia como `▁palavra`, o que mata o garfo que troca `deletar_comentario` por `deletar_comentar`. As 64.000 peças originais **mantêm o id** (só appenda), então o embedding treinado continua válido e só as linhas novas precisam de init. Medido no dataset: `-2,8%` de tokens e round-trip 2336/2336. |
| `expandir_vocab.py` | Ponte entre o tokenizer v2 e o modelo. `montar_pasta_tokenizer_v2` monta `models/manaca-tokenizer-v2/` (config oficial + v2 como `tokenizer.json`) e o CLI confere com o transformers que ela carrega 115148 tokens e faz round-trip. `expandir_modelo` faz o resize das embeddings de entrada **e** de saída (não são amarradas) e inicia cada linha nova pela média das peças que o tokenizer antigo usava (`A` herda `▁a`); modelo já expandido passa intacto, e matriz com padding (o base tem 64128 linhas para 64000 peças) é aceita. Recusa v2 cujas 64000 primeiras peças divergem do tokenizer do modelo. Importado por `pretreino_continuado.py` e `train_qlora.py`. |
| `montar_corpus_cpt.py` | Monta `<saida>/{treino,val}.jsonl` a partir do checkout MIT do FastAPI e, com `--extra`, dos JSONL colhidos (licença preservada por documento e somada no manifest): `fastapi/` e `docs_src/` (código) + `docs/pt/` (prosa pt-BR, sem as diretivas `{* ... *}` do mkdocs). Importa de `gen_dataset.py` as entidades e instruções de teste e **descarta** qualquer documento que as contenha. A val sai por fonte, pelos menores hashes de caminho, então é determinística e cobre código e prosa. |
| `colher_codigo_ptbr.py` | Varre os 880 shards do `codeparrot/github-code-clean` (314 GB de download, ~1,5 GB em disco no pico: cada shard vai para um diretório temporário e é apagado) e fica só com Python de licença permissiva que tenha ≥ 5 marcadores portugueses **fortes** (`senha`, `usuario`, `retorna`; nunca `do`/`as`/`os`, que colidem com inglês). Rendimento medido nos shards 0 e 1: ~30k caracteres por shard, ~0,1% do Python. Deduplica por sha256 e retoma de onde parou (`<saida>.feitos`). Alimenta o `--extra` do `montar_corpus_cpt.py`. |
| `colher_wikipedia_ptbr.py` | Prosa pt da `wikimedia/wikipedia` (dump `20231101.pt`) até um teto de caracteres, pulando verbete curto. É o que ensina maiúscula em português (início de frase, nome próprio). **Licença CC BY-SA 3.0**, registrada por documento. |
| `rodar_cpt_grande_3090.sh` | Pipeline do CPT grande na 3090: colhe código e wiki, monta `data/cpt_grande/`, CPT a partir do base, SFT só-LoRA e eval sem remendo. Apaga os `checkpoint-*` (estado do otimizador) porque o disco de lá está a 95%. Pode ser rodado de novo: a colheita retoma. |
| `pretreino_continuado.py` | Pré-treino continuado, full FT bf16 com Adam 8-bit: carrega o base, expande para o v2 via `expandir_vocab.py`, empacota o corpus em blocos de 2048 separados por EOS e treina com loss em todo token. Mede a val loss **antes** do primeiro passo (linha de base do esquecimento) e grava `val_loss.json`. Salva modelo + tokenizer v2, que é o `--model` do `train_qlora.py`. **Não executado** (sem NVIDIA). |
| `train_qlora.py` | QLoRA 4-bit + LoRA r=32 no **base**, com o stack de `../requirements-3090.txt` (trl 0.12.2) (`AIInstituteLNCC/manaca-1b-base`, CC BY 4.0). Fiel ao `sft/train.py` oficial: template Alpaca-PT e loss só na resposta, com `check_collator` que aborta se o marcador não casar (senão a loss vira NaN). Com `--tokenizer` (v2) expande o modelo e treina `embed_tokens`/`lm_head` (`modules_to_save`). Partindo de um checkpoint já expandido (CPT) o padrão é **só LoRA**, e `--treinar-vocab` liga o vocabulário mesmo assim. Medido em 2026-09-30, a partir do CPT pequeno: só LoRA empacou em loss 0.21, `case_rate` 0.09 e `compile_rate` 0.0 (`Column` e `▁column` nasceram com a mesma linha); com `--treinar-vocab`, `case_rate` 1.0, mas `rotas` 0.0. O `lm_head` treinável rebaixa toda peça que o SFT não pede, e a entidade inédita some (`deletar_ comentario`, `class Perfil`). Por isso o empate tem de ser desfeito no CPT, com corpus grande, e não no SFT. `--subword-alpha` com v2 é recusado (a amostragem só conhece o vocab oficial). **Não executado na máquina de desenvolvimento** (sem NVIDIA) — rodar na 3090 com `--smoke` primeiro. |
| `eval_code.py` | Avaliação determinística, **sem juiz LLM**: taxa de compilação (`ast.parse`), acerto dos marcadores estruturais extraídos da referência (rota, função, `status_code`), `case_rate` (saída que já nasce com o caso de Python), `docstring_rate` (toda rota com docstring capitalizada) e quantos exemplos precisaram de restauro. `--sem-restauro` pontua o texto cru — é o eval sem remendo. Por padrão avalia o split de **teste** e gera até 1024 tokens (a maior referência tem 697). Saída que não tokeniza conta como falha, não derruba o eval. |
| `sandbox_gerado.py` | Chamado por `eval_code.py --executar`: sobe a saída num `TestClient` e chama cada rota declarada; reprova em exceção ou 5xx. O payload vem da classe Pydantic de entrada do próprio código gerado (a que termina em `criar`, sem olhar caixa — `UsuarioCriar` e `usuariocriar`). |
| `sync_3090.sh` | rsync do projeto para a máquina da 3090 (`usuario@maquina-3090:~/manaca_build/`, sobrescrevível por `DESTINO=`). Deixa os pesos de fora, porque o base vem do Hub, mas leva `models/manaca-tokenizer-v2/`, que `pretreino_continuado.py` lê do disco, e `data/upstream/fastapi` (56 MB), que o `montar_corpus_cpt.py` remonta na 3090. Aborta se essa pasta ainda não foi montada por `expandir_vocab.py`. Aceita `--dry-run`. |
| `rodar_tudo_3090.sh` | Roda na 3090, sem supervisão, o pipeline depois do `sync_3090.sh`: venv (`requirements-3090.txt` + `requirements-dev.txt`, que o `eval --executar` exige), smoke do CPT e do SFT, CPT, SFT e eval sem remendo. Para no primeiro erro. Cada etapa grava em `logs/<etapa>.log` e o resultado sai em `results/cased.json`. Val loss que sobe no CPT gera aviso, mas não aborta. |
| `chat_oficial.py` | Cópia do `scripts/chat.py` oficial do repo da LNCC, com uma linha patchada para preferir `xpu` (a iGPU Intel) em vez de `cuda`. É a referência de inferência oficial; não faz parte do fine-tune. |

## Comandos

```bash
# 1. gerar treino + teste (CPU, segundos)
python scripts/gen_dataset.py

# 2. colher os exemplos oficiais do FastAPI (MIT, precisa de rede)
python scripts/harvest_fastapi_docs.py --out data/fastapi_oficial.jsonl

# 3. traduzir os identificadores colhidos para portugues
python scripts/ptbrificar_identificadores.py \
    --in data/fastapi_oficial.jsonl --out data/fastapi_oficial_ptbr.jsonl

# 4. baseline zero-shot no split de teste (roda na iGPU via XPU)
python scripts/eval_code.py --model AIInstituteLNCC/manaca-1b-base

# 5. pasta do tokenizer v2 + corpus do pre-treino continuado (CPU)
python scripts/expandir_vocab.py
python scripts/montar_corpus_cpt.py --commit a3d205bf19640528718cb4f05ab77f4dfca6ad9a

# 6. pre-treino continuado (RTX 3090, depois de: uv pip install -r requirements-3090.txt)
python scripts/pretreino_continuado.py --model AIInstituteLNCC/manaca-1b-base \
    --out checkpoints/manaca-cpt-cased --smoke

# 7. SFT a partir do checkpoint do CPT (ja tem o vocab do v2)
accelerate launch scripts/train_qlora.py \
    --model checkpoints/manaca-cpt-cased \
    --data data/fastapi_ptbr.jsonl \
    --out checkpoints/manaca-fastapi-cased --smoke

# 8. eval sem remendo
python scripts/eval_code.py --model checkpoints/manaca-fastapi-cased \
    --executar --sem-restauro
```

## Invariantes do harvester (verificados nos 289 exemplos reais)

| Checagem | Resultado |
|---|---|
| `ast.parse` depois da renomeação | 289/289 válidos, 0 revertidos |
| `kwargs` de API do FastAPI (`response_model=`, `status_code=`, `responses=`, `tags=`, `lifespan=`, `directory=`…) | 0 alterados |
| chamada a classe **não-local** com `kwarg` em português | 0 violações |
| imports antes × depois | 61 módulos, idênticos |

Renomeações: 1463. Colisões evitadas: 60.

O caso que mostra por que a decisão é via `ast` e não por texto:

```diff
-description = """
+descricao = """
-    description=description,
+    description=descricao,
```

O `kwarg` do FastAPI continua `description=`; só o **valor** vira `descricao`. Uma
troca textual teria produzido `descricao=descricao`, que o FastAPI ignora em
silêncio.

## Experimentos de fine-tune (medidos)

Mesmo split de teste (`comentario` e `endereco`, inéditos no treino), mesma LoRA
r=32, 3 épocas, 2336 exemplos exceto onde indicado. `runs_rate` só existe a partir
do `--executar`.

| run | `compile` | `runs` | `marker` | `rotas` | `funcoes` |
|---|---|---|---|---|---|
| base (instruct, 3 exemplos) | 0.0 | — | 0.0 | — | — |
| 6 entidades / 192 exemplos | 0.3125 | — | 0.0 | **0.0** | **0.0** |
| **75 entidades / 2336 exemplos** | **1.0** | **0.7188** | 0.6875 | **0.9688** | 0.6875 |
| 75 entidades + subword α=0,2 | 1.0 | 0.5938 | **0.0** | 0.1875 | **0.0** |
| 75 entidades + `--full` | 1.0 | 0.5938 | 0.5938 | 0.875 | 0.5938 |
| **75 entidades + `--reparar`** | **1.0** | **1.0** | **0.8125** | 0.9688 | 0.8125 |

**O resultado final:** `runs_rate` **1.0** — as 32 saídas, para as duas entidades
inéditas, rodam. O reparo atuou em **9 das 32**, que eram exatamente as 9 que
davam `NameError`, e não mexeu em `rotas` nem `status`. O `funcoes` subiu de
0,6875 para 0,8125 pela regra do sufixo. As 6 que ainda erram o marcador mas
rodam são diferença de nomenclatura, não defeito funcional.

**O que o salto de 6 → 75 entidades provou:** a estrutura do programa generaliza
(rotas de 0 → 0,97), mas com 6 entidades o modelo decorou uma tabela de consulta
em vez de aprender a copiar o nome. 24 das 73 entidades de treino fragmentam em
identificador (`fornecedor` em 4 peças) e o modelo acerta todas as 24 — decorou.

**O que restou:** a grafia da entidade dentro do identificador. `▁comentario` é um
token único e `com|enta|rio` é outra segmentação; o modelo não deriva uma da outra
e recorre ao verbo concorrente (`deletar_comentar`). Erro **lexical**, 9 de 32.

**Subword regularization (`--subword-alpha`) foi medido e rejeitado.** Piorou tudo
e trocou erro lexical por erro **ortográfico**, inventando grafias inexistentes
(`enderico`, `endercriaresposta`). O motivo está em
`decisions/subword-regularization-rejeitada.md` no ai-memory: a técnica ganha
quando aplicada no pré-treino **e** no fine-tuning, e só no fine-tuning com dado
pequeno pode perder para a tokenização determinística.

Próximo ataque: **fine-tune completo**, porque a LoRA congela `embed_tokens` e
`lm_head`, e é nas embeddings que mora a composição de caractere (SpellingBee,
NAACL 2022) — que é exatamente o que falta para derivar `com|enta|rio` de
`▁comentario`. Use `--full`.

## O que sobrou: pós-processamento

Os três ataques de treino ao defeito de sub-palavra falharam (LoRA, subword,
full FT) — a evidência está na tabela acima e em `decisions/` no ai-memory. O que
resta é determinístico e roda sem GPU:

```bash
# avalia o adapter bom COM o reparo, e mostra quanto ele compra
uv run python scripts/eval_code.py \
    --model checkpoints/manaca-fastapi-qlora-75ent \
    --reparar --executar --out /tmp/ft_reparado.json
```

`--reparar` roda `reparar_identificadores.py` depois do `restore_case.py`, e o
relatório ganha o campo `reparados` — quantas saídas tiveram algum identificador
consertado. Sem `--reparar` os números ficam idênticos ao run medido antes.

## Baseline

A medição de 2026-09-28 (instruct, 3 exemplos, 0.0/0.0) foi descartada: ela
usava `max_new_tokens=512`, que corta as referências, e um template sem acento
que o instruct não viu no SFT. Refazer com o base no split de teste (comando 2).

