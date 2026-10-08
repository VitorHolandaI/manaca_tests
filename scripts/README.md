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
| `gen_dataset.py` | Gera o dataset Alpaca-PT **cased** e determinístico (`nome_de_classe` para classe, `capitalizar_frase` para instrução, docstring em toda rota) (**75 entidades** × 5 operações, mais 266 extras com poucos exemplos × variações de frase e comentário) e separa **treino** e **teste**: as entidades de `ENTIDADES_TESTE` (`comentario`, `endereco`) ficam inteiras fora do treino. O pool de entidades é grande de propósito — com 6 entidades o modelo decorou uma tabela de consulta em vez de aprender a copiar o nome, e `rotas`/`funcoes` deram 0/32 no primeiro fine-tune. Cada entidade tem um conjunto de campos fixo, ciclado de `CONJUNTOS_CAMPOS`, que vale para as entidades de teste. `exemplos_teste_amplo` escreve `data/fastapi_ptbr_teste_amplo.jsonl` (364 exemplos): o split de teste original mais todas as combinações das 10 entidades de `NOMES_TESTE_AMPLO`. Como esses nomes nunca entram no treino, os adapters já treinados podem ser medidos nele sem refazer SFT (`eval_code.py --ref data/fastapi_ptbr_teste_amplo.jsonl`). Com `scripts/nomes_colhidos.tsv` presente (`--nomes-colhidos`), `exemplos_colhidos` dá 2 exemplos a cada nome colhido e recusa com erro qualquer nome já usado, para que um nome de teste nunca vaze pro treino. Sem o arquivo, avisa e segue. Além delas, `exemplos_extras` acrescenta ao treino 6 exemplos sorteados de cada um dos 266 nomes de `nomes_entidades.NOMES_EXTRAS` (3932 exemplos no total), no fim da lista para não mexer no shuffle. No treino, `_sortear_campos` sorteia um conjunto por exemplo, com um `Random` próprio (`seed + 1`), para que o shuffle e o split de teste fiquem **idênticos** ao medido antes (sha256 `44be4950…`, travado em teste). O sorteio veio da rodada de 2026-10-01, em que o modelo escreveu `class Membro` num pedido de comentário; a hipótese é que campo preso à entidade ensina a lembrar a entidade pelos campos em vez de copiar o nome da instrução. Não depende de nada externo. Escreve os dois JSONL + `manifest.json` com contagens, seed e sha256. O código gerado é exercitado como app real em `tests/test_gen_dataset.py` (202 testes, CRUD completo em cada entidade). |
| `harvest_fastapi_docs.py` | Clona o `fastapi/fastapi` (MIT) em profundidade 1 e colhe os pares **prosa oficial pt-BR → código do `docs_src`**. O join é determinístico: cada página em `docs/pt/docs/` cita o arquivo de código por caminho (`{* ../../docs_src/x.py *}`). Só toca a rede — nada é traduzido por LLM, a prosa já é humana (tradução oficial da comunidade). Só usa arquivos que passam em `ast.parse`, e grava `manifest_oficial.json` com o commit de origem. |
| `ptbrificar_identificadores.py` | Renomeia identificadores do inglês para português usando **`ast`, não texto**. Essa distinção é o ponto: em `FastAPI(description="x")` o `description` é um `ast.keyword` e nunca é tocado, enquanto em `class Item(BaseModel): description: str` ele é um `ast.Name` e vira `descricao`. `kwargs` só são renomeados quando o chamador é uma classe definida no próprio exemplo (`Item(name=...)` → `Item(nome=...)`), nunca quando são API do FastAPI (`responses=`, `lifespan=`). Detecta **colisões por arquivo** — se `read_main` e `main` fossem os dois para `principal`, ambos voltam ao original. |
| `subword_sampling.py` | **Subword regularization** (Kudo 2018; Provilkov 2020): sorteia segmentações Unigram para o mesmo texto, o que quebra a dependência de uma tokenização canônica. Não havia como usar: o `tokenizers` expõe `nbest_size`/`alpha` no `Unigram` e **não os usa** (testado: mudar não muda a saída), e o `tokenizer.model` do SentencePiece tem `enable_sampling` mas seu normalizador `nmt_nfkc_cf` troca toda quebra de linha por espaço — `encode("a\nb") == encode("a b")` —, o que torna Python multi-linha impossível. Então a amostragem é feita aqui, sobre o vocabulário do `tokenizer.json`. `viterbi()` reproduz o tokenizer oficial e é o portão de sanidade: se divergir, os scores ou o byte-fallback estão errados. |
| `reparar_identificadores.py` | Pós-processa o que o treino **não conserta** (medido: LoRA, subword e full FT falharam). Duas passadas determinísticas, sem GPU: (1) tira acento de identificador, porque o prior ortográfico português vaza em cima da convenção ASCII — o full FT produziu `comentárioresposta`; (2) nome **usado e não definido em lugar nenhum** é typo, e o alvo é o nome definido mais próximo por distância de edição. Guardas: distância ≤ 3 **e** prefixo comum ≥ 4, senão um nome que legitimamente vem de fora seria mapeado errado — e o código passaria a compilar errado em silêncio. Também corrige o sufixo de nomes declarados (`deletar_comentar` → `deletar_comentario`), que é o que o marcador do eval exige. |
| `molde_fastapi.py` | O texto do arquivo FastAPI + SQLAlchemy que vira a resposta de cada exemplo: `Campo`, `Entidade`, `nome_de_classe` e `montar_saida` (cabeçalho, model, sessão, schemas e uma rota por operação). Saiu do `gen_dataset.py` em 2026-10-02 sem mudar um byte da saída (os três sha256 do dataset ficaram iguais). O `gen_dataset.py` importa e reexporta esses nomes, então `tests/app_de_referencia.py` e `test_sandbox_gerado.py` continuam importando de lá. |
| `colher_nomes_classe.py` | Roda na 3090, onde estão os JSONL colhidos. Junta duas fontes de nomes: classes de uma palavra só do código colhido (`class Fatura(`) que aparecem pelo menos `--minimo-wiki` vezes (padrão 20) como palavra na Wikipédia colhida, e palavras da própria Wikipédia com sufixo de substantivo (`-cao`, `-mento`, `-dade`, `-agem`...). `parece_entidade` descarta infinitivo, particípio, plural e final que o português quase não usa. A primeira colheita, só com classes e sem esse filtro, deu 422 nomes com `eliminar`, `object`, `emitido` e `visitas`. Fica com os `--maximo` (padrão 2000) mais frequentes, exclui todo nome já usado por `gen_dataset.py` e `nomes_entidades.py`, inclusive os de teste, e gera o plural pelas regras regulares. Escreve `scripts/nomes_colhidos.tsv`, que é versionado porque o dataset depende dele. |
| `nomes_entidades.py` | Só dados: os nomes de entidade do dataset de SFT, no singular e no plural. `NOMES_ENTIDADES` (75, as 8 primeiras são as originais) ganha todas as combinações de operação e frase; `NOMES_EXTRAS` (266) ganha só `EXEMPLOS_POR_EXTRA` exemplos cada. `carregar_nomes_colhidos` lê o TSV de `colher_nomes_classe.py` e recusa linha fora do formato apontando arquivo e número da linha. `NOMES_TESTE_AMPLO` (10) nunca entra no treino e forma, com `comentario` e `endereco`, o teste amplo de 12 entidades. Importado só por `gen_dataset.py`. Os extras vieram da rodada de 2026-10-02: o modelo escreveu `Endereco` certo em 16 de 16 e `Comentario` em 0 de 16 (saiu `Perfil`, `Paciente`), porque `comentario` e `▁Com enta rio` não têm peça em comum e o SFT só ensinava a tradução entidade por entidade. |
| `restore_case.py` | Restaura o caso canônico de bibliotecas (`fastapi`→`FastAPI`, `httpexception`→`HTTPException`, `true`→`True`…). É **token-aware** (usa `tokenize`), então não toca em strings, comentários nem no nome do módulo em `from fastapi import ...`. `nomes_a_restaurar()` diz quais tokens mudam; `eval_code.py` importa as duas funções. É um **dicionário de 15 nomes**, não um restaurador geral: não toca em prosa nem em nome de entidade. |
| `build_tokenizer_v2.py` | Constrói o tokenizer v2 atacando as duas raízes no nível do alfabeto: (1) remove o normalizer `Lowercase` e appenda maiúsculas (`F`, `Á`, `FastAPI`…), abrindo o canal de caso; (2) appenda a **forma sem espaço** de toda palavra que só existia como `▁palavra`, o que mata o garfo que troca `deletar_comentario` por `deletar_comentar`. As 64.000 peças originais **mantêm o id** (só appenda), então o embedding treinado continua válido e só as linhas novas precisam de init. Medido no dataset: `-2,8%` de tokens e round-trip 2336/2336. |
| `expandir_vocab.py` | Ponte entre o tokenizer v2 e o modelo. `montar_pasta_tokenizer_v2` monta `models/manaca-tokenizer-v2/` (config oficial + v2 como `tokenizer.json`) e o CLI confere com o transformers que ela carrega 115148 tokens e faz round-trip. `expandir_modelo` faz o resize das embeddings de entrada **e** de saída (não são amarradas) e inicia cada linha nova pela média das peças que o tokenizer antigo usava (`A` herda `▁a`); modelo já expandido passa intacto, e matriz com padding (o base tem 64128 linhas para 64000 peças) é aceita. Recusa v2 cujas 64000 primeiras peças divergem do tokenizer do modelo. Importado por `pretreino_continuado.py` e `train_qlora.py`. |
| `montar_corpus_cpt.py` | Monta `<saida>/{treino,val}.jsonl` a partir do checkout MIT do FastAPI e, com `--extra`, dos JSONL colhidos (licença preservada por documento e somada no manifest): `fastapi/` e `docs_src/` (código) + `docs/pt/` (prosa pt-BR, sem as diretivas `{* ... *}` do mkdocs). Importa de `gen_dataset.py` as entidades e instruções do teste amplo (que contém o split de teste) e **descarta** qualquer documento que as contenha. O CPT grande de 2026-10-01 foi montado antes do teste amplo existir, então só travou `comentario` e `endereco`. A val sai por fonte, pelos menores hashes de caminho, então é determinística e cobre código e prosa. |
| `colher_codigo_ptbr.py` | Varre os 880 shards do `codeparrot/github-code-clean` (314 GB de download, ~1,5 GB em disco no pico: cada shard vai para um diretório temporário e é apagado) e fica só com Python de licença permissiva que tenha ≥ 5 marcadores portugueses **fortes** (`senha`, `usuario`, `retorna`; nunca `do`/`as`/`os`, que colidem com inglês). Rendimento medido nos shards 0 e 1: ~30k caracteres por shard, ~0,1% do Python. Deduplica por sha256 e retoma de onde parou (`<saida>.feitos`). Alimenta o `--extra` do `montar_corpus_cpt.py`. |
| `colher_wikipedia_ptbr.py` | Prosa pt da `wikimedia/wikipedia` (dump `20231101.pt`) até um teto de caracteres, pulando verbete curto. É o que ensina maiúscula em português (início de frase, nome próprio). **Licença CC BY-SA 3.0**, registrada por documento. |
| `caixa_marcada.py` | Representa a maiúscula como marcador, sem perda: `Comentario` vira `<|maj|>comentario`, `HTTPException` vira `<|mai|>http<|maj|>exception`, camelCase quebra em partes. O marcador vai **antes** do espaço que precede a palavra (`class<|maj|> comentario`), para a palavra continuar sendo a peça `▁comentario` da instrução. Letra que não volta igual (`İ`, `ß`) fica literal. Com `fronteira=True` (tokenizer v4), toda palavra sai na forma com espaço e `<|cola|>` diz que o espaço não existia: `criar_agendamento` vira `<|cola|> criar_<|cola|> agendamento`, e a palavra é sempre a peça `▁agendamento`. No v3, a forma colada era outra peça e perdia para `agenda`, entidade do treino. `codificar_para(tok, texto)` escolhe o modo pelo vocabulário (`usa_fronteira`). Esquema de Land e Meister (arXiv 2608.08847), estendido para identificador. `usa_caixa_marcada(tok)` diz se o tokenizer é o v3; é por ela que `pretreino_continuado.py`, `train_qlora.py` e `eval_code.py` ligam a codificação sozinhos. Veio da rodada de 2026-10-03: com a maiúscula dentro da peça (v2), o Manacá acertou a classe de entidade inédita em 32% dos pedidos e o Qwen2.5-0.5B, no mesmo dataset, em 100%. |
| `build_tokenizer_v3.py` | Monta `models/manaca-tokenizer-v3`: as 64.000 peças originais (NFKC + Lowercase, como o Manacá), as 43.120 peças nuas minúsculas do v2 (`comentario` sem `▁`, para `deletar_comentario`) e os 2 marcadores no fim (ids 107120 e 107121), sem nenhuma peça com maiúscula. Os marcadores entram com `normalized=False` (o Lowercase não pode tocar neles) e `special=False` (o decode com `skip_special_tokens` tem de manter). Confere a ida e volta antes de gravar. Usa `expandir_vocab.montar_pasta_tokenizer_v2` para montar a pasta. |
| `build_tokenizer_v4.py` | Monta `models/manaca-tokenizer-v4`: o tokenizer **original** do Manacá mais `<|maj|>`, `<|mai|>` e `<|cola|>` (ids 64000 a 64002), 64.003 tokens. Caixa e espaço antes eram as duas fontes de formas duplicadas da mesma palavra (Land e Meister); com as duas viradas marcador, as peças nuas do v2/v3 deixam de ser precisas. Confere a ida e volta antes de gravar. |
| `remapear_vocab.py` | Leva o CPT grande (v2, 115.148 linhas) para o v3 (107.122) copiando cada linha de embedding e de `lm_head` **pela peça**, não pelo id. Só as peças declaradas como `novos` são iniciadas: marcador de caixa pela média das linhas das letras maiúsculas da origem (`A`, `▁B`...), outro token (`<|cola|>`) pela média geral. Marcador que a origem já tem (v3 para v4) é copiado como qualquer peça. Chamado por `pretreino_continuado.py` quando o tokenizer é o v3 e o vocabulário do modelo é outro. |
| `colher_gigaverbo.py` | Prosa pt do GigaVerbo-v2 (`Polygl0t/gigaverbo-v2`, subconjunto `edu_high`, o corpus do Tucano 2), para o CPT do v3 ter texto com maiúscula de sobra. Lê `--shards` (padrão 4) dos 56 shards, espalhados, sorteia os row groups dentro de cada um (o shard vem agrupado por fonte), filtra `edu_score >= 3,5` e `toxic_score <= 1,5` e para no teto `--max-tokens` (pela `token_count` do próprio GigaVerbo). Licença por fonte: cada documento leva `gigaverbo:<fonte>` no campo `licenca`. |
| `rodar_v3_3090.sh` | Pipeline do v3 na 3090: colhe o GigaVerbo, monta `data/cpt_v3/` (código, wiki e GigaVerbo), smoke de 2 passos, CPT a partir do `manaca-cpt-grande` com o v3, SFT só-LoRA e eval no teste amplo (`results/amplo_v3.json`). Para no primeiro erro. |
| `rodar_v4_3090.sh` | Pipeline do v4 na 3090: smoke de 2 passos, CPT a partir do `manaca-cpt-v3` com o v4 (só `<|cola|>` nasce), SFT só-LoRA e eval no teste amplo (`results/amplo_v4.json`). Reusa `data/cpt_v3`, porque o corpus guarda texto cru e a codificação acontece ao carregar. |
| `rodar_cpt_grande_3090.sh` | Pipeline do CPT grande na 3090: colhe código e wiki, monta `data/cpt_grande/`, CPT a partir do base, SFT só-LoRA e eval sem remendo. Apaga os `checkpoint-*` (estado do otimizador) porque o disco de lá está a 95%. Pode ser rodado de novo: a colheita retoma. |
| `pretreino_continuado.py` | Com o tokenizer v3, `preparar_textos` codifica cada documento com `caixa_marcada` (e conta os que já tinham o marcador literal, que saem) e `_carregar` remapeia o modelo de origem peça por peça (`remapear_vocab.py`); se o modelo já tem o vocabulário do v3, não mexe. Pré-treino continuado, full FT bf16 com Adam 8-bit: carrega o base, expande para o v2 via `expandir_vocab.py`, empacota o corpus em blocos de 2048 separados por EOS e treina com loss em todo token. Mede a val loss **antes** do primeiro passo (linha de base do esquecimento) e grava `val_loss.json`. Salva modelo + tokenizer v2, que é o `--model` do `train_qlora.py`. **Não executado** (sem NVIDIA). |
| `train_qlora.py` | Com o tokenizer v3, `textos_de_treino` codifica prompt e resposta com `caixa_marcada`, e `marca_de_resposta` codifica o `### Resposta:` do mesmo jeito, senão o collator não acharia a resposta. O EOS entra cru depois da codificação: codificado com fronteira, `</s>` virava `</<|cola|> s>`, o modelo nunca via o EOS e não parava de gerar (compile_rate 0,0 na primeira rodada do v4, 2026-10-04). QLoRA 4-bit + LoRA r=32 no **base**, com o stack de `../requirements-3090.txt` (trl 0.12.2) (`AIInstituteLNCC/manaca-1b-base`, CC BY 4.0). Fiel ao `sft/train.py` oficial: template Alpaca-PT e loss só na resposta, com `check_collator` que aborta se o marcador não casar (senão a loss vira NaN). Com `--tokenizer` (v2) expande o modelo e treina `embed_tokens`/`lm_head` (`modules_to_save`). Partindo de um checkpoint já expandido (CPT) o padrão é **só LoRA**, e `--treinar-vocab` liga o vocabulário mesmo assim. Medido em 2026-09-30, a partir do CPT pequeno: só LoRA empacou em loss 0.21, `case_rate` 0.09 e `compile_rate` 0.0 (`Column` e `▁column` nasceram com a mesma linha); com `--treinar-vocab`, `case_rate` 1.0, mas `rotas` 0.0. O `lm_head` treinável rebaixa toda peça que o SFT não pede, e a entidade inédita some (`deletar_ comentario`, `class Perfil`). Por isso o empate tem de ser desfeito no CPT, com corpus grande, e não no SFT. `garantir_pad_distinto` troca o pad por um token especial já existente quando ele é o próprio EOS, como no Qwen2.5 e no Qwen3 base (`<|endoftext|>` nos dois); o collator do transformers 4.46.3 põe -100 em todo label igual ao pad, e o modelo nunca aprenderia a parar. Foi preciso para o controle com `Qwen/Qwen2.5-0.5B` (2026-10-03); o Qwen3 ficou de fora porque exige transformers 4.51. `--subword-alpha` com v2 é recusado (a amostragem só conhece o vocab oficial). **Não executado na máquina de desenvolvimento** (sem NVIDIA) — rodar na 3090 com `--smoke` primeiro. |
| `eval_code.py` | Com o tokenizer v3, `prompts_para` codifica o prompt e `desfazer_marcador` devolve a maiúscula à saída antes de pontuar. `--base-4bit` carrega o modelo base em NF4, igual ao `train_qlora.py` no treino do adapter; sem ele o base vai em bf16. Entrou porque no v4 o eval em bf16 trocou `connect_args={` por `connect_args=Depends` em todos os pedidos (runs 0,0): depois de `=`, o `{` é o byte cru `<0x7B>` (o vocabulário do Manacá não tem a peça), e `<|cola|>` é o token mais comum nesse lugar. Esse caminho só roda com GPU e não tem teste automatizado. Avaliação determinística, **sem juiz LLM**: taxa de compilação (`ast.parse`), acerto dos marcadores estruturais extraídos da referência (rota, função, `status_code`), `case_rate` (saída que já nasce com o caso de Python), `docstring_rate` (toda rota com docstring capitalizada), `entidade_rate` (a classe do model SQLAlchemy é a da referência; `entidade_certa` pega a troca por uma entidade do treino, que compila, roda e passa nos outros marcadores) e quantos exemplos precisaram de restauro. `--sem-restauro` pontua o texto cru — é o eval sem remendo. Por padrão avalia o split de **teste** e gera até 1024 tokens (a maior referência tem 697). Saída que não tokeniza conta como falha, não derruba o eval. A geração em lote corta cada saída no primeiro EOS ou pad (`cortar_no_fim`): o pad do Qwen (`<|fim_pad|>`) não é token especial e o `skip_special_tokens` deixaria ele colado no código. |
| `sandbox_gerado.py` | Chamado por `eval_code.py --executar`: sobe a saída num `TestClient` e chama cada rota declarada; reprova em exceção ou 5xx. O payload vem da classe Pydantic de entrada do próprio código gerado (a que termina em `criar`, sem olhar caixa — `UsuarioCriar` e `usuariocriar`). |
| `sync_3090.sh` | rsync do projeto para a máquina da 3090 (`usuario@maquina-3090:~/manaca_build/`, sobrescrevível por `DESTINO=`). Deixa os pesos de fora, porque o base vem do Hub, mas leva `models/manaca-tokenizer-v2/`, `-v3/` e `-v4/`, que `pretreino_continuado.py` e `train_qlora.py` leem do disco, e `data/upstream/fastapi` (56 MB), que o `montar_corpus_cpt.py` remonta na 3090. Aborta se alguma dessas pastas ainda não foi montada (`expandir_vocab.py`, `build_tokenizer_v3.py`, `build_tokenizer_v4.py`). Aceita `--dry-run`. |
| `rodar_tudo_3090.sh` | Roda na 3090, sem supervisão, o pipeline depois do `sync_3090.sh`: venv (`requirements-3090.txt` + `requirements-dev.txt`, que o `eval --executar` exige), smoke do CPT e do SFT, CPT, SFT e eval sem remendo. Para no primeiro erro. Cada etapa grava em `logs/<etapa>.log` e o resultado sai em `results/cased.json`. Val loss que sobe no CPT gera aviso, mas não aborta. |
| `chat_fastapi.py` | Chat no terminal com o Manacá-FastAPI (`checkpoints/manaca-cpt-v4` + o adapter `checkpoints/manaca-fastapi-v4`). Reaproveita do `eval_code.py` o prompt, a codificação de caixa (`prompts_para`, `desfazer_marcador`), o corte no EOS e a leitura do base declarado no adapter, para o modelo ver o mesmo formato do eval. Roda em CPU ou na GPU Intel (`xpu`), com o ambiente de `../requirements-chat.txt`. `--prompt` responde um pedido só; `--sem-nf4` carrega o base em bf16 puro. |
| `fundir_adapter.py` | Funde o adapter do SFT no base **já arredondado para NF4** (`nf4_simulado.py`) e salva um modelo único em safetensors (`models/manaca-fastapi-v4-fundido`), com o tokenizer do adapter e o `tokenizer.model` original. Fundido sobre o base em bf16, o v4 voltaria a errar o cabeçalho; sobre o NF4 simulado, o modelo único carrega os pesos que o adapter viu no treino. É a entrada do `convert_hf_to_gguf.py` do llama.cpp, que gera `models/manaca-fastapi-v4-f16.gguf`. |
| `chat_gguf.py` | Chat com o GGUF servido pelo `llama-server` (que roda na GPU Intel, build SYCL). O llama.cpp **não tokeniza igual ao treino**: o Manacá é Unigram com byte fallback, o modo Unigram do llama.cpp (`t5`) não tem byte fallback (`\n` vira `<unk>`) e o modo SentencePiece (`llama`) segmenta por fusões; em 100 textos, 6 saíram idênticos ao transformers em cada modo (2026-10-06). Com `--oficial`, conversa com o Manacá instruct original do LNCC (`models/manaca-instruct-ugm/F16-UGM.gguf`), com o tokenizer e o template do `chat_oficial.py`. Por isso o prompt vai ao `/completion` como lista de ids e os ids gerados voltam com `return_tokens`: tokenizar, codificar e decodificar ficam deste lado, com o tokenizer v4 do transformers. Só precisa de `transformers`, sem torch. O Ollama não serve: ele só aceita texto e tokeniza com o llama.cpp. Medido: mesma resposta do `chat_fastapi.py` (a menos dos comentários opcionais), app sobe na sandbox, cerca de 1 min por resposta contra mais de 10 em CPU. |
| `eval_gguf.py` | Gera, pelo `llama-server`, as respostas de uma amostra do teste amplo (`--por-entidade` pedidos de cada entidade, porque o começo do arquivo é só de comentário e endereço) e grava as respostas e a amostra usada (`.ref`), para o `eval_code.py --pred --ref` pontuar com a mesma métrica. Usa `chat_gguf.responder`. Serviu para escolher a quantização do GGUF. |
| `nf4_simulado.py` | Arredonda cada `nn.Linear` menos o `lm_head` para a grade NF4 em blocos de 64 e volta ao dtype original, em torch puro. Existe porque o adapter foi treinado sobre o base em NF4 e sobre o bf16 erra o cabeçalho (`connect_args=Depends`), e esta máquina não tem GPU NVIDIA para o bitsandbytes. Os 16 valores e o bloco vêm do `bitsandbytes` 0.45 (`get_4bit_type`, `quantize_4bit`); a dupla quantização do absmax não é simulada. Usado por `chat_fastapi.py`. |
| `md_para_docx.js` | Único arquivo em JavaScript da pasta: gera `docs/treino_cased.docx` a partir do `.md` (título, seções, parágrafos, listas, tabelas, `código` e links, só o que o doc usa). Usa a lib `docx` do npm, que não é dependência do projeto: `npm install --prefix /tmp/md docx && NODE_PATH=/tmp/md/node_modules node scripts/md_para_docx.js docs/treino_cased.md docs/treino_cased.docx` (o `require` procura a lib a partir da pasta do script, por isso o `NODE_PATH`). |
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

