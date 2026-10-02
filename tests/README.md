# tests/ — prova que o dado de treino funciona, não só que compila

`ast.parse` só pega erro de sintaxe. Foi assim que o dataset original passou na
"calibração" com nota 1.0 enquanto 144 das 256 referências davam
`UnboundLocalError` em runtime e o restauro emitia `from FastAPI import ...`.
Esta pasta cobre essa lacuna: sobe o código que o modelo vai decorar como app
FastAPI real e percorre o CRUD. Não precisa de GPU nem de torch.

## Como rodar

O projeto não tem um comando de teste único (sem `pyproject.toml`/`Makefile`).
O comando usado é:

```bash
uv run --with-requirements requirements-dev.txt pytest tests/
```

## Arquivos

- `conftest.py` — põe `scripts/` no `sys.path`. Os scripts não são pacote:
  `eval_code.py` já importa `restore_case` pelo diretório, e os testes seguem a
  mesma convenção em vez de inventar um pacote.
- `app_de_referencia.py` — `AppDeReferencia` executa o código de uma saída
  **cru**, sem `restore_case` (desde o dataset cased a referência não pode
  depender de restauro), e embrulha o `app` num `TestClient`. Monta o payload a partir de `gen_dataset.Entidade.campos`. É
  usado só por `test_gen_dataset.py`. O escopo do `exec` **precisa** de um
  `__name__`: sem ele as classes nascem com `__module__ == "builtins"` e o
  pydantic deixa de honrar o `from_attributes` do modelo de resposta, o que
  derruba os 16 casos do CRUD com `ResponseValidationError`. Objetos comuns
  passam; só o objeto do SQLAlchemy é recusado.
- `test_gen_dataset.py` — determinismo do gerador, split treino/teste sem
  vazar entidade nem instrução, e o teste principal: para cada entidade, com e
  sem comentários, criar → listar → buscar → atualizar → deletar → 404. Esse
  teste não chama `create_all` por conta própria; o código gerado precisa
  criar as tabelas sozinho. Cobre também o contrato cased: nenhuma saída tem
  nome para restaurar, toda instrução começa em maiúscula, toda rota tem
  docstring capitalizada, e a regressão do split de teste que saía **vazio**
  quando a marca ainda procurava `class comentario(base):`.
  Trava o sorteio de campos: a mesma entidade de treino aparece com mais de
  um conjunto de campos, e o sha256 do split de teste continua o medido em
  2026-10-01, para os números das rodadas seguirem comparáveis.
  Os nomes extras não repetem, não colidem com as entidades originais nem
  com variável do molde, cada um entra com `EXEMPLOS_POR_EXTRA` exemplos, e
  uma amostra deles roda criar e deletar como app real.
  O teste amplo usa só nomes fora do treino, contém o split de teste
  original, cobre 12 entidades, e a referência de cada entidade nova roda
  como app real.
  Nomes colhidos entram com `EXEMPLOS_POR_COLHIDO` exemplos cada, sem mexer
  nos exemplos de antes, e um nome já usado (de teste, original ou extra) é
  recusado com erro. O leitor do TSV aponta a linha fora do formato.
- `test_colher_nomes_classe.py` — as regras de plural (`transacao`, `animal`,
  `funil`, `item`, `fornecedor`), a dobra de acento que casa a Wikipédia com o
  identificador, a extração só de classe de uma palavra (`ItemPedido` e
  `class` dentro de string ficam de fora), o filtro pela Wikipédia e pelos
  nomes já usados, a regressão da primeira colheita (`eliminar`, `object`,
  `emitido` e `visitas` passavam), a fonte por sufixo da Wikipédia, o corte
  pelos mais frequentes, e a leitura de JSONL com `\u2028` dentro do texto.
- `test_restore_case.py` — o nome do módulo em `from`/`import` fica intacto,
  nomes de biblioteca são restaurados, strings e comentários não são tocados, e
  código que não tokeniza vira `ValueError`.
- `test_build_tokenizer_v2.py` — as funções puras do construtor do tokenizer v2
  (corte de score, log-prob das cased, append que preserva a ordem das 64k
  peças) rodam **sem** a lib `tokenizers`; o teste de ponta a ponta fica atrás
  de `importlib.util.find_spec("tokenizers")` e valida o que importa: o garfo
  morto (`deletar_comentario` termina na peça única `comentario`), o canal de
  caso aberto (`FastAPI` deixa de virar byte) e os ids originais intactos.
- `test_eval_code.py` — o harness pontua a referência com 1.0, saída
  quebrada conta como falha sem derrubar a avaliação, e o limite de geração
  cobre a maior referência. Cobre também o `por_marcador` (para separar "errou
  a rota" de "errou o nome da função") e a geração em lote, que já teve o bug
  de cortar o prompt pelo tamanho sem pad. E o eval sem remendo:
  `--sem-restauro` não esconde saída minúscula (`fastapi()` dá `NameError`),
  `case_rate`/`docstring_rate`, e o critério do passo 4 — o ground truth do
  split de teste dá `runs_rate` 1.0 sem restauro e sem reparo.
  O `entidade_rate` tem regressão própria: `class Membro(Base)` no lugar de
  `Comentario` reprova, mesmo compilando e com a rota certa.
- `test_sandbox_gerado.py` — o `rodar_rotas` aceita o app completo **e** uma
  saída de uma operação só: o dataset gera uma operação por exemplo, então uma
  saída de `delete` não tem POST e tem de passar mesmo assim (a primeira versão
  exigia o CRUD inteiro e reprovava 8 das 32 saídas — media a premissa, não o
  modelo). E o que importa: **pega o nome trocado que compila**. Renomear a
  classe e deixar as referências quebra em runtime com `NameError`, sem o
  `ast.parse` ver nada. É o teste que justifica o `--executar` existir. Tem
  também a regressão do payload: `endswith("criar")` não via `UsuarioCriar`, o
  POST ia sem corpo e o 422 passava como sucesso.
- `test_expandir_vocab.py` — a pasta do v2 é montada com os arquivos certos e
  falha com o caminho quando falta um; `A` herda a peça `▁a` do tokenizer
  antigo e o `▁` solto sai da média; o mapa cobre todo id novo; e o
  `expandir_modelo` inicia entrada e saída, não mexe em modelo já expandido e
  recusa tamanho estranho. Usa um `ModeloFalso` nomeado no lugar do
  `PreTrainedModel`. Os testes com tokenizer real ficam atrás de `skipif`
  (precisam de `models/` e da lib `tokenizers`), os de embedding atrás de
  `torch` — nenhum dos dois está em `requirements-dev.txt`.
- `test_montar_corpus_cpt.py` — monta um upstream falso em `tmp_path` e cobre
  as três fontes, o caso preservado, a remoção da diretiva do mkdocs só na
  prosa, o descarte de documento contaminado (`class Comentario`) e a val
  determinística e nunca vazia.
  Com `--extra`: agrupa por origem, aponta a linha sem licença e aplica a
  mesma guarda e a val por origem aos documentos colhidos.
- `test_colher_codigo_ptbr.py` — o detector de português quebra camelCase e snake_case, não conta palavra que colide com inglês, e o filtro exige Python + licença permissiva + português + tamanho são. O filtro de shard roda sobre um parquet falso em `tmp_path` (atrás de `pyarrow`).
- `test_colher_wikipedia_ptbr.py` — pula esboço, marca a licença CC BY-SA e para no orçamento de caracteres, sobre um parquet falso (atrás de `pyarrow`).
- `test_pretreino_continuado.py` — as partes puras do CPT: empacotamento em
  blocos iguais separados por EOS (sobra descartada) e a leitura do corpus
  com o número da linha quebrada no erro. E os índices espaçados da val,
  que não pode medir só a primeira origem. O treino em si só roda na GPU.
- `test_ptbrificar_identificadores.py` — o teste que importa é a regressão de
  `FastAPI(description=x)`: o `kwarg` tem de sair `description=descricao`, nunca
  `descricao=descricao` (que o FastAPI ignoraria em silêncio). Também cobre
  campo de modelo renomeado, `kwarg` de classe local acompanhando o campo,
  atributo/import/string intactos, colisão (`read_main` + `main` → só um vira
  `principal`) e o `col_offset` do `ast`, que é em **bytes** UTF-8 — por isso
  tem um caso com `"ação"` antes do identificador na mesma linha.
- `test_harvest_fastapi_docs.py` — monta uma árvore falsa de repo em `tmp_path`
  (nada de rede) e cobre a resolução do caminho do include, a limpeza da prosa
  (inclui a remoção completa do `{* ... *}`, que já foi bug), o descarte de
  código que não compila, a contagem de citação repetida e a escrita do JSONL
  com e sem o campo `_origem`.

## Fluxo

`test_gen_dataset.py` → `gen_dataset.montar_saida` → `app_de_referencia` →
`exec` → `TestClient`. Os demais arquivos chamam as
funções de `scripts/` direto, sem I/O: o de harvest monta a árvore em `tmp_path`
e nunca toca a rede, e o de ptbrificar só passa string para `renomear`.
