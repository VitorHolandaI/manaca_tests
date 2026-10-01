# docs/data/ — dataset e licenças

Esta pasta documenta **de onde vêm os dados** do fine-tune de geração de código
PT-BR e **qual a licença de cada coisa**. A regra do projeto é: só entra fonte
aberta e livre pra uso (comercial inclusive).

## O dataset deste projeto

| Item | Valor |
|---|---|
| Arquivos | `data/fastapi_ptbr.jsonl` (treino) e `data/fastapi_ptbr_teste.jsonl` (teste), gerados, fora do git |
| Gerador | `scripts/gen_dataset.py` |
| Exemplos | treino 2336 linhas / 1168 instruções únicas · teste 64 / 32 |
| Split | entidades `comentario` e `endereco` inteiras só no teste |
| Formato | Alpaca `{instruction, input, output}` |
| Template | Alpaca-PT sem acento (`preambulo + ### Instrucao: / ### Resposta:`) — o mesmo de `train_qlora.montar_texto` e `eval_code._prompt` |
| Tokens por exemplo | treino 484 · 606 · 730, teste 494 · 610 · 728 (min · mediana · max; teto 2048) |
| **Licença** | **CC0-1.0** — gerado por nós, sem dado de terceiro |

Como é inteiramente sintético (templates combinatórios sobre 75 entidades × 5
operações), não herda licença de ninguém. Isso elimina o risco jurídico que
costuma vir embutido em datasets de código raspados do GitHub.

### Contrato de caixa

O alvo é **todo minúsculo**, porque o Manacá-1B é case-blind (0 peças maiúsculas
no vocabulário de 64k). A capitalização das bibliotecas é devolvida no inference
por `scripts/restore_case.py`. Ver a seção "Restrição central" em
`scripts/README.md`.

## O dataset colhido do FastAPI (MIT)

| Item | Valor |
|---|---|
| Arquivo | `data/fastapi_oficial_ptbr.jsonl` (gerado; fora do git) |
| Geradores | `scripts/harvest_fastapi_docs.py` + `scripts/ptbrificar_identificadores.py` |
| Exemplos | 289 |
| Origem | `fastapi/fastapi` em `docs/pt/docs/` × `docs_src/` |
| **Licença** | **MIT** — obra derivada; exige manter o aviso de copyright |
| Tokens | ~86K (contexto máx. 2384, cabe em 4096 com folga) |

Diferente do dataset sintético, aqui **nada é inventado**: a prosa é a tradução
oficial da comunidade (pt-BR, não pt-PT — `arquivo` 205×, `ficheiro` 0×) e o
código é o exemplo canônico do próprio FastAPI.

O join é determinístico: cada página em `docs/pt/docs/` cita o arquivo de código
por caminho, no formato `{* ../../docs_src/x/tutorial001.py *}`.

O porquê de MIT importar: o conteúdo humano em português mais óbvio seria o
`pt.stackoverflow.com`, mas ele é **CC BY-SA** — o ShareAlike contamina a saída.
O FastAPI é MIT, então a obra derivada continua permissiva.

**Atenção:** o `docs_src/` é compartilhado entre os 12 idiomas, então o código
vem em inglês. A tradução dos identificadores é feita por
`scripts/ptbrificar_identificadores.py`, com decisão semântica via `ast`, e o
resultado mantém a API da biblioteca intacta.

## Fontes externas avaliadas

Nível de licença verificado em 2026-09-28 (cards do Hugging Face / repositórios).

| Fonte | Conteúdo | Licença | Uso comercial | Situação |
|---|---|---|---|---|
| `nvidia/OpenCodeInstruct` | 5M instruções Python | **CC BY 4.0** | sim | ✅ candidato |
| `nvidia/OpenCodeGeneticInstruct` | 15M Python sintético | **CC BY 4.0** | sim | ✅ candidato |
| `m-a-p/CodeFeedback-Filtered-Instruction` | 74k código | **Apache 2.0** | sim | ✅ candidato |
| `flytech/python-codes-25k` | 25k Python | **MIT** | sim | ✅ candidato |
| **`fastapi/fastapi`** (`docs/pt` + `docs_src`) | 289 pares prosa pt-BR → código FastAPI | **MIT** | sim | ✅ **em uso** |
| `HuggingFaceTB/stackexchange_2025_md` (`pt.stackoverflow.com`) | 596 MB de Q&A humano pt-BR em markdown, com código | CC BY-SA 3.0/4.0 (por post) | sim, com ShareAlike | ⚠️ polui a licença |
| `diiogofernands/ptbr-cpt-code-v1` | instruções PT-BR de código | não declarada | não claro | ❌ **401, virou privado** |
| `NickIBrody/python-code-instructions-85k` | 85k Python | o card diz *"should not be treated as a clean MIT-licensed dataset"* | não claro | ❌ evitar |
| `DominguesM/Canarim-Instruct-PTBR` | 316k PT-BR | derivado de Alpaca/self-instruct (gerado por OpenAI) | ambíguo | ⚠️ evitar |
| `Madras1/corpus-ptbr-v1` | corpus PT-BR | ODC-By, mas subset sintético tem ToS de LLM | parcial | ⚠️ parcial |

**Decisão:** o projeto tem dois datasets. O **sintético** (`gen_dataset.py`) é CC0
e dá volume na tarefa exata. O **colhido** vem do `fastapi/fastapi` (MIT) e dá os
padrões canônicos da biblioteca com prosa oficial pt-BR. As demais fontes acima
ficam como referência caso se queira mais diversidade — nesse caso, usar só as ✅
e manter o registro da licença no manifest.

### Por que não usar CodeAlpaca

`CodeAlpaca` é gerado por `text-davinci-002` a partir do Self-Instruct. A licença
declarada é Apache-2.0, mas o conteúdo derivado de saída da OpenAI carrega os
Termos de Uso da OpenAI. É o mesmo problema que torna o instruct do Manacá
CC BY-NC: **saída de modelo proprietário na mistura contamina a licença**. Como a
regra aqui é livre pra uso, ficou de fora.

## Nota sobre o modelo base

Para o fine-tune ser livre pra uso, a base tem de ser também:

| Modelo | Licença | Serve? |
|---|---|---|
| `AIInstituteLNCC/manaca-1b-base` | **CC BY 4.0** | ✅ usar este |
| `AIInstituteLNCC/manaca-1b-instruct` | CC BY-NC 4.0 | ❌ não-comercial |

O instruct é NC porque a mistura de SFT dele inclui Alpaca-PT e XLSum. O base não.
Por isso `scripts/train_qlora.py` parte do **base**.

## Reproduzir a verificação

```bash
python scripts/gen_dataset.py                                  # gera treino + teste + manifest
python -c "import json;print(json.load(open('data/manifest.json')))"
# o manifest registra contagens, seed, sha256 dos dois splits e a licença
```
