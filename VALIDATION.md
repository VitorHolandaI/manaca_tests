# VALIDATION.md — relatório para validação independente

Escopo: scaffold do fine-tune de geração de **código FastAPI em PT-BR** sobre o
Manacá-1B base. Sessões de 2026-09-28. Este documento existe para que **outro
agente** confira cada afirmação com o comando que a produziu.

Regra deste relatório: toda afirmação tem (a) o comando, (b) a saída esperada, e
(c) se foi **executado** ou é apenas **afirmado**. Não confie no texto — rode.

---

## Revisão (2026-09-28, segunda sessão)

A primeira versão deste relatório dizia "harness calibrado = 1.0" e
"256/256 compilam". Os números estavam certos, mas `ast.parse` só prova
sintaxe. A revisão rodou o código de referência de verdade e achou quatro
defeitos que invalidariam o experimento na 3060:

| # | Defeito | Evidência | Correção |
|---|---|---|---|
| 1 | buscar/atualizar/deletar faziam `produto = sessao.query(produto)` → `UnboundLocalError` em runtime (144/256 exemplos); o app também nunca criava as tabelas | rodado com FastAPI + TestClient | variável local virou `registro`; `base.metadata.create_all(bind=engine)` depois do modelo |
| 2 | `restore_case` trocava o nome do módulo: `from FastAPI import FastAPI` | saída de `restaurar()` | `nomes_a_restaurar()` pula o módulo em `from`/`import` |
| 3 | `max_new_tokens=512` cortava a geração em 196/256 referências (hoje seriam 220/256) | medido com o tokenizer oficial | padrão 1024, flag `--max-new-tokens` |
| 4 | treino e avaliação no mesmo arquivo → a nota mediria decoreba | leitura do código | split com entidades inteiras fora do treino (`comentario`, `endereco`) |

Também corrigido: saída que não tokeniza agora vira falha em vez de derrubar o
eval; o fallback `from transformers import DataCollatorForCompletionOnlyLM` não
existia (conferido no 4.46.3 e no 5.17); `tokenizer=` virou
`processing_class=` (assinatura do `SFTTrainer` no trl 0.12.2); o eval usa
`torch_dtype=`, que as duas versões do transformers aceitam; e as versões do
treino foram fixadas em `requirements-3060.txt`.

---

## 0. Ambiente necessário para reproduzir

| Item | Caminho / valor |
|---|---|
| Repo | `.` (git init, branch `main`, sem commits) |
| Venv com torch XPU + transformers 5.17 | `~/.venvs/manaca-xpu` |
| Stack do treino (3060) | `requirements-3060.txt` — torch 2.5.1+cu124, transformers 4.46.3, trl 0.12.2, peft 0.13.2. Resolvido com `uv pip compile`, **ainda não instalado numa 3060** |
| Stack dos testes | `requirements-dev.txt` (pytest, fastapi, sqlalchemy, httpx) |
| Tokenizador oficial | `models/manaca-hf-official/tokenizer.model` |
| Modelo instruct oficial (3,44 GB) | `models/manaca-hf-official/` (sha256 `0c95dc5a…418ff3`) |
| GGUF F16-UGM (3,45 GB) | `models/manaca-instruct-ugm/F16-UGM.gguf` (sha256 `538713e5…f1074`) |
| GPU local | Intel Arc 130T (iGPU, XPU). **Sem NVIDIA** — nada de CUDA/bitsandbytes aqui |

---

## 1. Afirmações com evidência

### 1.1 O gerador produz treino + teste, determinístico

```bash
python scripts/gen_dataset.py
# esperado:
# treino: 192 exemplos em data/fastapi_ptbr.jsonl (sha256 72d3809fbbcc2296fc76c68224a4f8f3ecb7a490daafaafdfd80eb54d57f5dd6)
# teste:  64 exemplos em data/fastapi_ptbr_teste.jsonl (sha256 b810da090d21719a24c7e63bd0d8f2cbe164abf52f9c13801486ca344b5bca1c)
```

Estável com `--seed 1234` (o default). O teste tem 32 instruções únicas sobre
duas entidades que o treino nunca viu.

Status: **executado**, e gerado duas vezes com `cmp` idêntico.

### 1.2 O código de referência roda, não só compila

```bash
uv run --with-requirements requirements-dev.txt pytest tests/
```

O teste principal (`tests/test_gen_dataset.py::test_referencia_roda_o_crud_inteiro`)
sobe cada entidade como app real e faz criar → listar → buscar → atualizar →
deletar → 404.

Status: **escrito, ainda não executado** — aguardando o ok para rodar a suíte.

### 1.3 O orçamento de tokens cabe no contexto

Medido com o tokenizer oficial (HF, com BOS), no template completo:
treino **505 · 627 · 738** (min · mediana · max), teste **512 · 630 · 751**.
A maior saída sozinha tem 697 tokens, por isso a geração do eval vai até 1024.

Status: **executado**. Script na seção 5.

### 1.4 O harness de avaliação está calibrado

```bash
python scripts/eval_code.py --pred data/fastapi_ptbr_teste.jsonl
# esperado: {"total": 32, "compile_rate": 1.0, "marker_rate": 1.0,
#            "precisaram_de_restauro": 32, "falhas": []}
```

`--ref` agora aponta para o split de teste por padrão. Calibração continua sendo
só sintaxe + marcadores; a execução real é coberta pelo item 1.2.

Status: **executado** → conforme esperado.

### 1.5 Baseline zero-shot

A medição antiga (instruct, 3 exemplos, 0.0/0.0) **não vale mais**: foi feita
com `max_new_tokens=512`, no dataset antigo, e com template sem acento, que não
é o que o instruct viu no SFT. Refazer no base, no split de teste:

```bash
python scripts/eval_code.py --model AIInstituteLNCC/manaca-1b-base
```

Status: **não executado** depois das correções.

### 1.6 Lint e type-check limpos

```bash
F="scripts/gen_dataset.py scripts/restore_case.py scripts/eval_code.py scripts/train_qlora.py tests/"
uv run --with ruff ruff check $F                      # All checks passed!
uv run --with ruff ruff format --check $F             # já formatado
MYPYPATH=scripts uv run --with mypy --with-requirements requirements-dev.txt \
    mypy $F --ignore-missing-imports                  # no issues found
```

Status: **executado**. `scripts/chat_oficial.py` está fora do escopo (código
oficial baixado).

---

## 2. O que NÃO foi provado (validar com prioridade)

| Item | Por que não foi provado | Como validar |
|---|---|---|
| **Suíte em `tests/`** | escrita, não rodada | `uv run --with-requirements requirements-dev.txt pytest tests/` |
| **`train_qlora.py` roda** | Não há NVIDIA nesta máquina | Na 3060, com `requirements-3060.txt`: `accelerate launch scripts/train_qlora.py --smoke`. Loss finita e `check do collator OK` |
| **O fine-tune melhora o `compile_rate`** | O treino nunca rodou | Treinar no split de treino, avaliar no de teste, comparar com o baseline do base (1.5) |
| **Generalização para entidade nova** | O teste tem entidades novas, mas os campos delas (`rua`, `cep`…) não aparecem na instrução | `compile_rate` e `marker_rate` não exigem campos; o schema gerado vai ser inventado, e isso é esperado |
| **Diversidade do dataset** | 192 exemplos de templates combinatórios | Olhar 10 exemplos aleatórios |

---

## 3. Riscos e pontos de atenção

1. **`restore_case.py` tem mapa fixo.** Um identificador de domínio chamado
   `session`, `column` etc. seria capitalizado errado. O gerador usa `sessao`.
2. **O alvo de treino é 100% minúsculo, por design.** O tokenizer normaliza para
   minúsculas (`Abaixo está Instrução` → `▁abaixo ▁está ▁instrução`).
3. **QLoRA é opcional numa 3060.** O modelo em bf16 tem ~2,4 GB; LoRA sem 4-bit
   cabe e tira o bitsandbytes da frente. Não foi mudado.
4. **Template sem acento.** `train_qlora.py` e `eval_code.py` usam
   `### Instrucao:`, e o `chat_oficial.py` usa `### Instrução:`. Treino e eval
   estão consistentes entre si, mas o template não é idêntico ao oficial.
5. **`data/` e `models/` são gitignored** com `/data/` ancorado na raiz, para
   `docs/data/README.md` não ser ignorado junto.

---

## 4. Arquivos entregues

| Arquivo | Papel | Verificado? |
|---|---|---|
| `scripts/gen_dataset.py` | gerador + split treino/teste (CC0) | ✅ executado |
| `scripts/restore_case.py` | restaura caso canônico, pula nome de módulo | ✅ executado via eval |
| `scripts/eval_code.py` | avaliação sem juiz LLM | ✅ executado com `--pred` |
| `scripts/train_qlora.py` | QLoRA no base CC BY 4.0 | ❌ **não executado** |
| `scripts/chat_oficial.py` | `chat.py` oficial + patch `xpu` | fora do fine-tune |
| `tests/` | execução real do código de referência | ⏳ escrito, não rodado |
| `requirements-3060.txt` | stack fixado do treino | resolvido, não instalado |
| `requirements-dev.txt` | stack dos testes | — |

---

## 5. Reproduzir as medições citadas

```bash
cd .
PY=~/.venvs/manaca-xpu/bin/python

# 1.3 — orçamento de tokens
$PY - <<'PY'
import json, statistics
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/manaca-hf-official")
PRE = "Abaixo esta uma instrucao que descreve uma tarefa. Escreva uma resposta que atenda adequadamente ao pedido."
for f in ("data/fastapi_ptbr.jsonl", "data/fastapi_ptbr_teste.jsonl"):
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    cs = [len(tok(f"{PRE}\n\n### Instrucao:\n{r['instruction']}\n\n### Resposta:\n{r['output']}")["input_ids"]) for r in rows]
    print(f, min(cs), int(statistics.median(cs)), max(cs))
PY
# esperado: treino 505 627 738 · teste 512 630 751
```

---

## 6. Julgamento honesto do que isto vale

Antes da revisão, a infraestrutura **não** estava pronta: o modelo ia decorar
código que quebra em runtime e seria avaliado nos próprios dados de treino, com
a geração cortada. Isso foi corrigido. Falta rodar a suíte em `tests/` e fazer
o `--smoke` na 3060.

Veredito: **a infraestrutura está pronta para o experimento, pendente da suíte
e do smoke test; o experimento não foi feito.**
