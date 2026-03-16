# Alpha Bank Privacy-Preserving LLM Preprocessing

## Коротко о проекте

Этот репозиторий содержит **notebook-first baseline** для privacy-preserving предобработки русскоязычных пользовательских сообщений перед использованием LLM.  
Текущая реализованная основа — **hybrid entity extraction pipeline**: model-based extraction + deterministic normalization + regex fallback для структурированных PII.  
Сейчас проект ориентирован на reproducible offline inference и генерацию submission artifacts; packaging в сервис — следующий шаг.

## Business problem

В банковских и enterprise workflow пользовательские сообщения часто содержат personal и sensitive data.  
Если отправлять raw text напрямую в LLM, повышается риск утечки PII в external processing и logs.  
Практичное решение — добавить preprocessing layer, который обнаруживает и санитизирует sensitive spans до downstream model calls.

## Task formulation

Техническая задача:

1. detect sensitive entities и их character spans в русском тексте;
2. привести extraction output к стабильному structured format;
3. использовать deterministic fallback для highly structured identifiers;
4. подготовить выход для masking-oriented downstream LLM workflow.

## Current implemented approach

Текущий репозиторий находится в состоянии **notebook-first**.  
Канонический implemented path — `Untitled1.ipynb`, где есть:

- prompt-based model extraction в JSON-like output;
- JSON block extraction и parsing;
- deterministic normalization/validation сущностей (`start`, `end`, `label`);
- regex fallback для structured identifiers.

В ноутбуке также есть token-classification training branch, но это не основной стабильный runtime path.

## Почему важен hybrid pipeline

Практическая сила подхода — в комбинации гибких и deterministic шагов:

- model extraction закрывает ambiguous/free-form случаи;
- normalization обеспечивает schema consistency и убирает malformed output;
- regex fallback повышает coverage для structured PII patterns.

Именно эта hybrid схема — ключевая инженерная ценность текущего решения.

## Pipeline / architecture (current state)

```text
Input text
  -> Prompted model extraction
  -> JSON block extraction
  -> Entity normalization and validation
  -> Regex fallback (если model output пустой или невалидный)
  -> Structured entities для offline artifacts
```

## Implemented

- Notebook-based hybrid entity extraction pipeline.
- Deterministic output normalization и label filtering.
- Regex fallback для structured PII patterns.
- Batch inference flow в ноутбуке.
- Offline artifact generation в ноутбуке.

## Planned

Ниже — план packaging, это пока не реализовано как standalone modules:

- FastAPI proxy service (`/health`, `/detect`, `/mask`, `/demask`, `/proxy/chat`).
- Request-scoped masking/de-masking module с reversible placeholders.
- Dockerized runtime service.
- Lightweight API tests и masking tests.

## Repository structure (actual)

```text
project-root/
  Untitled1.ipynb
  README.md
  SUBMISSION.md
  local training data (private/local)
  local test data (private/local)
  local submission artifact(s)
```

Private/local data присутствуют в working directory и не должны публиковаться в public repository.

## How to run

Сейчас рабочий путь — выполнение notebook.

1. Создать и активировать Python virtual environment.
2. Установить dependencies, которые используются в notebook cells.
3. Открыть `Untitled1.ipynb` в Jupyter.
4. Настроить локальные paths/variables в cells.
5. Запустить канонический hybrid inference блок для генерации artifacts.

Пример setup:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install transformers datasets accelerate seqeval scikit-learn pandas torch tqdm jupyter
jupyter lab
```

## Limitations

- Runtime остаётся notebook-first, а не packaged service.
- Нет implemented API entrypoint в repository files.
- Нет implemented Docker runtime files.
- Нет formal test suite.
- Нет опубликованного reproducible latency report.

## Possible improvements

1. Extract notebook logic в небольшой `src/` service package.
2. Добавить FastAPI proxy endpoints для detect/mask/demask/proxy-chat flow.
3. Добавить deterministic placeholder mapping для reversible masking.
4. Добавить streaming-safe response restoration.
5. Добавить lightweight tests для normalization/masking/API contract.
6. Добавить Docker image для reproducible deployment.

## Tech stack

- Python
- Jupyter Notebook
- Hugging Face Transformers
- PyTorch
- pandas
- regex-based post-processing

## TODO

- [ ] Package notebook logic into modules.
- [ ] Add FastAPI proxy implementation.
- [ ] Add Docker runtime.
- [ ] Add tests for critical components.
- [ ] Add strict evaluation report (span + category) without invented metrics.
- [ ] Add safe public sample data and keep private data out of public repo.

