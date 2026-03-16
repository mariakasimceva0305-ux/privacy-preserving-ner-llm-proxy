# Alpha Bank: privacy-preserving preprocessing для LLM

## Краткое описание

Репозиторий содержит рабочий baseline для кейса Альфа-Банка по защите персональных данных в пользовательских сообщениях перед использованием LLM.  
Текущая реализация построена как `notebook-first` решение и опирается на `hybrid pipeline`: model-based extraction, deterministic normalization и regex fallback для структурированных PII.

## Какую проблему решает

В реальных банковских сценариях в тексте клиента часто встречаются персональные и чувствительные данные.  
Передача raw text в LLM без предварительной обработки повышает риск утечек.  
Этот проект закрывает задачу безопасной предобработки: выделить чувствительные сущности и подготовить текст к контролируемому downstream использованию.

## Ключевая идея решения

Основная инженерная идея — не полагаться только на один метод, а использовать `hybrid pipeline`:

- model-based extraction для гибких и неоднозначных случаев;
- deterministic normalization для стабилизации структуры ответа;
- regex fallback для хорошо формализуемых идентификаторов.

Такой подход дал наиболее практичный результат в текущей серии экспериментов.

## Ключевые возможности

- Извлечение сущностей из русскоязычного текста в структурированном виде.
- Нормализация и валидация сущностей по полям `start`, `end`, `label`.
- Дедупликация и ограничение по допустимым категориям.
- Fallback-обработка через regex для `EMAIL`, `PHONE`, `CARD`, `PASSPORT`, `SNILS`, `INN`.
- Формирование offline artifacts для submission.

## Как работает pipeline

```text
Входной текст
  -> model-based extraction (JSON-like output)
  -> JSON block extraction
  -> deterministic normalization и validation
  -> regex fallback при пустом/невалидном результате
  -> структурированный список сущностей
```

## Почему этот подход практичен

- Model extraction покрывает сложные случаи, где regex недостаточен.
- Deterministic post-processing снижает нестабильность model output.
- Regex fallback повышает полноту на структурированных форматах.
- Итоговый контур хорошо подходит как практический baseline для privacy-preserving preprocessing, а не только как учебный NER-эксперимент.

## Что реализовано сейчас

Текущий код находится в notebook:

- `notebooks/alpha_bank_privacy_pipeline.ipynb` — канонический рабочий pipeline;
- batch inference;
- генерация submission artifacts.

Важно: в текущем состоянии репозиторий не упакован в отдельный сервисный runtime.

## Структура репозитория

```text
project-root/
  notebooks/
    alpha_bank_privacy_pipeline.ipynb
  README.md
  SUBMISSION.md
  requirements.txt
  .gitignore
  local private data and local artifacts (not for public publishing)
```

## Как запустить

1. Создать и активировать `virtual environment`.
2. Установить зависимости.
3. Открыть notebook и выполнить канонический inference-блок.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Ограничения

- Текущий runtime — `notebook-first`.
- Нет выделенного service entrypoint.
- Нет формализованного набора tests.
- Нет отдельного воспроизводимого отчета по latency и quality-метрикам.

## Дальнейшее развитие

1. Вынести pipeline из notebook в модульную структуру `src/`.
2. Упаковать решение в минимальный `FastAPI proxy`.
3. Добавить request-scoped masking/de-masking.
4. Добавить базовые tests для критичных частей pipeline.
5. Добавить Docker-упаковку для воспроизводимого запуска.

## Tech stack

- Python
- Jupyter Notebook
- Hugging Face Transformers
- PyTorch
- pandas
- NumPy
- scikit-learn
- regex-based post-processing

