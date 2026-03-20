# privacy-preserving-ner-llm-proxy

Educational project for privacy-preserving preprocessing of Russian text before LLM usage.

## Repository Contents

- `notebooks/alpha_bank_privacy_pipeline.ipynb` - main implementation notebook.
- `SUBMISSION.md` - submission notes.
- `requirements.txt` - dependency list.

## Implemented Functionality

The notebook pipeline includes:

- entity extraction from text,
- normalization and validation of extracted entities,
- regex fallback for structured personal data patterns,
- batch-style processing for submission artifact generation.
