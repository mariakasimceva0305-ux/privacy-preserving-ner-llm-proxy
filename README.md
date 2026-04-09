# Privacy-Preserving NER LLM Proxy

A privacy-first NLP preprocessing pipeline for **Russian text before LLM usage**. The project extracts sensitive entities, normalizes them, and masks or transforms protected fragments before the downstream request reaches a language model.

## Why This Project
In real-world LLM workflows, the model is not always the first place to start. In regulated or privacy-sensitive scenarios, a better design is often:
1. inspect the raw input;
2. detect sensitive spans;
3. normalize and mask them;
4. only then pass the sanitized request to an LLM.

This repository is built around that principle.

## Problem Statement
Protect sensitive information in Russian-language user text while preserving enough semantic structure for downstream LLM processing.

## Approach
The pipeline combines:
- **NER-style entity extraction**
- **normalization / validation**
- **regex fallback rules**
- **masking / replacement logic**

This hybrid approach is useful because purely learned extraction may miss patterned sensitive data, while purely rule-based masking may be too brittle on natural text.

## Repository Structure
```text
notebooks/
  alpha_bank_privacy_pipeline.ipynb  # main notebook with the pipeline
SUBMISSION.md                        # project submission notes
requirements.txt
README.md
```

## Pipeline Stages
### 1) Entity Detection
Detect candidate sensitive spans in Russian text.

### 2) Normalization
Standardize extracted values to reduce noisy variation and improve downstream matching.

### 3) Regex Fallback
Apply deterministic rules for structured or weakly captured sensitive patterns.

### 4) Sanitization
Replace or mask protected fragments before the request goes further.

## Design Choice: Hybrid NER + Rules
This design is deliberate.

**Benefits:**
- better coverage across free-form and patterned entities
- more transparent error analysis than black-box prompting alone
- easier compliance-oriented review of preprocessing behavior

**Trade-off:**
- rule layers require maintenance
- normalization must be designed carefully to avoid over-masking

## Evaluation Strategy
For this type of pipeline, the key questions are:
- what percentage of sensitive spans are detected?
- how many false positives are introduced?
- what classes benefit most from regex fallback?
- how much meaning is preserved after masking?

Recommended reporting:
- precision / recall / F1 for entity extraction
- coverage gain from hybrid logic compared with a single method
- qualitative examples of safe transformation

## Running Locally
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
jupyter notebook
```

Then open `notebooks/alpha_bank_privacy_pipeline.ipynb`.

## Where This Project Is Strong
- privacy-aware LLM preprocessing
- NLP pipeline design beyond prompt-only solutions
- transparent handling of structured sensitive patterns
- interview discussion around compliance, trade-offs, and error analysis

## Suggested Next Improvements
- convert notebook logic into a package / service module
- add labeled evaluation examples by entity class
- separate masking policy from extraction logic
- add reversible pseudonymization mode where applicable
- provide before/after transformation examples in the README

## Limitations
- notebook-centric implementation is good for demonstration but not yet ideal for deployment
- entity classes and masking policy depend on task requirements
- privacy protection is only as strong as the covered entity definitions

## Takeaway
This repository demonstrates a mature idea: in sensitive LLM workflows, **preprocessing can be as important as generation**. The project is valuable because it focuses on a real system boundary — what should happen *before* a model sees the text.
