# Alpha Bank Privacy-Preserving LLM Preprocessing

## Problem

Financial support and service messages often contain personal and sensitive user data.  
If raw text is sent directly to an LLM, this increases the risk of exposing PII to external model processing and logging systems.  
The practical task is to introduce a preprocessing layer that reliably identifies sensitive spans and prepares safer downstream usage.

## Current solution

The current repository implements a notebook-first **hybrid detection pipeline**.  
The canonical flow combines:

- model-based extraction for flexible and ambiguous entities in Russian text;
- deterministic normalization and validation of spans and labels;
- regex fallback for highly structured identifiers (phone, email, card-like, passport-like, and similar patterns).

This is the best-performing practical approach among explored variants in this project.

## Why the approach is practical

A model-only pipeline may produce unstable formats, while regex-only logic misses many natural-language cases.  
The hybrid design addresses both issues: model extraction gives semantic flexibility, and deterministic post-processing stabilizes output quality.  
Regex fallback improves coverage for structured patterns that are important in real user traffic.

## Why it is safer than direct LLM calls

The implemented extraction and normalization layer is designed as a safety-oriented preprocessing stage.  
It converts raw text into explicit sensitive spans and structured entities, which is the required foundation for masking before LLM calls.  
This reduces reliance on opaque model behavior and enables controlled privacy logic.

## What is implemented already

- Notebook-based hybrid extraction pipeline.
- JSON extraction and deterministic entity normalization.
- Regex fallback for structured PII.
- Batch inference and offline artifact generation.

## Next packaging step

The natural next step is to package this baseline into a minimal proxy service:

- FastAPI endpoints for `detect`, `mask`, `demask`, and `proxy/chat`;
- request-scoped reversible placeholders;
- optional upstream LLM call with safe restoration;
- Dockerized runtime and lightweight tests.

This packaging step is planned and is not claimed as implemented in the current repository snapshot.

