# Alpha Bank Privacy-Preserving LLM Preprocessing

## Problem

User messages in financial workflows often contain personal and sensitive data.  
Sending raw text directly to an LLM increases leakage risk and reduces control over how sensitive fragments are processed.  
The practical goal is to add a preprocessing layer that identifies sensitive spans and stabilizes extraction output before downstream LLM usage.

## Current solution

The current repository provides a notebook-first baseline built around a **hybrid detection pipeline**.  
The canonical flow combines:

- model-based extraction for flexible and ambiguous entities in Russian text;
- deterministic normalization and validation of spans and labels;
- regex fallback for highly structured identifiers (phone, email, card-like and passport-like patterns, and related fields).

Among explored variants in this project, this hybrid approach gave the strongest practical behavior.

## Why the approach is practical

A model-only flow can produce unstable output formats, while regex-only logic misses many natural-language cases.  
The hybrid design addresses both limitations: model extraction gives semantic flexibility, deterministic post-processing stabilizes outputs, and regex fallback improves coverage for structured PII.

## Why it is safer than direct LLM calls

The implemented extraction and normalization layer is a safety-oriented preprocessing stage.  
It converts raw text into explicit sensitive spans and structured entities, which is the required basis for masking before LLM calls.  
This makes downstream privacy handling more controlled than direct raw-text forwarding.

## What is implemented already

- Notebook-based hybrid extraction pipeline (`model + normalization + regex fallback`).
- JSON block extraction and deterministic entity normalization.
- Batch inference flow and offline submission artifact generation.

## Next packaging step

The natural next step is packaging this baseline into a minimal proxy service:

- FastAPI endpoints for `detect`, `mask`, `demask`, and `proxy/chat`;
- request-scoped reversible placeholders;
- optional upstream LLM call with safe restoration;
- Dockerized runtime and lightweight tests.

This packaging step is planned and is not claimed as implemented in the current repository snapshot.

