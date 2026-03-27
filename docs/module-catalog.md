# Module Catalog

Cross-project inventory of reusable agent components drawn from MADP, ContextForge, ARENA, and VaultLog.
Use this as a reference when building new templates — check here before writing something from scratch.

---

## Query & Research

| Module | Origin project | Description | Ported to template |
|---|---|---|---|
| `query_understanding` | MADP | Intent classification, query normalisation | research-agent |
| `vector_retriever` | MADP / ARENA | Dense similarity search over a Qdrant collection | — |
| `graph_retriever` | MADP | Cypher generation + Neo4j traversal | — |
| `context_fusion` | MADP | Merges + deduplicates vector + graph results | — |

## Reasoning & Generation

| Module | Origin project | Description | Ported to template |
|---|---|---|---|
| `reasoner` | MADP | Grounded answer generation with citations | research-agent (adapted) |
| `verifier` | MADP | Faithfulness + confidence scoring | research-agent (adapted) |
| `response_formatter` | MADP | Structures final JSON output | research-agent |

## Ingestion & Processing

| Module | Origin project | Description | Ported to template |
|---|---|---|---|
| `parser` | MADP | Docx extraction, layout-aware chunking | — |
| `pii_redactor` | MADP | Presidio-based PII masking | — |
| `chunker` | MADP | Sliding window chunker with overlap | — |
| `embedder` | MADP / ARENA | nomic-embed-text embeddings → Qdrant | — |
| `entity_extractor` | MADP | Clause, term, risk, penalty extraction | — |

## Context Engineering (ContextForge)

| Module | Origin project | Description | Ported to template |
|---|---|---|---|
| `system_prompt_altitude` | ContextForge | Dynamic system prompt injection by task tier | — |
| `context_window_manager` | ContextForge | Token budget tracking + truncation strategy | — |
| `retrieval_router` | ContextForge | Routes queries to vector vs graph vs cache | — |

## Evaluation (ARENA)

| Module | Origin project | Description | Ported to template |
|---|---|---|---|
| `faithfulness_scorer` | ARENA | RAGAS-style faithfulness metric | — |
| `noise_injector` | ARENA | Adds synthetic noise to test RAG robustness | — |
| `eval_runner` | ARENA | Batch eval loop with result logging | — |

## Audit & Compliance (VaultLog)

| Module | Origin project | Description | Ported to template |
|---|---|---|---|
| `audit_writer` | VaultLog / MADP | Immutable Postgres audit trail per action | — |
| `approval_logger` | VaultLog | Approval chain logger for compliance workflows | — |

---

*Update this table whenever a module is extracted into a template.*
