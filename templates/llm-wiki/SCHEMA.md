# Wiki Schema

This document defines the structure, conventions, and workflows for this wiki.
The LLM reads it at the start of every session to operate as a consistent wiki maintainer.

---

## Directory structure

```
wiki/
├── index.md          # catalog of all pages — always written last
├── log.md            # append-only operation history — always written last
├── overview.md       # high-level synthesis of the entire wiki
├── entities/         # people, organizations, products, systems
├── concepts/         # ideas, patterns, frameworks, techniques
├── sources/          # one summary page per ingested source
└── lint-<timestamp>.md  # versioned lint reports (YYYY-MM-DDTHH-MM)

sources/              # raw, immutable input documents — never modify
```

---

## Page format

Every wiki page uses this structure:

```markdown
---
title: <Page Title>
type: entity | concept | source | overview | query
tags: [tag1, tag2]
sources: [source-slug-1, source-slug-2]
updated: YYYY-MM-DD
---

# Page Title

One-sentence summary of this page.

## Body

Main content. Use ## sections freely. Cross-reference other pages
using standard markdown links: [Entity Name](../entities/entity-name.md)

## References

- [Source Title](../sources/source-slug.md)
```

**File naming:** lowercase, hyphen-separated. `machine-learning.md`, `acme-corp.md`.

---

## index.md format

```markdown
# Wiki Index

## Entities
- [Acme Corp](wiki/entities/acme-corp.md) — B2B SaaS company, focus of competitive analysis

## Concepts
- [Retrieval-Augmented Generation](wiki/concepts/rag.md) — LLM pattern for grounding answers in documents

## Sources
- [Author (2024) - Paper Title](wiki/sources/author-2024-paper-title.md) — 2 pages updated

## Queries
- [Comparison: X vs Y](wiki/queries/comparison-x-vs-y.md) — filed 2026-04-04
```

Update index.md on every ingest. Add new pages; update source counts on changed pages.
**Always write index.md after all other pages in the operation.**

---

## log.md format

Append-only. One entry per operation. Entry format:

```
## [YYYY-MM-DD] ingest | <Source Title>
## [YYYY-MM-DD] query | <Question (truncated to 60 chars)>
## [YYYY-MM-DD] lint | <N> issues found
```

**Always write log.md after all other pages, including index.md.**

---

## Ingest workflow

1. Read the source file.
2. Read `wiki/index.md` to understand current structure.
3. Write a summary page to `wiki/sources/<slug>.md`.
4. For each significant entity or concept in the source:
   - If the page exists: read it, then write an updated version.
   - If the page does not exist: create it.
5. Update cross-references on related pages if needed.
6. Write `wiki/index.md` with any new or updated entries.
7. Append to `wiki/log.md`.

A single ingest typically touches 5–15 pages.

---

## Query workflow

1. Read `wiki/index.md`.
2. Identify the 3–5 most relevant pages.
3. Read each page.
4. Synthesize an answer with citations linking to wiki pages.
5. If the answer constitutes a useful, reusable analysis, indicate it
   could be filed — but wait for user confirmation before writing.
6. If the user confirms filing: write to `wiki/queries/<slug>.md`,
   update `wiki/index.md`, append to `wiki/log.md`.

---

## Lint workflow

1. Use `list_wiki` to discover all pages.
2. Read every page.
3. Flag the following issues:
   - **Contradiction:** two pages make conflicting claims.
   - **Orphan:** a page has no inbound links from other wiki pages.
   - **Stale:** a claim is likely superseded by a newer source.
   - **Missing page:** an entity or concept is mentioned but has no page.
   - **Missing cross-reference:** a page mentions an entity that has a page but doesn't link to it.
4. Write findings to `wiki/lint-<YYYY-MM-DDTHH-MM>.md`.
5. Print findings to stdout.
6. Append to `wiki/log.md`.

---

## Hard rules

- **Never modify files in `sources/`.** They are immutable source of truth.
- **Never write `index.md` or `log.md` until all other pages in the operation are written.**
- **Confine all writes to the `wiki/` directory.** The tool enforces this — do not attempt workarounds.
- **One source page per ingested document.** Do not merge sources.
- **Prefer updating existing pages over creating new ones** when the entity or concept already exists.
