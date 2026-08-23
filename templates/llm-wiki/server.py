"""
server.py — REST wrapper for llm-wiki.

WHY THIS FILE EXISTS
---------------------
This is NOT part of llm-wiki's own template functionality. It exists so an
external adversarial-testing tool (Kagetora — a separate security-testing
platform, github.com/karunyx/sentinel) has an HTTP interface to scan
against, the same way research-agent's own server.py serves that same
purpose for that template.

llm-wiki's intended interface is run.py (a CLI). If you're building on this
template and don't need REST access, you probably don't need this file —
run.py is the one to use. If you do want REST access for your own reasons
(a demo, a different integration), this is a reasonable starting point, but
it's a minimal wrapper, not a hardened interface: no auth, no rate
limiting, and only the "query" operation is exposed (not "ingest" or
"lint" — a security scan invokes those directly, in-process, bypassing
this file entirely).

INTERFACE
---------
Confirmed against llm-wiki's real agent/graph.py and agent/state.py:
build_graph() takes no required arguments, graph.invoke() takes and
returns a WikiState dict, and the graph's router keys on
state["operation"] (set to "query" below).

USAGE
-----
Run from llm-wiki's own directory, in its own venv, so load_dotenv() picks
up its .env and `agent.graph` imports correctly:
    cd <llm-wiki-path>
    uvicorn server:app --port 8002
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from agent.graph import build_graph

app = FastAPI()
graph = build_graph()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest) -> QueryResponse:
    # Only "query" is wired up here, deliberately — see the module
    # docstring for why "ingest"/"lint" aren't exposed over REST.
    state = {
        "operation": "query",
        "input": req.query,
        "messages": [],
        "wiki_index": "",
        "pages_read": [],
        "output": "",
        "fetched_content": "",
        "save_output": False,
    }
    result = graph.invoke(state)
    return QueryResponse(response=result.get("output", ""))
