
"""
legal_api.py — FastAPI server for Legal Precedent Recommendation System
Run: uvicorn legal_api:app --host 0.0.0.0 --port 8000 --reload
"""
 
import json, base64, io, random
from pathlib import Path
from typing import List, Optional, Dict
 
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed on server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
 
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
# ── Import model + helpers from notebook module ───────────────────────────────
# When running standalone, these come from the notebook-exported module.
# When running from the notebook cell below, they are already in scope.
try:
    from legal_precedent_han import (
        Case, LegalHAN, CourtImportanceLayer,
        PERSIST_DIR, CFG, TIER_LABELS, load_all,
    )
except ImportError:
    pass   # Running inside notebook — all names already in global scope
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas (request + response models)
# ─────────────────────────────────────────────────────────────────────────────
 
class RecommendRequest(BaseModel):
    doc_id : str
    top_k  : int = 10
 
class CaseSummary(BaseModel):
    doc_id      : str
    title       : str
    court       : str
    court_tier  : int
    court_tier_label: str
    year        : int
    n_citations : int
    n_statutes  : int
 
class PrecedentResult(BaseModel):
    rank                : int
    doc_id              : str
    title               : str
    court               : str
    court_tier_label    : str
    year                : int
    similarity          : float
    is_actual_citation  : bool
 
class RecommendResponse(BaseModel):
    query_doc_id    : str
    query_title     : str
    query_court     : str
    recommendations : List[PrecedentResult]
 
class GraphStats(BaseModel):
    n_cases     : int
    n_statutes  : int
    n_courts    : int
    n_judges    : int
    n_citation_edges    : int
    n_invoke_edges      : int
    n_heard_by_edges    : int
    n_decided_by_edges  : int
    tier_distribution   : Dict[str, int]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# App + CORS
# ─────────────────────────────────────────────────────────────────────────────
 
app = FastAPI(
    title       = "Legal Precedent Recommender API",
    description = "HAN-based precedent recommendation on Indian court citation graph",
    version     = "1.0.0",
)
 
# Allow all origins during development — tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Global server state — loaded ONCE at startup
# ─────────────────────────────────────────────────────────────────────────────
 
STATE: Dict = {}   # populated in startup, read-only during requests
 
 
@app.on_event("startup")
async def startup():
    """
    Loads all artefacts from PERSIST_DIR into SERVER_STATE at startup.
    Called once when the server process starts — not on every request.
    """
    artefacts = load_all()
    STATE["cases"]       = artefacts["cases"]
    STATE["case_idx"]    = artefacts["case_idx"]
    STATE["statute_idx"] = artefacts["statute_idx"]
    STATE["court_idx"]   = artefacts["court_idx"]
    STATE["judge_idx"]   = artefacts["judge_idx"]
    STATE["graph_data"]  = artefacts["graph_data"]
    STATE["model"]       = artefacts["model"]
    STATE["idx_to_case"] = {v: c for c, v in artefacts["case_idx"].items()}
    STATE["case_map"]    = {c.doc_id: c for c in artefacts["cases"]}
    print("🚀 API server ready.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _get_all_embeddings() -> torch.Tensor:
    """Single forward pass to get all case embeddings. Cached per request cycle."""
    model      = STATE["model"]
    graph_data = STATE["graph_data"]
    model.eval()
    with torch.no_grad():
        x_dict  = {nt: graph_data[nt].x.to(CFG["device"]) for nt in graph_data.node_types}
        ei_dict = {et: graph_data[et].edge_index.to(CFG["device"])
                   for et in graph_data.edge_types
                   if hasattr(graph_data[et], "edge_index")}
        ct = graph_data["case"].court_tier.to(CFG["device"])
        return model(x_dict, ei_dict, ct)   # [N_cases, D]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
 
@app.get("/health")
def health():
    """
    Server health check.
    Returns model info, number of cases loaded, and device.
    Useful for the frontend to confirm the backend is live before making requests.
    """
    return {
        "status"    : "ok",
        "n_cases"   : len(STATE.get("cases", [])),
        "device"    : str(CFG["device"]),
        "model"     : "LegalHAN",
        "embed_model": CFG.get("embed_model", "InLegalBERT"),
    }
 
 
@app.get("/cases", response_model=List[CaseSummary])
def list_cases(
    court_tier: Optional[int] = Query(None, description="Filter by tier: 0=SC, 1=HC, 2=Lower"),
    limit     : int           = Query(100,  description="Max cases to return"),
):
    """
    Returns a list of all cases in the system.
    Optional ?court_tier=0 returns only Supreme Court cases.
    Each item includes metadata but NOT the full judgement text (use /cases/{doc_id}).
    """
    cases = STATE["cases"]
    if court_tier is not None:
        cases = [c for c in cases if c.court_tier == court_tier]
    cases = cases[:limit]
 
    return [
        CaseSummary(
            doc_id           = c.doc_id,
            title            = c.title,
            court            = c.court,
            court_tier       = c.court_tier,
            court_tier_label = TIER_LABELS.get(c.court_tier, "Unknown"),
            year             = c.year,
            n_citations      = len(c.citations),
            n_statutes       = len(c.statutes),
        )
        for c in cases
    ]
 
 
@app.get("/cases/{doc_id}")
def get_case(doc_id: str):
    """
    Returns full metadata for a single case: title, court, judges, statutes,
    citation list, and the first 500 chars of the cleaned judgement text.
    Raises 404 if doc_id is not in the graph.
    """
    case = STATE["case_map"].get(doc_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case '{doc_id}' not found.")
 
    return {
        "doc_id"          : case.doc_id,
        "title"           : case.title,
        "court"           : case.court,
        "court_tier"      : case.court_tier,
        "court_tier_label": TIER_LABELS.get(case.court_tier, "Unknown"),
        "year"            : case.year,
        "judges"          : case.judges,
        "statutes"        : case.statutes,
        "citations"       : case.citations,
        "text_excerpt"    : case.text[:500] + "…" if len(case.text) > 500 else case.text,
    }
 
 
@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Core recommendation endpoint.
 
    Given a case doc_id, returns top-K precedents ranked by cosine similarity
    of their HAN embeddings. Embeddings capture both textual content (via
    InLegalBERT) AND structural position in the citation graph (via HAN).
    Supreme Court cases are re-weighted by the CourtImportanceLayer.
 
    Request body:
        { "doc_id": "SC_001", "top_k": 10 }
    """
    case_idx = STATE["case_idx"]
    case_map = STATE["case_map"]
    idx_to_case = STATE["idx_to_case"]
 
    if req.doc_id not in case_idx:
        raise HTTPException(status_code=404, detail=f"Case '{req.doc_id}' not in graph.")
 
    # Get embeddings + compute cosine similarity
    all_emb    = _get_all_embeddings()
    query_idx  = case_idx[req.doc_id]
    q_emb      = F.normalize(all_emb[query_idx].unsqueeze(0), dim=-1)
    all_norm   = F.normalize(all_emb, dim=-1)
    sims       = (q_emb @ all_norm.T).squeeze()
    sims[query_idx] = -1.0   # exclude self
 
    topk_scores, topk_indices = sims.topk(req.top_k)
    topk_scores  = topk_scores.cpu().tolist()
    topk_indices = topk_indices.cpu().tolist()
 
    query_case   = case_map[req.doc_id]
    actual_cites = set(query_case.citations)
 
    results = []
    for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices), start=1):
        rec_id   = idx_to_case.get(idx)
        rec_case = case_map.get(rec_id)
        if not rec_case:
            continue
        results.append(PrecedentResult(
            rank                = rank,
            doc_id              = rec_id,
            title               = rec_case.title,
            court               = rec_case.court,
            court_tier_label    = TIER_LABELS.get(rec_case.court_tier, "Unknown"),
            year                = rec_case.year,
            similarity          = round(score, 4),
            is_actual_citation  = rec_id in actual_cites,
        ))
 
    return RecommendResponse(
        query_doc_id    = req.doc_id,
        query_title     = query_case.title,
        query_court     = query_case.court,
        recommendations = results,
    )
 
 
@app.get("/graph/stats", response_model=GraphStats)
def graph_stats():
    """
    Returns counts of all node types and edge types in the loaded graph,
    plus the court tier distribution. Useful for the frontend dashboard.
    """
    graph_data = STATE["graph_data"]
 
    def edge_count(src_type, rel, dst_type):
        key = (src_type, rel, dst_type)
        if key in graph_data.edge_types:
            return graph_data[key].edge_index.shape[1]
        return 0
 
    tier_counts = {TIER_LABELS[t]: 0 for t in range(3)}
    for case in STATE["cases"]:
        tier_counts[TIER_LABELS.get(case.court_tier, "Lower Court")] += 1
 
    return GraphStats(
        n_cases              = graph_data["case"].x.shape[0],
        n_statutes           = graph_data["statute"].x.shape[0],
        n_courts             = graph_data["court"].x.shape[0],
        n_judges             = graph_data["judge"].x.shape[0],
        n_citation_edges     = edge_count("case", "cites",      "case"),
        n_invoke_edges       = edge_count("case", "invokes",    "statute"),
        n_heard_by_edges     = edge_count("case", "heard_by",   "court"),
        n_decided_by_edges   = edge_count("case", "decided_by", "judge"),
        tier_distribution    = tier_counts,
    )
 
 
@app.get("/graph/plot")
def graph_plot():
    """
    Generates and returns a base64-encoded PNG of the citation subgraph.
    The frontend can render this directly with: <img src="data:image/png;base64,..."/>
    Nodes are colour-coded: red=SC, blue=HC, green=Lower Court.
    """
    cases    = STATE["cases"]
    case_idx = STATE["case_idx"]
    idx_to_case_id = STATE["idx_to_case"]
 
    G = nx.DiGraph()
    for case in cases:
        for cited_id in case.citations:
            if cited_id in case_idx:
                G.add_edge(case.doc_id, cited_id)
 
    tier_colours = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}
    tier_map     = {c.doc_id: c.court_tier for c in cases}
    node_colours = [tier_colours.get(tier_map.get(n, 2), "grey") for n in G.nodes()]
 
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=2)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colours,
            node_size=700, font_size=7, arrows=True, edge_color="#aaaaaa")
    legend_handles = [
        mpatches.Patch(color=c, label=TIER_LABELS[t])
        for t, c in tier_colours.items()
    ]
    ax.legend(handles=legend_handles)
    ax.set_title("Legal Citation Graph")
 
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
 
    return {"image_base64": img_b64, "format": "png"}
