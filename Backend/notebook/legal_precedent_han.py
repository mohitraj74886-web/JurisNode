"""
legal_precedent_han.py — Blueprint Library
Contains the structural definitions required by the FastAPI server to load the trained model and data.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv

# ── Global Configuration & Defaults ───────────────────────────────────────────
CFG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "embed_model": "law-ai/InLegalBERT",
    "embed_dim": 768,
    "hidden_dim": 256,
    "out_dim": 64,
    "han_heads": 8,
    "dropout": 0.5,
}

TIER_LABELS = {0: "Supreme Court", 1: "High Court", 2: "Lower Court"}

# ── Data Blueprints ───────────────────────────────────────────────────────────
@dataclass
class Case:
    doc_id: str
    title: str
    court: str  
    court_tier: int = 2  
    year: int = 0
    judges: List[str] = field(default_factory=list)
    text: str = ""  
    summary: str = ""  
    citations: List[str] = field(default_factory=list)  
    statutes: List[str] = field(default_factory=list)  
    embedding: Optional[List[float]] = None  

# ── Neural Network Architecture Blueprints ────────────────────────────────────
class CourtImportanceLayer(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.tier_weights = nn.Embedding(3, emb_dim)
        nn.init.normal_(self.tier_weights.weight, mean=1.0, std=0.1)

    def forward(self, x: torch.Tensor, court_tier: torch.Tensor) -> torch.Tensor:
        tier_w = self.tier_weights(court_tier)
        return x * tier_w


class LegalHAN(nn.Module):
    def __init__(self,
                 in_dims    : Dict[str, int],
                 hidden_dim : int = CFG["hidden_dim"],
                 out_dim    : int = CFG["out_dim"],
                 heads      : int = CFG["han_heads"],
                 dropout    : float = CFG["dropout"],
                 metadata   : tuple = None):
        super().__init__()

        self.projections = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for ntype, dim in in_dims.items()
        })

        self.han1 = HANConv(
            in_channels    = hidden_dim,
            out_channels   = hidden_dim,
            metadata       = metadata,
            heads          = heads,
            dropout        = dropout,
            negative_slope = 0.2,
        )
        self.han2 = HANConv(
            in_channels    = hidden_dim,
            out_channels   = out_dim,
            metadata       = metadata,
            heads          = 1,
            dropout        = dropout,
            negative_slope = 0.2,
        )

        self.court_importance = CourtImportanceLayer(out_dim)
        self.drop   = nn.Dropout(dropout)
        self.act    = nn.ELU()

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[tuple, torch.Tensor],
                court_tier: torch.Tensor) -> torch.Tensor:
        
        h_dict = {
            ntype: self.projections[ntype](x)
            for ntype, x in x_dict.items()
            if ntype in self.projections
        }

        h_dict = self.han1(h_dict, edge_index_dict)
        h_dict = {
            ntype: self.drop(self.act(h))
            for ntype, h in h_dict.items()
            if h is not None
        }

        h_dict = self.han2(h_dict, edge_index_dict)
        h_dict = {
            ntype: self.drop(h)
            for ntype, h in h_dict.items()
            if h is not None
        }

        case_emb = h_dict.get("case")
        if case_emb is None:
            raise RuntimeError("HAN produced no 'case' embeddings — check metadata.")

        case_emb = self.court_importance(case_emb, court_tier)
        return case_emb


# ── Server Boot-Up Helper ─────────────────────────────────────────────────────
def load_all(persist_dir: Path, device: torch.device = CFG["device"]):
    """Restores everything from disk for the FastAPI server."""
    print(f"📂 Loading from {persist_dir.resolve()}...")

    # ── FIX: OS Compatibility for PyTorch (Linux to Windows) ──────────────
    import platform
    import pathlib
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    # ──────────────────────────────────────────────────────────────────────

    # Load Cases (From data directory based on your folder structure)
    data_dir = persist_dir.parent / "data"
    with open(data_dir / "cases.json") as f:
        raw = json.load(f)
    cases = [Case(**c) for c in raw]
    
    # Load Indexes
    with open(persist_dir / "indexes.json") as f:
        idx_data = json.load(f)
    case_idx    = {k: int(v) for k, v in idx_data["case_idx"].items()}
    statute_idx = {k: int(v) for k, v in idx_data["statute_idx"].items()}
    court_idx   = {k: int(v) for k, v in idx_data["court_idx"].items()}
    judge_idx   = {k: int(v) for k, v in idx_data["judge_idx"].items()}

    # Load PyG Graph
    graph_data = torch.load(persist_dir / "graph.pt", map_location=device, weights_only=False)

    # Load Model (Ensure this filename matches what is in your artifacts folder!)
    ckpt = torch.load(persist_dir / "best_legal_han.pt", map_location=device, weights_only=False)

    loaded_cfg  = ckpt.get("cfg", CFG)
    loaded_dims = ckpt.get("in_dims", {
        nt: graph_data[nt].x.shape[1] for nt in graph_data.node_types
    })
    loaded_meta = ckpt.get("metadata", graph_data.metadata())

    restored_model = LegalHAN(
        in_dims  = loaded_dims,
        hidden_dim = loaded_cfg.get("hidden_dim", 256),
        out_dim    = loaded_cfg.get("out_dim", 64),
        heads      = loaded_cfg.get("han_heads", 8),
        dropout    = loaded_cfg.get("dropout", 0.5),
        metadata   = loaded_meta,
    ).to(device)
    restored_model.load_state_dict(ckpt["model_state"])
    restored_model.eval()

    return {
        "cases"       : cases,
        "case_idx"    : case_idx,
        "statute_idx" : statute_idx,
        "court_idx"   : court_idx,
        "judge_idx"   : judge_idx,
        "graph_data"  : graph_data,
        "model"       : restored_model,
    }