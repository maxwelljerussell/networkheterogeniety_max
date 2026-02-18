from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Literal, Optional
import json
import numpy as np
from datetime import datetime


@dataclass(frozen=True)
class NetworkGrid:
    """
    Network-level hyperspectrum grid.
    Independent of batch or dataset.
    """
    network_group_id: str          # e.g. "heterogeneity_A"
    network_index: str             # "0000"

    max_extent: float              # e.g. 50.0
    n_y: int                       # e.g. 1000

    ks: np.ndarray                 # shape (K,)
    ys: np.ndarray                 # shape (n_y,)

    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            object.__setattr__(
                self,
                "created_at",
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
            )

        object.__setattr__(self, "ks", np.asarray(self.ks, dtype=np.float32))
        object.__setattr__(self, "ys", np.asarray(self.ys, dtype=np.float32))

        if self.ks.ndim != 1 or self.ys.ndim != 1:
            raise ValueError("ks and ys must be 1D arrays")

        if self.ys.shape[0] != self.n_y:
            raise ValueError(
                f"n_y mismatch: expected {self.n_y}, got {self.ys.shape[0]}"
            )

    # ---------- persistence ----------

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "network_group_id": self.network_group_id,
            "network_index": self.network_index,
            "max_extent": self.max_extent,
            "n_y": self.n_y,
            "created_at": self.created_at,
        }

        np.savez_compressed(
            path,
            meta=json.dumps(meta, sort_keys=True),
            ks=self.ks,
            ys=self.ys,
        )

    @staticmethod
    def from_npz(path: Path) -> "NetworkGrid":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))

        return NetworkGrid(
            network_group_id=meta["network_group_id"],
            network_index=meta["network_index"],
            max_extent=float(meta["max_extent"]),
            n_y=int(meta["n_y"]),
            ks=z["ks"],
            ys=z["ys"],
            created_at=meta.get("created_at", ""),
        )

@dataclass(frozen=True)
class Hyperspectrum:
    """
    One flattened hyperspectrum sample.

    sample_kind:
      - "28x28" : one hyperspectrum per image
      - "7x7"   : concatenation of 16 patch hyperspectra per image
    """

    # Network identity
    network_group_id: str
    network_index: str
    batch_id: str                # still needed for provenance

    # Sample identity
    sample_kind: Literal["28x28", "7x7"]
    img_idx: int
    label: int

    # 7x7 only
    patch_global_ids: Optional[np.ndarray] = None  # shape (16,)

    # Parameters (cache safety)
    max_extent: float = 50.0
    n_y: int = 1000
    hs_params: Dict[str, Any] = None
    post_params: Dict[str, Any] = None

    # Data (always flattened)
    feat: np.ndarray = None      # shape (F,)

    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            object.__setattr__(
                self,
                "created_at",
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
            )

        object.__setattr__(self, "hs_params", dict(self.hs_params or {}))
        object.__setattr__(self, "post_params", dict(self.post_params or {}))

        f = np.asarray(self.feat, dtype=np.float32)
        if f.ndim != 1:
            raise ValueError(f"feat must be 1D flattened, got {f.shape}")
        object.__setattr__(self, "feat", f)

        if self.sample_kind == "7x7":
            if self.patch_global_ids is None:
                raise ValueError("7x7 sample requires patch_global_ids")
            pg = np.asarray(self.patch_global_ids, dtype=np.int32)
            if pg.shape != (16,):
                raise ValueError("patch_global_ids must have shape (16,)")
            object.__setattr__(self, "patch_global_ids", pg)
        else:
            if self.patch_global_ids is not None:
                raise ValueError("28x28 sample must not have patch_global_ids")
            object.__setattr__(self, "patch_global_ids", None)

    # ---------- persistence ----------

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "network_group_id": self.network_group_id,
            "network_index": self.network_index,
            "batch_id": self.batch_id,
            "sample_kind": self.sample_kind,
            "img_idx": self.img_idx,
            "label": self.label,
            "max_extent": self.max_extent,
            "n_y": self.n_y,
            "hs_params": self.hs_params,
            "post_params": self.post_params,
            "created_at": self.created_at,
        }

        arrays = {"feat": self.feat}
        if self.patch_global_ids is not None:
            arrays["patch_global_ids"] = self.patch_global_ids

        np.savez_compressed(
            path,
            meta=json.dumps(meta, sort_keys=True),
            **arrays,
        )

    @staticmethod
    def from_npz(path: Path) -> "Hyperspectrum":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))

        return Hyperspectrum(
            network_group_id=meta["network_group_id"],
            network_index=meta["network_index"],
            batch_id=meta["batch_id"],
            sample_kind=meta["sample_kind"],
            img_idx=int(meta["img_idx"]),
            label=int(meta["label"]),
            patch_global_ids=z["patch_global_ids"] if "patch_global_ids" in z else None,
            max_extent=float(meta["max_extent"]),
            n_y=int(meta["n_y"]),
            hs_params=dict(meta.get("hs_params", {})),
            post_params=dict(meta.get("post_params", {})),
            feat=z["feat"],
            created_at=meta.get("created_at", ""),
        )
