"""pytest fixtures shared across the integration suite."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg_max(root: Path) -> dict:
    return yaml.safe_load(open(root / "configs/cpu_max.yaml"))


@pytest.fixture(scope="session")
def processed(root: Path) -> Path:
    return root / "data/processed"


# Large processed arrays that are rebuilt locally rather than committed.
REBUILT_BY = {
    "geophysics_patches.npy": "python -m src.data.process_geophysics",
    "train_mask.npy": "python -m src.data.build_labels",
}


@pytest.fixture(scope="session")
def require_data(processed: Path):
    """Skip a test, naming the rebuild command, when it needs processed data not in the repo."""
    def check(*names: str) -> None:
        missing = [n for n in names if not (processed / n).exists()]
        if missing:
            how = "; ".join(f"{n}: {REBUILT_BY.get(n, 'see the data pipeline')}" for n in missing)
            pytest.skip(f"needs processed data that is not committed ({how})")
    return check


@pytest.fixture(scope="session")
def labels(processed: Path) -> np.ndarray:
    return np.load(processed / "labels.npy")


@pytest.fixture(scope="session")
def grid_coords(processed: Path):
    import pandas as pd
    return pd.read_csv(processed / "grid_coordinates.csv")


@pytest.fixture(scope="session")
def cpu_max_state(root: Path):
    ck = torch.load(root / "outputs/checkpoints/random_cpu_max_seed42.pt",
                    map_location="cpu", weights_only=False)
    return ck["state_dict"]


@pytest.fixture(scope="session")
def cpu_max_model(cfg_max, cpu_max_state):
    from src.models.geoprospectnet import GeoProspectNet
    m = GeoProspectNet(cfg_max).to("cpu")
    m.load_state_dict(cpu_max_state)
    m.eval()
    return m
