"""Shared pytest fixtures.

This replication uses only synthetic financial-network instances (paper
Sec. 7.1); there is no market-data path, so no ``market_data`` marker is
registered and no live API fixture is required.
"""
from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import build_network


@pytest.fixture
def small_instance():
    """A small (n=5, m=2) ER(0.4) instance, seed 0, for src/ unit tests."""
    return build_network(n=5, m=2, graph_type="ER", density=0.4, seed=0)


@pytest.fixture
def toy_two_bank():
    """A hand-built n=2, m=1 instance with a known default-free nominal
    system, used for exact regression checks of the clearing primitives."""
    p_bar_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    p_bar = np.array([1.0, 1.0])
    c_bar = np.array([1.0, 1.0])
    s_matrix = np.array([[1.0], [0.5]])
    return {"p_bar": p_bar, "A": a, "c_bar": c_bar, "S": s_matrix, "P_bar": p_bar_matrix}
