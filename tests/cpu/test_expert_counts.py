"""The routed-expert usage counters (0144, TF_GLM_EXPERT_COUNTS) on CPU: every row's top-k picks land in the cell of
its phase, class, layer and expert; the shared expert's slot and the MTP layer are not counted; rows are counted once a
forward; flush moves the counts into float64 totals and zeroes the device side; totals.npz and hourly snapshots hold
only expert-level numbers and a restart continues from them."""

import time

import numpy as np
import torch

from tensorfold.families.glm5_next.cuda import expert_counts as ec

L, E, K = 4, 8, 2


def make(tmp_path, monkeypatch, first_moe=1):
    monkeypatch.setenv("TF_GLM_EXPERT_COUNTS_DIR", str(tmp_path))
    return ec.ExpertCounts(L, E, K, torch.device("cpu"), first_moe)


def picks(rows):
    """pick [R, K + 1] int32 (last slot the shared expert, id E) and wts fp32."""
    pick = torch.tensor([r + [E] for r in rows], dtype=torch.int32)
    wts = torch.tensor([[0.25 * (j + 1) for j in range(K)] + [1.0] for _ in rows], dtype=torch.float32)
    return pick, wts


def test_cells_phase_class(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch)
    pick, wts = picks([[0, 3], [3, 7], [1, 0]])
    c.classes([(2, False), (1, True)])               # rows 0-1 foreground, row 2 background
    c.add(1, pick, wts, 3, prefill=True)
    c.add(2, pick, wts, 3, prefill=False)
    c.add(L, pick, wts, 3, prefill=True)             # the MTP layer: not counted
    c.flush()
    n = c.total_count
    assert n.sum() == 2 * 3 * K
    assert n[0, 0, 1, 3] == 2 and n[0, 0, 1, 0] == 1 and n[0, 0, 1, 7] == 1
    assert n[0, 1, 1, 1] == 1 and n[0, 1, 1, 0] == 1
    assert n[1, 0, 2, 3] == 2 and n[1, 1, 2, 1] == 1
    assert n[:, :, :, :].sum(axis=(2, 3)).tolist() == [[4, 2], [4, 2]]
    assert c.total_weight[0, 0, 1, 3] == 0.5 + 0.25          # row 0 slot 1, row 1 slot 0
    assert c.total_rows.tolist() == [[2, 1], [0, 0]]         # only the first MoE layer (1) counts rows
    assert c.count.sum() == 0 and c.weight.sum() == 0 and c.rows_dev.sum() == 0


def test_pattern_reused_and_replaced(tmp_path, monkeypatch):
    c = make(tmp_path, monkeypatch)
    pick, wts = picks([[2, 5], [4, 6]])
    c.classes([(2, True)])
    c.add(1, pick, wts, 2, prefill=True)
    c.classes([(2, True)])                           # same pattern: no writes
    c.add(1, pick, wts, 2, prefill=True)
    c.classes([(1, False), (1, False)])
    c.add(1, pick, wts, 2, prefill=True)
    c.flush()
    assert c.total_count[0, 1, 1].sum() == 2 * 2 * K
    assert c.total_count[0, 0, 1].sum() == 2 * K


def test_save_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_GLM_EXPERT_COUNTS_EVERY_S", "0")
    monkeypatch.setenv("TF_GLM_EXPERT_COUNTS_SNAPSHOT_S", "0")
    c = make(tmp_path, monkeypatch)
    pick, wts = picks([[0, 1]])
    c.classes([(1, False)])
    c.add(1, pick, wts, 1, prefill=True)             # startup work: dropped by the first tick
    c.tick()
    assert c.count.sum() == 0 and not (tmp_path / "totals.npz").exists()
    c.add(1, pick, wts, 1, prefill=True)
    time.sleep(0.01)
    c.tick()
    with np.load(tmp_path / "totals.npz") as z:
        assert sorted(z.files) == ["count", "experts", "image", "layers", "rows", "saved", "started", "top_k", "weight"]
        assert z["count"].shape == (2, 2, L, E) and z["count"].sum() == K
    assert len(list((tmp_path / "hourly").glob("*.npz"))) == 1
    d = make(tmp_path, monkeypatch)                  # a restart continues the totals
    assert d.total_count.sum() == K and d.total_rows.sum() == 1


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TF_GLM_EXPERT_COUNTS", raising=False)
    assert not ec.enabled()
