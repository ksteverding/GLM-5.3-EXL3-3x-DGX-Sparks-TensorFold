"""Full GLM-5.3's kept prompts on disk under --parallel (0147: TF_GLM_DISK_CACHE in ``multi.MultiDecoder``), on CPU with
the tiny checkpoint of ``full_fakes``: kept prompts are written in the background as they are kept; a request whose
prompt starts with a state on disk, but not in memory (after a restart, or after its kept prompt was evicted), loads it
into a fresh extent and resumes from it. Every reply must equal the same request served alone (``Serial``), with
DSpark's window loaded too; a write whose extent moves starts again from the new rows; and over three ranks a state one
rank cannot read is loaded by none (every rank forgets its file)."""

import multiprocessing as mp
import os
import threading

import pytest
import torch

from full_fakes import install_fake_experts, write_checkpoint
from test_full_multi_engine import (EXT, Host, Serial, _env, check, drive, follow, load_w, replies, request, sampled,
                                    tokens, write_dspark)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("glm53multidisk"))


@pytest.fixture(scope="module")
def dspark(tmp_path_factory):
    return write_dspark(tmp_path_factory.mktemp("dspark") / "d")


@pytest.fixture
def fakes(monkeypatch):
    install_fake_experts(monkeypatch)
    _env(monkeypatch)
    for k in ("TF_GLM_MULTI_PREFILL", "TF_GLM_MULTI_VERIFY", "TF_GLM_MULTI_DEPTH", "TF_GLM_FILL_BUDGET_MS",
              "TF_GLM_DSPARK_MULTI_POLICY", "TF_GLM_FILL_ROWS"):
        monkeypatch.delenv(k, raising=False)


class DiskHost(Host):
    """``Host`` with the engine's disk tier and its ok-flag all-gather (``GlmEngine._gather_ints``)."""

    def __init__(self, *a, disk=None, **kw):
        super().__init__(*a, **kw)
        self.disk = disk

    def _gather_ints(self, values):
        if self.comm is None:
            return [list(values)]
        mine = torch.tensor(values, dtype=torch.int32)
        got = torch.zeros((self.world * len(values),), dtype=torch.int32)
        self.comm.all_gather(mine, got)
        return got.view(self.world, len(values)).tolist()


def cache(root, rank=0, world=1):
    from tensorfold.families.glm5_next.cuda.pcache import DiskCache

    return DiskCache(root, "test", rank=rank, world=world, cp=1, kv="bf16", threads=2, keep_free=0)


def build(folder, streams, disk, *, w=None, rank=0, world=1, comm=None, dspark=None, entries=8):
    """A ``MultiDecoder`` with a fresh pool of streams + 2 extents (as ``test_full_multi_engine.multi``) and ``disk``."""
    from tensorfold.families.glm5_next.cuda import decode
    from tensorfold.families.glm5_next.cuda.multi import MultiDecoder

    w = w if w is not None else load_w(folder, rank, world, comm)
    pool = (streams + 2) * EXT
    drafter = None
    if dspark is not None:
        from tensorfold.families.glm5_next.cuda.dspark import Drafter

        drafter = Drafter(dspark, w, capacity=pool, tap_rows=16, ring=True)
    e = decode.Engine(w, capacity=pool, max_rows=16, prefill_rows=64, graphs=False, long_context=True,
                      taps=drafter.tap_layers if drafter is not None else (), streams=streams, pool_rows=pool,
                      kv="bf16")
    host = DiskHost(w, e, drafter, rank, world, comm, limit=EXT - 16, disk=disk)
    host.cache_entries = entries
    return MultiDecoder(host, streams), host


def test_restart_resumes_from_disk_equal_serial(folder, dspark, fakes, tmp_path):
    """Two DSpark requests keep their prompts (on disk too); after a restart (a fresh pool, a new DiskCache on the
    same folder) one prompt extended, the other repeated (its head row on disk) and a new one: the first two load
    their states from disk (DSpark window included), and every reply equals serial."""
    a, b = tokens(40, 61), tokens(26, 62)
    first = [request(a, 6), request(b, 5, sampled(17), at=1)]
    disk = cache(tmp_path / "pc")
    dec, host = build(folder, 2, disk, dspark=dspark)
    streams = drive(dec, host, first)
    assert disk.writer.drain(60)
    assert disk.stats["written"] == 2 and disk.stats["hits"] == 0, disk.stats
    ref = Serial(dec.w)
    check(replies(streams), [ref.reply(r) for r in first], first)
    disk.close()

    disk = cache(tmp_path / "pc")                         # the restart: the index read back, nothing in memory
    assert len(disk.entries) == 2
    dec, host = build(folder, 2, disk, w=dec.w, dspark=dspark)
    again = [request(a + tokens(12, 63), 7, sampled(5)), request(b, 5, sampled(17)), request(tokens(15, 64), 4, at=2)]
    streams = drive(dec, host, again)
    assert streams[0].cached == 40 and streams[0].glm_lane["disk_cached"] == 40, streams[0].cached
    assert streams[1].cached == 26, "the repeated prompt replays from its state on disk (its head row)"
    assert streams[2].cached == 0
    assert disk.stats["hits"] == 2, disk.stats
    assert sum(s.drafted for s in streams) > 0, "the resumed DSpark streams drafted"
    assert dec.health()["disk_cache"]["hits"] == 2
    check(replies(streams), [ref.reply(r) for r in again], again)
    disk.close()


def test_evicted_prompt_reloads_from_disk(folder, fakes, tmp_path):
    """One kept prompt at a time (TF_GLM_CACHE_ENTRIES=1): the second request's state evicts the first one's, whose
    write ends first; a later request extending the first prompt loads it back from disk; replies equal serial."""
    a, b = tokens(33, 71), tokens(20, 72)
    reqs = [request(a, 4), request(b, 4, at=1), request(a + tokens(9, 73), 6, sampled(3), at=12)]
    disk = cache(tmp_path / "pc")
    dec, host = build(folder, 2, disk, entries=1)
    streams = drive(dec, host, reqs)
    assert streams[2].cached == 33, streams[2].cached
    assert disk.stats["hits"] == 1 and disk.stats["cancelled"] == 0, disk.stats
    ref = Serial(dec.w)
    check(replies(streams), [ref.reply(r) for r in reqs], reqs)
    disk.close()


def test_moved_extent_restarts_its_write(folder, fakes, tmp_path):
    """A write still waiting when its kept prompt's extent moves is given up and started again from the new rows: the
    old rows overwritten after the move, the file holds the state; a restart resumes from it as serial."""
    from tensorfold.families.glm5_next.cuda.pcache import Cancelled

    disk = cache(tmp_path / "pc")
    gate = threading.Event()
    write = disk._write

    def held(job):                                        # the writer waits for the gate (or a cancel)
        while not gate.wait(0.01):
            if job.cancel:
                raise Cancelled()
        return write(job)

    disk._write = held
    a = tokens(37, 81)
    dec, host = build(folder, 2, disk)
    streams = drive(dec, host, [request(a, 3)])
    (c,) = dec.kept
    x = c.extent
    old, base = x.base, dec.pool.place(x.size)
    assert base is not None and (base >= old + x.size or base + x.size <= old)
    dec._move(x, base, x.size)
    for p in dec.arena.planes:                            # the old rows reused by something else
        p.tensor[old // p.div:(old + x.size) // p.div].fill_(3)
    gate.set()
    assert disk.writer.drain(60)
    assert disk.stats["cancelled"] == 1 and disk.stats["written"] == 1, disk.stats
    disk.close()

    disk = cache(tmp_path / "pc")
    dec, host = build(folder, 2, disk, w=dec.w)
    reqs = [request(a + tokens(8, 82), 5)]
    got = drive(dec, host, reqs)
    assert got[0].cached == 37
    check(replies(got), [Serial(dec.w).reply(r) for r in reqs], reqs)
    assert replies(streams)[0] == Serial(dec.w).reply(request(a, 3))
    disk.close()


# -- three ranks ---------------------------------------------------------------------------------------------------
def _rank(folder, rank, world, port, root, out_q):
    import torch.distributed as dist

    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)
    import pytest as _pt

    mp_ = _pt.MonkeyPatch()
    install_fake_experts(mp_)
    _env(os.environ)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    ranks = world

    class Comm:
        world_size = ranks
        world = ranks

        def all_gather(self, send, recv):
            dist.all_gather(list(recv.view(ranks, -1).unbind(0)), send.contiguous().view(-1))

    try:
        a, b = tokens(30, 91), tokens(22, 92)
        mine = root / f"rank{rank}"                        # each rank its own disk
        disk = cache(mine, rank, world)
        dec, host = build(folder, 2, disk, rank=rank, world=world, comm=Comm())
        first = [request(a, 4), request(b, 4, at=1)]
        if rank == 0:
            drive(dec, host, first)
        else:
            follow(dec, host)
        assert disk.writer.drain(120) and disk.stats["written"] == 2, disk.stats
        if rank == 2:                                      # rank 2 lost its file of b
            disk.path(disk.key(b)).unlink()
        host._gather_ints([1])                             # every rank wrote (and rank 2 lost) before the restart
        disk.close()

        disk = cache(mine, rank, world)
        dec, host = build(folder, 2, disk, w=dec.w, rank=rank, world=world, comm=Comm())
        again = [request(a + tokens(7, 93), 5, sampled(44)), request(b + tokens(6, 94), 5, at=1)]
        if rank == 0:
            streams = drive(dec, host, again)
            got = {i: list(s.out[:s.count]) for i, s in enumerate(streams)}
            sids = {i: s.sid for i, s in enumerate(streams)}
            cached = [s.cached for s in streams]
        else:
            seen = follow(dec, host)
            got, sids, cached = {sid: list(s.out[:s.count]) for sid, s in seen.items()}, None, None
        files = (disk.contains(disk.key(a)), disk.contains(disk.key(b)))
        ref = Serial(dec.w)
        out_q.put((rank, got, sids, cached, files, [ref.reply(r) for r in again], None))
        disk.close()
    except Exception:                                    # noqa: BLE001
        import traceback

        out_q.put((rank, None, None, None, None, None, traceback.format_exc()))
        raise
    finally:
        dist.destroy_process_group()


def test_three_ranks_agree_on_disk_states(folder, fakes, tmp_path):
    """Three ranks (rank 0 deciding, the others applying its messages, each rank its own disk folder): after a
    restart a state every rank holds is loaded on all; a state rank 2 lost is loaded by none, and every rank forgets
    its file of it; every rank's replies equal serial."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 39000 + os.getpid() % 2000
    procs = [ctx.Process(target=_rank, args=(folder, r, 3, port, tmp_path, q)) for r in range(3)]
    for p in procs:
        p.start()
    got = []
    try:
        while len(got) < len(procs):
            got.append(q.get(timeout=2900))
            if got[-1][6] is not None:
                break
    finally:
        for p in procs:
            p.join(timeout=60 if len(got) == len(procs) and got[-1][6] is None else 1)
            if p.is_alive():
                p.terminate()
    got.sort(key=lambda x: x[0])
    for rank, *_, err in got:
        assert err is None, f"rank {rank}: {err}"
    assert len(got) == len(procs)
    _, zero, sids, cached, _, want, _ = got[0]
    assert cached == [30, 0], cached
    for i, toks in zero.items():
        assert toks == want[i], f"rank 0 request {i}: {toks} != serial {want[i]}"
    for rank, mine, _, _, files, w_, _ in got:
        assert files == (True, False), f"rank {rank} files of a, b: {files}"
        if rank:
            assert w_ == want
            for i, sid in sids.items():
                assert mine[sid] == want[i], f"rank {rank} request {i}: {mine[sid]} != {want[i]}"
