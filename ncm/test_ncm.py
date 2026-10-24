import torch, pytest
from multiscale import LiftingFilterBank, ResidualVQ
from moe import SparseMoE
from memory import ContextMemory


@pytest.mark.parametrize("T", [8, 17])
def test_filterbank_perfect_reconstruction_random_weights(T):
    torch.manual_seed(0)
    fb = LiftingFilterBank(16).double()
    for p in fb.parameters():  # arbitrary learned weights must still reconstruct
        p.data.normal_(0, 1)
    x = torch.randn(2, T, 16, dtype=torch.double)
    s, d = fb.analyze(x)
    assert s.shape[1] == (T + 1) // 2
    err = (fb.synthesize(s, d, T) - x).abs().max().item()
    assert err < 1e-10, err


def test_rvq_error_monotone_and_decode_matches():
    torch.manual_seed(0)
    rvq = ResidualVQ(8, levels=4, codes=64)
    z = torch.randn(1, 200, 8)
    errs = [(rvq(z, n)[0] - z).pow(2).mean().item() for n in (1, 2, 3, 4)]
    assert all(a >= b - 1e-6 for a, b in zip(errs, errs[1:])), errs  # nearest-code => nonincreasing
    zh, codes, loss = rvq(z)
    assert torch.allclose(rvq.decode(codes), zh, atol=1e-5) and loss.item() > 0


def test_moe_capacity_weights_and_determinism():
    torch.manual_seed(0)
    m = SparseMoE(16, 32, n_experts=4, k=2, capacity_factor=0.5).eval()
    x = torch.randn(64, 16)
    y1 = m(x)
    assert (m.stats["load"] <= m.stats["capacity"]).all() and m.stats["dropped"] > 0
    assert torch.equal(y1, m(x))
    assert torch.isfinite(y1).all() and m.aux_loss.item() >= 1.0 - 1e-4  # >=1, =1 iff perfectly balanced


def test_capture_reader_layout_and_truncation(tmp_path):
    import numpy as np
    from capture import load_layers
    a = np.arange(3 * 3 * 2 * 4, dtype="<f4")  # 3 positions, 2 layers+input, hc 2, embd 4 (synthetic layout check only)
    p = tmp_path / "l.bin"; a.tofile(p)
    x = load_layers(p, n_layers=2, hc=2, n_embd=4)
    assert x.shape == (3, 3, 2, 4) and x[1, 0, 0, 0] == 24
    assert load_layers(p, 2, 2, 4, layers=[2]).shape == (3, 1, 2, 4)
    a[:-1].tofile(p)
    with pytest.raises(ValueError):
        load_layers(p, 2, 2, 4)


def test_routing_reader(tmp_path):
    import numpy as np
    from capture import load_routing
    k, p = 3, tmp_path / "r.bin"
    rec = lambda layer, ids, w: np.concatenate([np.array([layer, k], "<i4"), np.array(ids, "<i4"), np.array(w, "<f4").view("<i4")])
    np.concatenate([rec(0, [5, 6, 7], [.5, .3, .2]), rec(47, [1, 2, 3], [0, 0, 0])]).tofile(p)
    l, ids, w = load_routing(p)
    assert l.tolist() == [0, 47] and ids[0].tolist() == [5, 6, 7] and abs(w[0].sum() - 1) < 1e-6
    np.concatenate([rec(48, [1, 2, 3], [0, 0, 0])]).tofile(p)  # layer outside 0..47 must be refused
    with pytest.raises(ValueError):
        load_routing(p)


def test_spec_residual_reader_keeps_only_valid_rows(tmp_path):
    import numpy as np
    from capture import load_spec_residuals
    hc, ne, T = 2, 3, 3
    with open(tmp_path / "s.bin", "wb") as f:
        for p, nv in ((0, 3), (3, 1)):  # second window: only the first row is true
            f.write(np.array([p, T, nv], "<i8").tobytes())
            f.write(np.arange(T, dtype="<i4").tobytes() + (np.arange(T, dtype="<i4") + 10).tobytes())
            f.write((np.arange(T * hc * ne, dtype="<f4") + 100 * p).tobytes())
    pos, tok, nxt, rows = load_spec_residuals(tmp_path / "s.bin", hc, ne)
    assert pos.tolist() == [0, 1, 2, 3] and tok.tolist() == [0, 1, 2, 0] and nxt.tolist() == [10, 11, 12, 10]
    assert rows.shape == (4, hc, ne) and rows[3, 0, 0] == 300
    (tmp_path / "t.bin").write_bytes((tmp_path / "s.bin").read_bytes()[:-4])
    with pytest.raises(ValueError):
        load_spec_residuals(tmp_path / "t.bin", hc, ne)


def test_memory_provenance_isolation_and_spill(tmp_path):
    torch.manual_seed(0)
    mem = ContextMemory(str(tmp_path), ram_budget_segments=2)
    key = torch.randn(8)
    for i in range(5):
        e = torch.randn(4, 8)
        if i == 1:
            e = key.expand(4, 8).clone()
        mem.add("A", i * 4, i * 4 + 4, e, f"segment {i}: code 0x{i:04x}")
    mem.add("B", 0, 4, key.expand(4, 8).clone(), "other conversation secret")
    hits = mem.retrieve("A", key, k=1)
    sc, seg, text = hits[0]
    assert (seg.seg_id, seg.start, seg.end) == (1, 4, 8) and text == "segment 1: code 0x0001"
    assert seg.text is None and seg.spill_path  # was spilled, still recoverable exactly
    assert all(h[1].conv == "A" for h in mem.retrieve("A", key, k=9))
    mem.invalidate("A", 1)
    assert all(h[1].seg_id != 1 for h in mem.retrieve("A", key, k=9))
