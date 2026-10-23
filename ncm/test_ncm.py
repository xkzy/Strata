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
