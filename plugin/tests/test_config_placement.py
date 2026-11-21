"""Settings parsing, slot/budget math and placement plans (pure Python, CPU)."""
import json
import struct

import numpy as np
import pytest

from edge0_vllm import config as C
from edge0_vllm import placement as P

from helpers import SRC_MODEL


def test_slot_geometry():
    assert C.SLOT_BYTES == 1_935_360
    assert C.W13_SHAPE == (112, 2560) and C.W2_SHAPE == (40, 3584)
    assert C.S13_SHAPE == (112, 1280) and C.S2_SHAPE == (40, 1792)
    # DESIGN.md 3.1: 10.2 GiB -> ~5,680 slots; 15.0 GiB -> ~8,320
    assert C.hot_slots_for_gib(10.2) == 5658
    assert 8300 < C.hot_slots_for_gib(15.0) < 8330
    assert C.hot_slots_for_gib(0) == 0


def test_settings_defaults_and_validation():
    s = C.Edge0Settings.from_env({})
    assert (s.dense, s.placement, s.cold_alloc, s.split, s.moe_impl, s.embed_host) == \
        ("bf16", "profile", "host_register", True, "marlin", True)
    assert s.expert_vram_gib is None and s.trace == frozenset()
    s = C.Edge0Settings.from_env({"EDGE0_EXPERT_VRAM_GIB": "9.5", "EDGE0_TRACE": "bw,routes", "EDGE0_SPLIT": "0",
                                  "EDGE0_PLACEMENT": "plan:/x.json", "EDGE0_DENSE": "q8"})
    assert s.expert_vram_gib == 9.5 and s.trace == {"bw", "routes"} and not s.split and s.dense == "q8"
    for bad in ({"EDGE0_DENSE": "fp8"}, {"EDGE0_PLACEMENT": "random"}, {"EDGE0_TRACE": "nope"},
                {"EDGE0_SPLIT": "maybe"}, {"EDGE0_EXPERT_VRAM_GIB": "-1"}, {"EDGE0_COLD_ALLOC": "x"}):
        with pytest.raises(ValueError):
            C.Edge0Settings.from_env(bad)
    for phase3 in ({"EDGE0_STREAM_MIN_TOKENS": "512"}, {"EDGE0_COLD_SCALES": "vram"},
                   {"EDGE0_COLD_ALLOC": "vmm_host_numa"}):
        with pytest.raises(NotImplementedError):
            C.Edge0Settings.from_env(phase3)
    s = C.Edge0Settings.from_env({"EDGE0_ADAPT": "1", "EDGE0_COLD_BSM": "64", "EDGE0_COLD_BLOCKS_PER_SM": "1"})
    assert s.adapt and s.cold_bsm == 64 and s.cold_blocks_per_sm == 1 and s.hot_bsm == 0
    with pytest.raises(ValueError):
        C.Edge0Settings.from_env({"EDGE0_COLD_BSM": "24"})


def test_checkpoint_spec():
    d = {"format": "edge0-vllm-v1", "latent_dim": 1792, "latent_groups": 8, "expert_intermediate_size": 640,
         "expert_format": "modelopt_nvfp4_stacked", "ple_file": "ple-fp8.gguf", "ple_data_offset": 352,
         "ple_rows": 320001536, "ple_row_bytes": 160, "ple_scale": 0.00019931793212890625,
         "profile_file": "profile.bin", "draft_vocab_file": "draft_vocab.json", "source_dir": "/x"}
    s = C.Edge0CheckpointSpec.from_dict(d)
    assert s.ple_rows * s.ple_row_bytes == 51_200_245_760
    with pytest.raises(ValueError):
        C.Edge0CheckpointSpec.from_dict({**d, "latent_dim": 1024})
    with pytest.raises(ValueError):
        C.Edge0CheckpointSpec.from_dict(None)


def _fake_profile(path, ranked, nl=48, ne=512):
    table = np.full((nl, ne), -1, np.int32)
    for i, (l, e) in enumerate(ranked):
        table[l, e] = i
    blob = b"STRP" + struct.pack("<5I", 1, nl, ne, len(ranked), len(ranked))
    blob += np.asarray(ranked, np.uint16).tobytes() + table.tobytes()
    path.write_bytes(blob)


def test_plan_from_ranking_and_maps(tmp_path):
    rng = np.random.default_rng(0)
    pairs = [(l, e) for l in range(48) for e in range(512)]
    rng.shuffle(pairs)
    _fake_profile(tmp_path / "profile.bin", pairs)
    prof = P.read_profile(tmp_path / "profile.bin")
    plans = P.plan_from_ranking(prof["ranked"], 48, 512, 6000)
    assert sum(p.n_hot for p in plans.values()) == 6000
    hot_pairs = {(l, e) for l, p in plans.items() for e in p.order[:p.n_hot]}
    assert hot_pairs == set(map(tuple, pairs[:6000]))
    for layer, p in plans.items():
        p.validate(512)
        hm, cm = p.hot_map(), p.cold_map()
        assert sum(x >= 0 for x in hm) == p.n_hot and sum(x >= 0 for x in cm) == p.n_cold
        for s, e in enumerate(p.order):
            assert (hm[e] == s) if s < p.n_hot else (cm[e] == s - p.n_hot)
    # L4 mini checkpoint: only the first 4 layers of the same ranking
    plans4 = P.plan_from_ranking(prof["ranked"], 4, 512, 1000)
    assert sum(p.n_hot for p in plans4.values()) == 1000 and len(plans4) == 4


def test_plan_identity_and_roundtrip(tmp_path):
    plans = P.plan_identity(48, 512, 5000)
    assert sum(p.n_hot for p in plans.values()) == 5000
    assert {p.n_hot for p in plans.values()} <= {104, 105}
    P.save_plan(tmp_path / "plan.json", plans, "test")
    again = P.load_plan(tmp_path / "plan.json", 48, 512)
    assert P.placement_digest(again) == P.placement_digest(plans)
    d = json.loads((tmp_path / "plan.json").read_text())
    d["layers"]["3"]["order"][0] = d["layers"]["3"]["order"][1]
    (tmp_path / "bad.json").write_text(json.dumps(d))
    with pytest.raises(ValueError):
        P.load_plan(tmp_path / "bad.json", 48, 512)


def test_resolve_placement_budget(tmp_path):
    with pytest.raises(ValueError, match="EDGE0_EXPERT_VRAM_GIB"):
        P.resolve_placement("profile", None, tmp_path, "profile.bin", 48, 512)
    plans, desc = P.resolve_placement("identity", 0.0, tmp_path, "profile.bin", 48, 512)
    assert sum(p.n_hot for p in plans.values()) == 0           # F0: every expert cold
    P.save_plan(tmp_path / "p.json", P.plan_identity(48, 512, 6000), "t")
    with pytest.raises(ValueError, match="exceed"):
        P.resolve_placement(f"plan:{tmp_path / 'p.json'}", 1.0, tmp_path, "profile.bin", 48, 512)


@pytest.mark.skipif(not (SRC_MODEL / "profile.bin").is_file(), reason="source model folder not mounted")
def test_real_profile_top_pairs():
    prof = P.read_profile(SRC_MODEL / "profile.bin")
    assert (prof["n_layers"], prof["n_expert"], prof["n_ranked"]) == (48, 512, 24576)
    # MODEL-SPEC 5.7: the top pairs are (15,494), (15,209), (31,6), (15,6), (3,507)
    assert [tuple(x) for x in prof["ranked"][:5].tolist()] == [(15, 494), (15, 209), (31, 6), (15, 6), (3, 507)]
    plans = P.plan_from_ranking(prof["ranked"], 48, 512, C.hot_slots_for_gib(9.5))
    n = [plans[l].n_hot for l in range(48)]
    assert sum(n) == C.hot_slots_for_gib(9.5)
    assert plans[15].order[:2] == [494, 209]
    print("profile placement at 9.5 GiB: n_hot per layer min/max", min(n), max(n))
