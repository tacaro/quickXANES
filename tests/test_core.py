"""Core checks on synthetic spectra with known answers, plus a real-file smoke test when data are present."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qxanes.lcf import lcf_combinatorial  # noqa: E402
from qxanes.normalize import normalize  # noqa: E402
from qxanes.processing import average, deadtime_factor, find_e0, find_feature  # noqa: E402

# real-data tests run when QXANES_TEST_DATA points at a folder of beamtime scans (e.g. SSRL/bl143_nov26/xas)
DATA = os.environ.get("QXANES_TEST_DATA",
                      os.path.join(os.path.dirname(__file__), "..", "..", "SSRL", "bl143_nov26", "xas"))


def grid():
    # coarse pre-edge, dense edge, medium post-edge, like a typical XANES region file
    return np.unique(np.concatenate([np.arange(4800, 4960, 10.0), np.arange(4960, 5010, 0.25),
                                     np.arange(5010, 5200, 2.0)]))


def edge(E, e0, width=1.5):
    from scipy.special import erf
    return 0.5 * (1 + erf((E - e0) / width))


def test_gaussian_tail_normalization_recovers_unit_step():
    E = grid()
    e0 = 4980.0
    tail = 0.4 + 3.0 * np.exp(-0.5 * ((E - 4870) / 60.0) ** 2)   # scatter peak tailing through the edge
    mu = tail + 0.8 * edge(E, e0) + 0.0005 * np.clip(E - e0, 0, None)
    r = normalize(E, mu, {"pre_model": "gaussian", "pre_range": [None, -15.0], "post_range": [30.0, None],
                          "post_degree": 1, "flatten": True})
    assert abs(r["e0"] - e0) < 0.5
    assert r["pre"]["model"] == "gaussian" and abs(r["pre"]["Ec"] - 4870) < 5
    assert abs(r["edge_step"] - 0.8) < 0.02
    pre = E < e0 - 20
    post = E > e0 + 40
    assert abs(np.mean(r["flat"][pre])) < 0.01
    assert abs(np.mean(r["flat"][post]) - 1.0) < 0.01


def test_lcf_recovers_fractions_without_forcing_sum():
    E = grid()
    a = edge(E, 4978) + 0.3 * np.exp(-0.5 * ((E - 4982) / 1.5) ** 2)
    b = edge(E, 4984)
    c = edge(E, 4990) + 0.2 * np.exp(-0.5 * ((E - 4995) / 3) ** 2)
    rng = np.random.default_rng(1)
    y = 0.3 * a + 0.6 * b + rng.normal(0, 0.002, E.size)          # sums to 0.9 on purpose
    fits, warn = lcf_combinatorial(E, y, {"A": (E, a), "B": (E, b), "C": (E, c)}, (4960, 5030), 3)
    assert not warn
    best = min((f for f in fits if len(f["components"]) == 2), key=lambda f: f["r_factor"])
    assert set(best["components"]) == {"A", "B"}
    assert best["p_vs_simpler"] < 1e-6          # B on top of A is clearly justified
    for f in fits:                               # adding C to A+B is not
        if set(f["components"]) == {"A", "B", "C"}:
            assert f["p_vs_simpler"] > 0.01 or f["weights"][f["components"].index("C")] < 0.01
    w = dict(zip(best["components"], best["weights"]))
    assert abs(w["A"] - 0.3) < 0.01 and abs(w["B"] - 0.6) < 0.01
    assert abs(best["sum"] - 0.9) < 0.01
    assert all(x >= 0 for f in fits for x in f["weights"])


def test_deadtime_paralyzable_round_trip():
    tau_us = 0.5
    r_true = np.array([1e3, 1e5, 5e5])
    r_meas = r_true * np.exp(-r_true * tau_us * 1e-6)
    f, sat = deadtime_factor(r_meas, 1.0, tau_us, "paralyzable")
    assert np.allclose(r_meas * f, r_true, rtol=1e-6) and not sat.any()
    f0, _ = deadtime_factor(r_meas, 1.0, 0.0)
    assert np.all(f0 == 1)


def test_calibration_features_and_average():
    E = grid()
    mu = edge(E, 4980) + 0.5 * np.exp(-0.5 * ((E - 4985.3) / 1.0) ** 2)
    assert abs(find_feature(E, mu, "peak", (4975, 4995)) - 4985.3) < 0.15
    assert abs(find_e0(E, edge(E, 4980.0)) - 4980.0) < 0.2
    avg = average([(E, mu), (E[::2], mu[::2])])
    assert avg["energy"].size == E.size and np.nanmax(np.abs(avg["mu"] - mu)) < 0.05


@pytest.mark.skipif(not os.path.isdir(DATA), reason="beamtime data not present")
def test_real_data_xdi_dat_parity_and_pipeline(tmp_path):
    from qxanes.readers import read_scan
    from qxanes.session import Session
    a = read_scan(os.path.join(DATA, "ChlorophenolRed_post_001_001.xdi"))
    b = read_scan(os.path.join(DATA, "ChlorophenolRed_post_001_001.dat"))
    assert all(np.allclose(a.columns[c], b.columns[c]) for c in a.colnames if c in b.columns)
    S = Session()
    S.load_dir(DATA, str(tmp_path))
    for s in S.stems():
        if s.startswith(("ChlorophenolRed_post", "NaCl_2500", "w6_XPAH9")):
            S.params["files"][s]["queued"] = True
    S.params["roles"]["ChlorophenolRed_post_001"] = "calibration"
    S.params["roles"]["NaCl_2500_001"] = "standard"
    S.params["calibration"]["reference_eV"] = 2821.2
    S.calibrate()
    out = S.run_all()
    assert out["normalized"] and out["lcf"]
    p = S.save_params(str(tmp_path / "p.json"))
    S2 = Session.from_json(p)
    r1 = S.normalized("w6_XPAH9_005")
    r2 = S2.normalized("w6_XPAH9_005")
    assert np.allclose(r1["flat"], r2["flat"])   # params JSON reproduces the result


@pytest.mark.skipif(not os.path.isdir(DATA), reason="beamtime data not present")
def test_per_file_channel_exclusion_and_queue_removal():
    from qxanes.session import Session
    S = Session()
    S.load_dir(DATA, os.path.join(os.path.dirname(__file__), "_tmp_out"))
    stem = "s4_w35_ascent_1_11_25_700_002_001"
    for s in S.stems():
        S.params["files"][s]["queued"] = s.startswith("s4_w35_ascent_1_11_25_700_002")
    scan = S.scan(stem)
    S.params["files"][stem]["exclude_channels"] = ["FF1"]
    assert S.used_channels(stem) == ["FF2", "FF3", "FF4"]
    res = S.signal(stem, calibrated=False)
    E = scan.energy
    _, idx = np.unique(E, return_index=True)
    expect = sum(scan.columns[c][idx] for c in ("FF2", "FF3", "FF4")) / scan.columns["I0"][idx]
    assert np.allclose(res["mu"], expect)
    # other replicates keep all channels
    assert S.used_channels("s4_w35_ascent_1_11_25_700_002_002") == ["FF1", "FF2", "FF3", "FF4"]
    # removing one replicate from the queue drops it from its group's average
    g = S.params["files"][stem]["group"]
    assert len(S.group_data(g)["used"]) == 3
    S.params["files"]["s4_w35_ascent_1_11_25_700_002_003"]["queued"] = False
    assert S.group_members(g) == [stem, "s4_w35_ascent_1_11_25_700_002_002"]
    assert len(S.group_data(g)["used"]) == 2


def test_legacy_channel_include_list_is_migrated():
    from qxanes.session import Session
    S = Session({"signal": {"channels": ["FF1", "FF2", "FF3", "FF4"]},
                 "files": {"a": {"group": "a", "channels": ["FF2", "FF3"]}, "b": {"group": "b", "channels": None}}})
    assert S.params["files"]["a"]["exclude_channels"] == ["FF1", "FF4"]
    assert S.params["files"]["b"]["exclude_channels"] == []
    assert "channels" not in S.params["files"]["a"]


@pytest.mark.skipif(not os.path.isdir(DATA), reason="beamtime data not present")
def test_locked_in_lcf_is_what_gets_exported(tmp_path):
    import pandas as pd
    from qxanes.session import Session
    S = Session()
    S.load_dir(DATA, str(tmp_path))
    for s in S.stems():
        if s.startswith(("NaCl_2500", "w3CPA_2500", "w9Phe_2500", "w6_XPAH9", "w7_XPAH11")):
            S.params["files"][s]["queued"] = True
    for g in ("NaCl_2500_001", "w3CPA_2500_002", "w9Phe_2500_003"):
        S.params["roles"][g] = "standard"
    _, fits, _ = S.lcf("w6_XPAH9_005")
    second = fits[1]["components"]                      # deliberately not the lowest-R fit
    S.lcf_select("w6_XPAH9_005", second)
    S.write_lcf()
    res = pd.read_csv(tmp_path / "lcf" / "lcf_results.csv")
    assert list(res["sample"]) == ["w6_XPAH9_005"]      # undecided w7_XPAH11 is left out
    assert set(res.loc[0, "components"].split(" + ")) == set(second)
    assert res.loc[0, "rank_by_r_factor"] == 2
    cand = pd.read_csv(tmp_path / "lcf" / "lcf_candidates.csv")
    assert cand["selected"].sum() == 1 and set(cand.loc[cand["selected"], "sample"]) == {"w6_XPAH9_005"}
    assert set(cand["sample"]) == {"w6_XPAH9_005", "w7_XPAH11_004"}
    assert (tmp_path / "lcf" / "fits" / "w6_XPAH9_005_selected.csv").exists()
    assert S.lcf_undecided() == ["w7_XPAH11_004"]
    # the decision survives the params JSON and is refit identically
    S2 = Session.from_json(S.save_params(str(tmp_path / "p.json")))
    S2.params["output_dir"] = str(tmp_path / "again")
    S2.write_lcf()
    res2 = pd.read_csv(tmp_path / "again" / "lcf" / "lcf_results.csv")
    num = res.select_dtypes("number").columns
    assert np.allclose(res[num].to_numpy(), res2[num].to_numpy(), equal_nan=True)
    # a locked combination outside the candidate list (more components than max) still resolves
    S.params["lcf"]["max_components"] = 1
    S.lcf_select("w6_XPAH9_005", ["NaCl_2500_001", "w9Phe_2500_003"])
    _, fits1, _ = S.lcf("w6_XPAH9_005")
    f, rank, warn = S.lcf_selected_fit("w6_XPAH9_005", fits1, {})
    assert f is not None and rank is None and set(f["components"]) == {"NaCl_2500_001", "w9Phe_2500_003"}
