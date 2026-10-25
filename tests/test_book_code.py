import os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import endless_beats as eb
import stereo_cat as sc

SR = 44100
O = sc.BandOp


def noise(n=30000, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 2))


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}")


def width(x):
    m, d = sc.to_md(x)
    return np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(m ** 2))


# ---------------------------------------------------------------- stereo_cat
def test_band_weights_partition_unity_and_validation():
    f = np.fft.rfftfreq(1 << 16, 1 / SR)
    for edges in ((), (120.0,), (250.0, 2500.0), (100.0, 400.0, 1600.0, 6400.0)):
        W = sc.band_weights(f, edges)
        assert W.shape == (len(edges) + 1, len(f)) and np.allclose(W.sum(axis=0), 1.0, atol=1e-12) and W.min() >= -1e-12
    raises(ValueError, lambda: sc.band_weights(f, (500.0, 400.0)))                   # not ascending
    raises(ValueError, lambda: sc.band_weights(f, (400.0, 450.0)))                   # crossovers overlap
    raises(ValueError, lambda: sc.band_weights(f, (-5.0,)))


def test_identity_recipe_and_band_split_reconstruct_the_input():
    x = noise()
    rec = sc.Recipe("id", (250.0, 2500.0), (O(), O(), O()))
    assert np.max(np.abs(sc.apply(x, SR, rec) - x)) < 1e-9
    assert np.max(np.abs(sum(sc.split_bands(x, SR, (250.0, 2500.0))) - x)) < 1e-9


def test_rotation_preserves_energy_and_90_degrees_turns_mono_into_out_of_phase():
    x = noise()
    m, d = sc.to_md(x)
    for deg in (0, 17, 45, 90, -135, 180):
        m2, d2 = sc.rotate(m, d, deg)
        assert abs(np.sum(m2 ** 2 + d2 ** 2) - np.sum(m ** 2 + d ** 2)) < 1e-6 * np.sum(m ** 2 + d ** 2)
    assert np.allclose(sc.rotate(m, d, 0)[0], m)
    mono = np.tile(np.random.default_rng(1).standard_normal((5000, 1)), (1, 2))
    y = sc.apply(mono, SR, sc.Recipe("r", (), (O(rotate_deg=90.0),)))
    assert np.max(np.abs(y[:, 0] + y[:, 1])) < 1e-9 * 10 and np.max(np.abs(y[:, 0] - mono[:, 0])) < 1e-9 * 10     # L = m, R = -m


def test_cut_collapses_or_halves_the_side_exactly():
    x = noise()
    y0 = sc.apply(x, SR, sc.example_recipes()["hard_collapse"])
    assert np.max(np.abs(y0[:, 0] - y0[:, 1])) < 1e-9                                  # fully mono
    y1 = sc.apply(x, SR, sc.example_recipes()["halve_side"])
    assert abs(width(y1) / width(x) - 0.5) < 1e-9                                       # exactly half the width


def test_mono_bass_collapses_lows_and_leaves_highs_alone():
    # Measure inside each side of the 120 Hz crossover (report bands <60 Hz and >240 Hz). Re-splitting AT the crossover would
    # count the transition region, where the original side signal legitimately survives (about 10% residual width there).
    x = noise(60000)
    y = sc.apply(x, SR, sc.example_recipes()["mono_bass"])
    before, after = sc.band_report(x, SR, (60.0, 240.0)), sc.band_report(y, SR, (60.0, 240.0))
    assert after[0]["width"] < 0.02 * before[0]["width"]                                 # measured 0.929 -> 0.0068
    assert abs(after[2]["width"] / before[2]["width"] - 1.0) < 1e-3                      # measured 1.0000
    assert 0.0 < after[1]["width"] < before[1]["width"]                                  # the transition band is partly collapsed


def test_widen_air_raises_only_the_high_band():
    x = noise(60000)
    y = sc.apply(x, SR, sc.example_recipes()["widen_air"])
    b, a = sc.band_report(x, SR, (4000.0,)), sc.band_report(y, SR, (4000.0,))
    assert abs(a[1]["width"] / b[1]["width"] - 1.6) < 0.08 and abs(a[0]["width"] / b[0]["width"] - 1.0) < 0.02


def test_place_is_constant_power_and_identity_at_centre():
    mono = np.tile(np.random.default_rng(2).standard_normal((4000, 1)), (1, 2))
    assert np.allclose(sc.place(mono, 0.0), mono)
    e = [np.sum(sc.place(mono, p) ** 2) for p in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    assert max(e) / min(e) < 1.0 + 1e-9
    hard = sc.place(mono, -1.0)
    assert np.max(np.abs(hard[:, 1])) < 1e-9


def test_operators_do_not_commute():
    x = noise(8000)
    m, d = sc.to_md(x)
    rp = sc.place(sc.from_md(*sc.rotate(m, d, 30)), 0.4)                                # R then P (the algebra's order)
    y = sc.place(x, 0.4)
    my, dy = sc.to_md(y)
    pr = sc.from_md(*sc.rotate(my, dy, 30))                                             # P then R
    assert np.max(np.abs(rp - pr)) > 1e-3


def test_recipe_and_input_validation():
    x = noise(2000)
    raises(ValueError, lambda: sc.apply(x, SR, sc.Recipe("bad", (200.0,), (O(),))))      # too few ops
    raises(ValueError, lambda: sc.apply(x, SR, sc.Recipe("bad", (), (O(pan=2.0),))))
    raises(ValueError, lambda: sc.apply(x, SR, sc.Recipe("bad", (), (O(diff=9.0),))))
    raises(ValueError, lambda: sc.apply(x[:, 0], SR, sc.example_recipes()["halve_side"]))   # mono array
    bad = x.copy(); bad[3, 0] = np.nan
    raises(ValueError, lambda: sc.apply(bad, SR, sc.example_recipes()["halve_side"]))


def test_band_report_on_mono_and_out_of_phase():
    mono = np.tile(np.random.default_rng(3).standard_normal((30000, 1)), (1, 2))
    r = sc.band_report(mono, SR, (250.0,))
    assert all(row["width"] < 1e-9 and row["correlation"] > 0.999999 for row in r)
    r = sc.band_report(mono * np.array([1.0, -1.0]), SR, (250.0,))
    assert all(row["correlation"] < -0.999999 for row in r)


def test_all_seven_example_recipes_run_and_stay_finite():
    x = noise(40000)
    for name, rec in sc.example_recipes().items():
        y = sc.apply(x, SR, rec)
        assert y.shape == x.shape and np.isfinite(y).all(), name
    assert len(sc.example_recipes()) == 7


# ---------------------------------------------------------------- endless_beats
def test_euclid_known_patterns_and_maximal_evenness():
    assert eb.euclid(3, 8) == [1, 0, 0, 1, 0, 0, 1, 0]                                  # the tresillo
    assert eb.euclid(0, 5) == [0] * 5 and eb.euclid(5, 5) == [1] * 5
    for n in range(1, 25):
        for k in range(0, n + 1):
            p = eb.euclid(k, n)
            assert sum(p) == k and len(p) == n
            if k:
                on = [i for i, v in enumerate(p) if v]
                gaps = [(on[(j + 1) % k] - on[j]) % n or n for j in range(k)]
                assert max(gaps) - min(gaps) <= 1, (k, n)                                # as even as integers allow
    raises(ValueError, lambda: eb.euclid(9, 8))
    raises(ValueError, lambda: eb.euclid(1, 0))


def test_half_time_and_halve_density():
    p = [1, 0, 1, 1, 0, 1, 0, 0]
    h = eb.half_time(p)
    assert len(h) == 2 * len(p) and sum(h) == sum(p) and [i for i, v in enumerate(h) if v] == [2 * i for i, v in enumerate(p) if v]
    assert eb.half_time(p, 1) == p
    d = eb.halve_density(p)
    assert len(d) == len(p) and sum(d) == (sum(p) + 1) // 2 and all(a <= b for a, b in zip(d, p))
    raises(ValueError, lambda: eb.half_time(p, 0))
    assert eb.rotate_pattern([1, 0, 0, 0], 1) == [0, 1, 0, 0] and eb.rotate_pattern([1, 0, 0, 0], 4) == [1, 0, 0, 0]


def test_render_is_deterministic_bounded_and_the_right_length():
    pats = eb.default_patterns()
    a = eb.render(pats, seed=5, humanize_ms=5, bars=2)
    b = eb.render(pats, seed=5, humanize_ms=5, bars=2)
    c = eb.render(pats, seed=6, humanize_ms=5, bars=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.dtype == np.float32 and a.ndim == 2 and a.shape[1] == 2 and np.max(np.abs(a)) <= 0.9 + 1e-6
    bar_s = 60 / 92 * 4
    assert abs(len(a) / SR - (2 * bar_s + 1.0)) < 0.01


def test_swing_delays_odd_steps_by_the_stated_fraction():
    pat = {"kick": [0, 1] + [0] * 14}                                                    # one kick on an odd step
    straight = eb.render(pat, bpm=120, bars=1, swing=0.0)[:, 0]
    swung = eb.render(pat, bpm=120, bars=1, swing=0.25)[:, 0]
    step = 60 / 120 * 4 / 16
    first = lambda y: int(np.argmax(np.abs(y) > 0.05 * np.max(np.abs(y))))
    assert abs((first(swung) - first(straight)) - 0.25 * step * SR) <= 3


def test_kick_is_darker_than_hat_and_hat_is_wide():
    k = eb.render({"kick": [1] + [0] * 15}, bars=1)
    h = eb.render({"hat": [1] + [0] * 15}, bars=1)
    cent = lambda y: float((np.abs(np.fft.rfft(y[:, 0])) * np.fft.rfftfreq(len(y), 1 / SR)).sum() / np.abs(np.fft.rfft(y[:, 0])).sum())
    assert cent(k) < 400 < cent(h)
    assert width(k.astype(np.float64)) < 1e-6 and width(h.astype(np.float64)) > 0.5          # centred kick, decorrelated hat


def test_render_validation():
    raises(ValueError, lambda: eb.render({"kick": [1] * 16}, bpm=5))
    raises(ValueError, lambda: eb.render({"cowbell": [1] * 16}))
    raises(ValueError, lambda: eb.render({"kick": [1] * 7}))
    raises(ValueError, lambda: eb.render({"kick": [1] * 16}, swing=0.9))


def test_demo_script_runs_and_shows_mono_bass_effect():
    d = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo.py"), d],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    for n in ("demo_beat", "demo_beat_cat_split", "demo_beat_mono_bass", "demo_beat_halve_side"):
        assert os.path.getsize(os.path.join(d, n + ".wav")) > 100000, n
    assert "after recipe 'mono_bass'" in r.stdout
