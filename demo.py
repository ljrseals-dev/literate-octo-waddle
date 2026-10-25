"""Render a four-bar beat, run the stereo-cat recipes over it, print before/after band reports and write WAVs."""
import sys, wave
import numpy as np
import endless_beats as eb
import stereo_cat as sc

def write_wav(path, x, sr=44100):
    a = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(a.tobytes())

def show(title, rows):
    print(title)
    for r in rows:
        lo, hi = r["band_hz"]
        print(f"   {lo:7.0f}-{hi:7.0f} Hz  level {r['level_db']:6.1f} dB  width {r['width']:5.2f}  corr {r['correlation']:+.2f}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    sr = 44100
    x = eb.render(eb.default_patterns(), bpm=92, bars=4, swing=0.12, humanize_ms=4, seed=7, sr=sr)
    write_wav(f"{out}/demo_beat.wav", x, sr)
    edges = (250.0, 2500.0)
    show("raw beat", sc.band_report(x, sr, edges))
    for name in ("cat_split", "mono_bass", "halve_side"):
        y = sc.apply(x, sr, sc.example_recipes()[name]).astype(np.float32)
        y /= max(1.0, float(np.max(np.abs(y))) / 0.89)
        write_wav(f"{out}/demo_beat_{name}.wav", y, sr)
        show(f"after recipe {name!r}", sc.band_report(y, sr, edges))
