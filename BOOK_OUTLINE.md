# Cutting the Stereo Cat
## A practical algebra for stereo space, generative beats, and the Endless Engine
*Working title. Alternates: "Halving the Cat", "Total Beats: Building a Generative Rhythm Engine You Can Audit".*

**Status of this document.** The outline, the explanations and the code below are complete and the code runs: 18 automated tests pass and the
demo renders audio (`python run_tests.py`, `python demo.py out/`). But the algebra is a **reconstruction** from a one-line summary of your
paper (`T = P∘R∘C` over M/D/F with seven recipes). Your DeepSeek export, `Vintage_Hum.txt` and the Melting Blue zips were not available in this session, so the definitions
of M, D, F and of your seven recipes are *mine*, marked `[YOURS TO CONFIRM]`, and the seven recipes in the code are examples, **not** the paper's. Send the sources and I will replace
my definitions with yours and let the tests show what changed.

## 0. The pitch in one paragraph
Stereo width is usually handled by feel. This book treats it as a small, testable algebra: split a mix by frequency (F), express each band as a Middle and a
Difference signal (M/D), and move it with three operators: **Cut** (scale M and D: halve the side, or collapse to mono), **Rotate** (turn the M/D plane: an
energy-preserving change of width and polarity) and **Place** (balance left and right). Composed as `T = P∘R∘C` they give seven repeatable recipes. The same
discipline is applied to a seeded, auditable generative beat engine (the Endless Engine) and to the "Total Beats" build prompt behind it: every claim in the book is tied to a test you can run.

**Audience:** producers and mix engineers who code a little; generative-music developers; students of audio DSP. **Not** a book about AI voice cloning, and it makes no claim about any artist.

## 1. Chapter outline
Each chapter: purpose, core ideas, code, exercise. `Code:` points to the files in this package.

### Part I. Foundations
1. **Stereo is a space, not two channels.** L/R as coordinates; why width and mono-compatibility are the same question. *Exercise:* fold a mix to mono and measure what vanished.
2. **Middle and Difference.** `M=(L+R)/2`, `D=(L−R)/2`, `L=M+D`, `R=M−D` (exact inverse). Mono = D is zero; out-of-phase = M is zero. *Code:* `to_md`, `from_md`.
3. **Splitting by frequency without damaging the signal.** Complementary raised-cosine crossovers whose weights sum to exactly 1, so doing nothing returns the input (error 1.8e-15 in the tests). The crossover-overlap rule. *Code:* `band_weights`, `split_bands`.
4. **Measuring a stereo image.** Per-band level, width = rms(D)/rms(M), L/R correlation; the mono-fold-down loss. *Code:* `band_report`.

### Part II. The Stereo Cat Koan
5. **The proverb and its era.** `[YOURS: research]` Origin and earliest sources of the proverb; the street, police, insect and rat context you want covered (including black rats). *Rule for this chapter: every historical claim carries a citation or is cut.* No history is asserted in this draft.
6. **The algebra `T = P∘R∘C`.** Definitions, order of composition, and why the operators do not commute (rotation acts on M/D, placement on L/R). *Code:* `apply`, `Recipe`, `BandOp`. *Test:* `test_operators_do_not_commute`. (Worked in section 2 below.)
7. **Cut: halving the side.** Scaling D by g multiplies width by exactly g; g=0 is mono. *Test:* halving gives width ratio 0.5 to 1e-9.
8. **Rotate.** An orthogonal rotation preserves energy; 90° turns a mono signal into pure out-of-phase. When rotation helps and when it wrecks mono compatibility.
9. **Place.** Constant-power balance, identity at the centre.
10. **The seven recipes.** `[YOURS TO CONFIRM]` the paper's seven; the code ships seven *examples* (`mono_bass`, `halve_side`, `widen_air`, `rotate_image`, `cat_split`, `hard_collapse`, `tilt_place`).
11. **When a recipe does nothing, and when it backfires.** Measured below: `mono_bass` on an already-mono bass changes nothing; `cat_split` makes the highs anti-correlated.

### Part III. The Endless Engine
12. **A beat grammar.** Steps, patterns, Euclidean rhythms (`E(3,8)` is the tresillo), rotation. *Code:* `euclid`, `rotate_pattern`.
13. **Time feel.** Swing as a fraction of a step; humanize in milliseconds; **half-time** (stretch the pattern over twice the time) versus **halving the density** (drop every other onset). `[YOURS: which does "halving" mean in your terms?]` *Code:* `half_time`, `halve_density`.
14. **Voices.** Kick (pitch sweep), snare (tone + partly shared noise), hat (decorrelated noise, therefore wide). *Code:* `render`.
15. **Seeds and determinism.** Same seed, same samples; why that makes generative music testable and shareable.
16. **Recipes on buses.** Run `T` per bus (drums, bass, air). *Code:* `demo.py`.
17. **Evaluating generative output.** What can be measured (timing, width, spectrum, repeatability) and what only listeners can judge; a listening protocol.

### Part IV. Total Beats
18. **The build prompt.** `[YOURS]` How the Melting Blue // Neural Tide 2030 complete build prompt is structured, section by section.
19. **From prompt to auditable pipeline.** Evidence-first engineering: a hash-chained ledger, tests that fail on broken code, honest status labels (the AOC practice).
20. **Case study: Melting Blue // Neural Tide.** `[YOURS: needs the zips and the sub-bass drone fix notes]`.

### Part V. Practice and responsibility
21. **Mixing for mono, phones, clubs.** Using the report numbers to decide.
22. **Voices, samples and disclosure.** Consent records, provenance notes, labelling synthetic audio. (Not legal advice; point readers to a rights professional.)
23. **Publishing your engine.** Licensing, what to keep out of a repository (large binaries, audio, secrets), release hygiene.

**Appendices:** A. Code map. B. Derivations (partition of unity, rotation is orthogonal, width scaling). C. The test suite as specification. D. Glossary. E. Reproducibility checklist.

## 2. Explanation: Chapter 6 written out (the algebra, with the real code)
**Band split F.** Each crossover at edge `e` is a ramp `s(f) = (1 − cos(π t))/2` with `t = clip(log2(f/e)/w + 0.5, 0, 1)`, `w` = width in octaves. The band weights are
`w_0 = 1 − s_0`, `w_b = s_{b−1} − s_b`, `w_last = s_last`, so `Σ_b w_b(f) = 1` for every frequency (a telescoping sum). Multiplying the spectrum by each `w_b` and summing the
inverse transforms returns the input exactly. Neighbouring crossovers must be at least `w` octaves apart, otherwise a weight goes negative; the code rejects that.

**C, R, P in each band** (`stereo_cat.py`):
```python
def cut(m, d, gm, gd):     return m * gm, d * gd                              # C
def rotate(m, d, deg):     c, s = cos(r), sin(r); return c*m - s*d, s*m + c*d # R, an orthogonal matrix
def place(lr, pan):        a = (pan + 1) * pi / 4                              # P, constant power, pan 0 = (1, 1)
                           return lr * [sqrt(2)*cos(a), sqrt(2)*sin(a)]
# apply(): for each band:  m, d = to_md(band);  m, d = cut(...);  m, d = rotate(...);  out += place(from_md(m, d), pan)
```
**What the tests prove** (each is a test in `tests/test_book_code.py`):
- *Identity.* A recipe with no change reconstructs the input to 1e-9 (measured 1.8e-15).
- *Rotation is energy-preserving.* `rotate` is orthogonal, so `Σ(M²+D²)` is unchanged for any angle; at 90° a mono signal becomes `L = m, R = −m` (sums to zero in mono).
- *Cut scales width exactly.* Width = rms(D)/rms(M); scaling D by g scales width by g. `halve_side` gives 0.5 to 1e-9; `hard_collapse` gives `L == R`.
- *Place is constant-power.* For a mono band, `L²+R²` is the same at every pan setting.
- *Non-commutativity.* Rotating then placing differs from placing then rotating (max difference > 1e-3 in the test), because rotation is defined on (M, D) and placement on (L, R).

**Worked measurement** (`python demo.py`: 4 bars at 92 BPM, swing 0.12, humanize 4 ms, seed 7; bands split at 250 Hz and 2.5 kHz):
| Band | raw beat width / corr | `cat_split` | `halve_side` |
|------|----------------------|-------------|--------------|
| 0-250 Hz | 0.01 / +1.00 | 0.00 / +1.00 | 0.01 / +1.00 |
| 250-2500 Hz | 0.66 / +0.39 | 0.72 / +0.32 | 0.33 / +0.80 |
| 2.5 kHz-22 kHz | 0.92 / +0.08 | 1.28 / **−0.25** | 0.46 / +0.65 |
Reading it honestly: (1) the bass was already mono, so `mono_bass` leaves this beat unchanged (identical to the raw row); (2) `cat_split` widens the highs but their correlation turns negative, which
means part of the highs will cancel when the track is folded to mono: a real cost, measurable before you ship; (3) `halve_side` halves width in every band. These are measurements of the signal, not of how it sounds.

**One more honest limit.** Re-measuring a processed signal with the same crossover double-counts the transition region: after `mono_bass`, the band report at the 120 Hz edge still shows ~10% of the original width, while the bands well inside each side show 0.7% (below 60 Hz) and 100% (above 240 Hz) as designed.

## 3. Explanation: the Endless Engine in miniature (`endless_beats.py`)
- `euclid(k, n)` returns the k-onset, n-step rhythm with onsets as evenly spaced as integers allow (the test checks the gap sizes differ by at most one for every k ≤ n ≤ 24; `E(3,8)` equals the tresillo `x..x..x.`).
- `render(patterns, bpm, bars, swing, humanize_ms, seed)` places hits on a grid; odd steps are delayed by `swing × step`; humanize adds seeded Gaussian jitter. The test measures the swing shift to within 3 samples.
- Voices are synthesized with numpy only: kick = exponential pitch sweep 150→45 Hz; snare = 190 Hz tone + partly shared noise; hat = high-passed, *decorrelated* noise (the test confirms a centred kick has width ~0 and the hat is wide), which is what gives the stereo recipes something to work on.
- Determinism: same seed gives bit-identical audio; a different seed with humanize on differs.

## 4. Running it
```
python run_tests.py      # 18 tests
python demo.py out/      # writes demo_beat.wav and three processed versions, prints the band reports
```
Requires Python 3.9+ and numpy. Add a recipe by constructing `Recipe(name, edges, ops)` with one `BandOp` per band.

## 5. What is verified, what is not
- **Verified (run here):** the 18 tests; the reconstruction, energy, width, constant-power and determinism properties above; the measured demo table.
- **Not verified:** that the definitions match your paper; any listening result (nobody has listened to the demo files); the proverb's history; the "Total Beats" and Melting Blue chapters (sources not available).
- **Needed from you to finish the book:** the DeepSeek "Stereo Cat Koan Project" export (paper, algebra, seven recipes, Python engine), `Vintage_Hum.txt`, the Melting Blue zips, your meaning of "halving", and citations for the proverb chapter.

## 6. A short proposal paragraph you can send
*"Cutting the Stereo Cat" is a practical, test-backed guide to stereo-field processing and generative rhythm. It presents a small algebra (frequency bands, Middle/Difference components, and three operators: cut, rotate, place), seven repeatable recipes, and a seeded beat engine, with runnable code and an automated test suite for every claim. Twenty-three chapters in five parts; a working code package and measured examples are available now. Author: Lewis Seals.*
