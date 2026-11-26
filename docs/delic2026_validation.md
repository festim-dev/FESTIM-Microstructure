# Validating the codimension-one model against patterned hydrogenography

Delic, Manhard and von Toussaint (2026), Nucl. Fusion,
doi:10.1088/1741-4326/aec6b0, image where deuterium leaves the back of
recrystallised W foils exposed to a 5 eV D plasma (6e19 D m^-2 s^-1, 96 h) on
the front. An array of indicator dots (5 um wide, 10 um pitch) records the
permeation map. The three observations a grain-boundary model has to explain:

1. At 600 K, in the 50 um foil (grain size / thickness AR 1.1), only dots on
   boundaries discolour.
2. At 600 K, in the 25 um foil (AR 1.6), grain centres discolour too, with a
   depleted band next to each boundary: the *inverse permeation halo*.
3. At 630 K and above, in the 50 um foil, flux spreads from some boundaries
   into the grain beside them: the *permeation halo*, often on one side only.

Three scripts do the comparison: `examples/delic2026_crosssection.py` (model and
synthetic indicator), `examples/delic2026_fit.py` (fit to digitised line
profiles) and `examples/delic2026_temperature.py` (fit to the fraction of
active boundaries against temperature).

## Model

A 2D cross-section through columnar grains, which is what the paper's SEM
fractographs show for the 25 and 50 um foils (the 100 um foil has a layered 3D
network and is out of scope here).

- One species per grain, lattice diffusivity Frauenfelder / sqrt(2).
- One codimension-one network for the boundaries, width `delta = 1 nm`,
  diffusivity `D_gb`, exchanging with each grain through
  `k (c_grain - c_gb / s)` (`fm.exchange`).
- Front: diffusion-limited implantation, `c = Gamma (1 - r) R_p / D` in every
  grain. Each boundary mouth has its own feed `m`: `c_gb = m s c_front`, with
  `m = 0` a closed mouth (no condition, no flux), `m = 1` a mouth in equilibrium
  with the lattice beside it, `m > 1` a mouth fed harder than the lattice.
- Back: the indicator as a perfect sink for every species.

Only two combinations of the boundary parameters matter: the conductance length
`lam = delta s D_gb / D` and the fin length `ell = sqrt(delta s D_gb / (2 k))`.
With `m = 1` the lattice and the network carry linear profiles and exchange
nothing, so every halo, inverse or not, needs a mouth out of equilibrium.

**Bulk flux.** Not fitted. Kapser et al. (2018), Nucl. Fusion 58 056027,
measured 1.7e14 D m^-2 s^-1 through 24.5 um of recrystallised W in the same
device and flux, identical at 300 K and 450 K, which they attribute to
diffusion-limited surfaces. With their reflection (0.9) that pins `R_p` to
0.69 nm. Bulk inventories over 96 h are then 5.8e19 D m^-2 for 25 um and
2.9e19 for 50 um, both below the ~1e20 detection limit the paper quotes for
yttrium. Everything in the model is linear in this source, so the fitted
transport parameters and feeds do not depend on it; only the fitted indicator
thresholds scale with it.

**Indicator.** `postprocessing.hydrogenography` integrates the back-surface
flux over each dot (lattice flux exactly on the P1 facets, plus any boundary
mouth inside the dot), converts the inventory to a grayscale change that is
zero below a detection limit and saturates above a second threshold, and
normalises per image as the paper does.

**Numerical checks.** The mesh is graded towards every boundary and both
surfaces, because the network relaxes to the lattice over `ell`, which can be
well below a micrometre. D is conserved to better than 1e-4 through a
mid-depth cut in every run of the scans; dot inventories change by about 1 %
from 0.25 to 0.125 um cells (0.2 % at the 700 K best fit).

## Fit

Line profiles read from Fig. 11(a, b) (600 K, 25 and 50 um) and Fig. 10(b)
(700 K, three profiles in the 50 um foil), to about +-0.03. Model and data are
normalised to the left boundary dot; in-grain dots and the right boundary dot
enter the misfit with sigma = 0.05. The scan solves a 9 x 6 grid of
`D_gb / D` (1e2 to 1e6) and `k` (1e-7 to 1e-2 m/s), three solves per profile
(closed mouths, each mouth fed alone), and caches the dots; the feeds and the
indicator response are then fitted by brute force over the cache, since
everything is linear in the feeds.

| | 600 K (2 profiles, 13 dots) | 700 K (3 profiles, 17 dots) |
|---|---|---|
| best chi2 | 26 | 32 |
| `D_gb / D` | 3e5 (valley: >= 3e4) | 1e3 (flat: anything >= 1e3) |
| `k` (m/s) | 1e-4 (valley: >= 1e-5) | 1e-2 (flat) |
| `lam`, `ell` | 316 um, 16 um | 1 um, 0.15 um |
| front-mouth feed `m` | 0.13, shared | left 2.4e3, right 0.4e3 to 1e3 |
| mouth intake / ions on a 1 nm strip | 0.1 (the closed-mouth estimate's own error is ~0.01) | 1.7 to 2 (1 to 18 in other variants) |
| detection, saturation (D m^-2) | 4.0e19, 7.5e19 | 3.1e19, 3.9e19 |

## What is robust, and what is not

Constraining the indicator's dynamic range (saturation at least 3x or 10x the
detection limit, or no saturation at all) and refitting gives:

- **600 K: closed mouths and strong short circuits.** In every variant
  `m` is 0 to 0.13 and `D_gb / D >= 3e5`; in every variant that fits well
  (chi2 <= 50) `k >= 1e-4` too. The chi2 map has a valley (chi2 26 to 36) for
  `D_gb / D >= 3e4` and `k >= 1e-5`, i.e. `lam` well beyond the foil
  thickness and `ell` below a grain width. The inverse halo is
  a strongly coupled network the plasma cannot feed directly, draining the
  lattice next to it. The fit needs the indicator to saturate within about 2
  to 4 times its detection limit (chi2 26 to 33; 50 at 10 times); with a
  linear indicator it degrades to 75.
- **700 K: over-fed mouths, transport unconstrained.** Every variant needs the
  halo side's mouth fed hundreds to thousands of times above equilibrium,
  taking in 1 to 18 times the ion flux that lands on a 1 nm strip. Chi2 is
  32 to 34 for any saturation (62 with a linear indicator) and 32 to 38 over
  most of the grid: the halos do not pin `D_gb` or `k`. Within this model the halo is a statement about the front mouths: they
  collect D from a strip wider than the boundary (surface diffusion), or from
  the radicals, which outnumber the ions a hundredfold.
- **The indicator thresholds** come out at 3e19 to 5e19 D m^-2: between the
  50 um and 25 um bulk inventories, as the absence and presence of in-grain
  signal at 600 K require. They scale with the bulk-flux calibration.

## Temperature series, 300 to 600 K

Table 2 of the paper counts the boundary dots that discolour in the 50 um foil:
6.6 % at 300 K, 35 % at 450 K, 36 % at 500 K, 42 and 45 % at 600 K, with no
in-grain signal below 630 K. `examples/delic2026_temperature.py` asks what
spread of boundaries produces that, keeping the 600 K result: closed mouths,
strong coupling (`k = 1e-2`), and the 600 K indicator (detection 4.0e19,
saturation 7.5e19, so a dot counts above 4.4e19 D m^-2).

**Response.** Each boundary is tagged as its own entity and given its own
`D_gb` (`diffusivity_by_entity`). In 10 random sections (87 boundaries) with
log-uniform `lam` from 0.1 to 1000 um, a boundary dot crosses the threshold at
`lam = 2.8 um` with almost no geometric scatter: activity is a step in
conductance. With closed mouths and a temperature-independent bulk flux, that
step does not depend on temperature either.

**Counts.** If every boundary has `ln lam = a - dE / kT` with `dE` normal
(`mu`, `sigma`), the active fraction is `Phi(alpha T - beta)`; if only a
fraction `p` of the boundaries can become active at all, `p Phi(alpha T -
beta)`. Binomial fits up to 600 K:

| model | chi2 / dof | result |
|---|---|---|
| one population | 59 / 3 | rejected: rises too slowly from 300 to 450 K and keeps rising |
| capable fraction | 4.8 / 2 | `p = 0.44`; half of those active at 390 K, 10 to 90 % over 230 K |

Forward runs with conductances drawn from that fit, through the whole
indicator (ten dot rows per section, per-image normalisation, 10 % threshold),
give 12.5, 40, 40 and 50 % at 300, 450, 500 and 600 K, against 6.6, 35, 36 and
42 to 45 %, within the sampling noise of about 54 boundaries per temperature.

**What it means.**

- About 44 % of the boundaries can carry D, and those switch on between
  roughly 300 and 500 K. The rest stay shut up to 600 K. The EBSD of the
  paper finds about 26 % low-angle and 10 % CSL boundaries, so 44 % is neither
  "all random high-angle" (64 %) nor their complement (36 %); EBSD of the
  imaged areas would settle which boundaries they are.
- The counts fix only `alpha` and `beta`. Pinning the third number with the
  600 K profile fit (active boundaries at `lam = 316 um`, or 30 um at the edge
  of the valley) gives `dE = 0.46 +- 0.10 eV` (or `0.23 +- 0.05 eV`) above the
  lattice, but with a prefactor `D0_gb s / D0` of about 1e9 (or 2e6). That is
  implausible for a diffusivity, so the switch-on is more likely an entry or
  detrapping barrier than boundary diffusion itself.
- Above 600 K the capable population levels off at 44 % while the data climb
  to 56 to 70 %: more boundaries join, together with the halos, which closed
  mouths cannot make. That is the change of mechanism the authors describe.

**Kapser.** The same population in 25 um sections (grains 39 +- 19 um) gives a
total permeation of 1.0 to 1.1 times the bulk at 300 K and 1.4 (`lam = 30 um`)
to 1.7 to 2.8 (`lam = 316 um`, two random draws) at 450 K: 40 to 180 % more
D through the foil at 450 K than at 300 K. Kapser et al. report the two
temperatures as indistinguishable, but give only a combined flux and their
confidence bands graphically. Either their uncertainty is that wide, or the
boundaries of the 25 um foil are fewer or weaker at 450 K than the 50 um
counts imply, or the mouths are not closed at the lower temperatures. Their
per-temperature slopes would decide it.

## Tensions and caveats

- **Kapser's total flux.** On a 25 um foil whose boundaries all behave like
  the 600 K fit, the total permeation is 2 to 4.3 times the bulk, carried
  70 to 87 % by the boundaries. With only the fitted 44 % capable and the
  temperature series' switch-on, the model still predicts 40 to 180 % more
  permeation at 450 K than at 300 K (see above), against Kapser's
  "indistinguishable".
- **Binarisation.** The paper's text describes a percentile threshold, but its
  counts and line profiles read as a threshold on the normalised range. This
  matters for counting dots, not for the line-profile fit above.
- **Geometry.** The profiles run obliquely across the grains, so real
  distances to the boundary are shorter than plotted; the cross-section is 2D
  and columnar; one dot row misses boundaries that fall between dots.
- **Steady state.** The paper's 96 h exposures are assumed steady. The authors
  read the halos as B-type kinetics, which is a transient regime; a transient
  run would test whether 96 h is long enough at 630 to 700 K.

## Next steps

1. Kapser's per-temperature permeation slopes (300 K and 450 K), to settle
   the tension above.
2. EBSD of the imaged areas, which the authors say they are acquiring: test
   whether the 44 % capable boundaries are a misorientation class, then
   assign parameters by class and predict which boundaries light up. The
   package already reads `.ctf` exports.
3. 3D: a foil slab with columnar grains (extruded EBSD or a Voronoi slab) and
   a 2D dot array, which needs a slab-shaped Voronoi domain and an extrusion
   of EBSD meshes.
4. Calibration of the WO3 indicator (detection and saturation against
   temperature), which the paper flags as missing and which this fit says
   matters.

## Reproduce

```
python examples/delic2026_fit.py scan --T 600   # ~15 min
python examples/delic2026_fit.py scan --T 700   # ~25 min, can run in parallel
python examples/delic2026_fit.py fit            # writes fig_fit.png and fit_summary.json
python examples/delic2026_crosssection.py       # the two illustrative studies
python examples/delic2026_temperature.py response   # ~4 min; needs fit_summary.json
python examples/delic2026_temperature.py fit
python examples/delic2026_temperature.py check      # ~5 min
```

Outputs go to `examples/results/delic2026_crosssection/`.
