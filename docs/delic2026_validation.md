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

Five scripts do the comparison: `examples/delic2026_crosssection.py` (model and
synthetic indicator), `examples/delic2026_fit.py` (fit to digitised line
profiles), `examples/delic2026_temperature.py` (fit to the fraction of active
boundaries against temperature), `examples/delic2026_transient.py` (is 96 h
steady state, with traps) and `examples/delic2026_3d.py` (a synthetic 3D foil
under the hexagonal dot array).

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
  of the valley), and using the transient thresholds of the next section,
  gives `dE = 0.31 +- 0.07 eV` (or `0.09 +- 0.02 eV`) above the lattice, with
  a prefactor `D0_gb s / D0` of about 1e8 (or 2e5). (At steady state: 0.46 or
  0.23 eV, prefactors 1e9 or 2e6.) Prefactors that large are implausible for a
  diffusivity, so the switch-on is more likely an entry or detrapping barrier
  than boundary diffusion itself.
- Above 600 K the capable population levels off at 44 % while the data climb
  to 56 to 70 %: more boundaries join, together with the halos, which closed
  mouths cannot make. That is the change of mechanism the authors describe.

**Kapser.** The same population in 25 um sections (grains 39 +- 19 um) gives a
total permeation of 1.0 to 1.1 times the bulk at 300 K and, at 450 K, 1.3
(`lam = 30 um`) to 2.9 (`lam = 316 um`) with the transient thresholds (1.4 to
2.8 with the steady-state ones): 20 to 180 % more D through the foil at 450 K
than at 300 K. Kapser et al. report the two temperatures as
indistinguishable, but neither the paper nor Kapser's thesis (IPP 2019-01)
gives the two slopes or their uncertainties: only the combined flux, with
confidence bands drawn in a figure, and scatter they attribute to
microstructure. A 20 % rise could hide in that; a 180 % rise could not. So
either the active boundaries sit at the weak end of the 600 K valley
(`lam` of tens of um), or the boundaries of the 25 um foil are fewer or
weaker at 450 K than the 50 um counts imply, or the mouths are not closed at
the lower temperatures. Their per-temperature slopes would decide it.

## Is 96 h steady state?

Everything above assumes, with the paper, that a dot's inventory is its
steady flux times 96 h. `examples/delic2026_transient.py` tests that with the
background traps of Kapser's own model of these foils: 1.5e-5 traps per W
atom, 1.5 eV, 1e13 s^-1. Kapser fitted his permeation data only from 48 h on
because of the time lag.

**Bulk.** Those traps saturate below 500 to 550 K, and the implanted D fills
them first. With a diffusion-limited source the filled zone crosses a foil
of thickness `L` after `N_t L^2 / (2 Gamma (1 - r) R_p)`, whatever the
temperature: 20 h for 25 um, 79 h for 50 um, 315 h for 100 um. 1D FESTIM
runs (lattice plus traps) give, for the D collected by an in-grain dot after
96 h as a fraction of the steady-state value:

| foil | 300 to 500 K | 600 K | 630 to 700 K |
|---|---|---|---|
| 25 um | 0.79 to 0.81 | 0.98 | 1.00 |
| 50 um | 0.17 to 0.24 | 0.92 | 0.98 to 1.00 |
| 100 um | 0 | 0.67 | 0.91 to 1.00 |

So 96 h is steady at 600 K and above in the 25 and 50 um foils, and is not
at 500 K and below in anything thicker than 25 um. In the 100 um foil the
lattice delivers nothing in 96 h up to 500 K.

**With a boundary.** 2D runs, one boundary between two half grains of the
50 um foil, closed mouth, the same traps in the grains (trapping and
de-trapping both slowed a hundredfold so that the mesh resolves the filling
front; 1D runs agree with the unscaled rates to about 1 %):

- At 600 K the dots of the line-profile best fit are within 0.2 to 6 % of
  their steady-state values in both foils, below the digitisation error. The
  600 K and 700 K fits stand.
- At 300 to 500 K a boundary has to fill the traps beside it before it
  delivers, so the dot on it needs `lam = 12 um` to cross the indicator
  threshold in 96 h, against 3.2 um at 600 K. (The boundary dots of a 300 K
  and a 450 K run agree to 1 %: in the saturating regime temperature drops
  out.)
- While the traps fill, the lattice beside an active boundary breaks through
  earlier than the grain centre: the dot 10 um away collects up to 1.5 times
  what the centre does. That is a weak halo, but far below the detection
  limit and nowhere near the size of the 700 K halos.

**Consequences for the temperature series.** Refitting the counts with the two
thresholds changes little: the capable fraction stays at 0.43 (chi2 4.5 for
2 degrees of freedom), half of it active by 390 K, and the 300, 450 and
500 K points alone, which share one threshold, still reject a single
population (chi2 14 for 1 degree of freedom). What moves are the pinned
energies and the Kapser prediction, both given above.

Not tested: deeper or denser traps, traps in the boundaries themselves, and
the trap evolution under the plasma that Kapser reports at 300 K. Deeper
traps would push the non-steady range to higher temperatures.

## A synthetic 3D foil

Everything above is a 2D cross-section read by a line of dots.
`examples/delic2026_3d.py` builds a 3D foil instead: a random Voronoi
tessellation of the surface, relaxed towards equal-sized grains of the
paper's mean size and extruded through the thickness (columnar, as the
fractographs show for the 25 and 50 um foils), meshed with tetrahedra graded
towards the boundaries and both surfaces. Every boundary plane is its own
tagged entity. The microstructure is invented: it stands in for the EBSD of
the imaged areas. The parameters are not refitted: the 600 K line-profile
fit (`lam = 316 um`, `k = 1e-4 m/s`, mouth feed 0.13, detection 4.0e19 and
saturation 7.5e19 D m^-2) on a random 44 % of the boundaries, the rest closed
and non-conductive. The back surface is read by the real array: 5 um discs on
a hexagonal grid of 10 um pitch, each collecting the lattice flux over its
disc and the boundary outflow along the chord of trace it covers
(`hydrogenography.hexagonal_centres`, `disc_signals`).

**Check.** One planar boundary through a 3D slab is the 2D problem. Against
the 2D cross-section on its fine mesh, the boundary outflow agrees to 0.1 %
and the lattice flux to 0.7 % beyond 10 um from the boundary and to 2 to 4 %
within it (the 3D surface mesh is 1.5 to 4 um). D through a mid-depth plane
and out of the back agree to 0.1 to 0.7 % in every 3D run.

**Maps at 600 K** (`fig_3d.png`, to set beside the paper's Fig. 11):

- 50 um foil (21 grains, 107 000 nodes, under a minute): only dots on
  conductive boundaries light up, and no in-grain dot does, as in Fig. 11(b).
- 25 um foil (26 grains): grains fenced by closed boundaries glow at up to
  0.5 of the boundary dots (median of the lit in-grain dots 0.34; the paper's
  profile shows 0.3 to 0.45), while the lattice beside a conductive boundary
  is dark: the inverse halo of Fig. 11(a), here as a map. Grains ringed by
  conductive boundaries stay dark.
- The boundaries carry 60 to 64 % of the permeation; the foils pass 1.6 to
  1.8 times the bulk flux.

**Counts.** Over six random 50 um microstructures (538 boundary dots), 43.5
+- 2.8 % of the boundary dots discolour and no in-grain dot does; Table 3
gives 44.9 +- 2.1 % and none. That is not an independent test, since the
44 % was fitted to those counts through the 2D model. What the 3D run adds is
that the dot array does not bias the count: 99.7 % of the dots touching a
conductive boundary are seen, however short the chord (the indicator
saturates at twice its detection limit), and none of those touching only
closed boundaries. The discoloured fraction therefore measures the
conductive fraction of the boundary length directly, and scatters with it
from one microstructure to the next (31 to 56 % here, for fields of about 90
boundary dots; the paper counts about 1000).

**Limits.** Which boundaries conduct is drawn at random, independently of
their neighbours and of any misorientation; real ones will be correlated
with grain orientation. The foil is columnar, so it says nothing about the
100 um foil. Steady state only, which the transient runs support at 600 K.

## Tensions and caveats

- **Kapser's total flux.** On a 25 um foil whose boundaries all behave like
  the 600 K fit, the total permeation is 2 to 4.3 times the bulk, carried
  70 to 87 % by the boundaries. With only the fitted 44 % capable and the
  temperature series' switch-on, the model still predicts 20 to 180 % more
  permeation at 450 K than at 300 K (see above), against Kapser's
  "indistinguishable".
- **Binarisation.** The paper's text describes a percentile threshold, but its
  counts and line profiles read as a threshold on the normalised range. This
  matters for counting dots, not for the line-profile fit above.
- **Geometry.** The profiles run obliquely across the grains, so real
  distances to the boundary are shorter than plotted, and the fits are to a
  2D columnar cross-section. The 3D foil removes the dot-row limitation but
  its microstructure is synthetic.
- **Steady state.** Holds at 600 K and above for the 25 and 50 um foils with
  Kapser's traps, and fails below (see the section above). The authors read
  the halos as B-type kinetics, a transient regime; with these traps the
  lattice is steady within minutes to a few hours at 630 to 700 K, so the
  transient does not make the 700 K halos.

## Next steps

1. Kapser's per-temperature permeation slopes (300 K and 450 K), to settle
   the tension above. They are not in the paper or the thesis, so this is a
   question for the authors.
2. EBSD of the imaged areas, which the authors say they are acquiring: test
   whether the 44 % capable boundaries are a misorientation class, then
   assign parameters by class and predict which boundaries light up. The
   package already reads `.ctf` exports.
3. 3D with the measured microstructure: `delic2026_3d.py` extrudes any set
   of grain polygons, so the EBSD grain map of an imaged area can replace the
   Voronoi one; and refit the line profiles on 3D grains, at their true
   distances from the boundaries. A non-columnar slab for the 100 um foil.
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
python examples/delic2026_transient.py bulk           # ~2 min
python examples/delic2026_transient.py section        # ~15 min
python examples/delic2026_transient.py consequences   # ~3 min; needs the two above
python examples/delic2026_3d.py check        # ~1 min
python examples/delic2026_3d.py foils        # ~2 min, writes fig_3d.png
python examples/delic2026_3d.py statistics   # ~3 min
```

Outputs go to `examples/results/delic2026_crosssection/`.
