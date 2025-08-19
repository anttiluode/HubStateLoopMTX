# HubStateLoopMTX

Vibecoded stuff.. 

Tiny experiments around a minimal “language” for complex systems based on Hubs (h), States (s) and Loops (l).
The repo contains small, self-contained demos (HTML + JS) and a couple of Python notebooks/scripts that visualize/solve things using the MTX idea.

What is MTX (one-liner)

h — hub / reset / broadcast: short, high-impact events that reconfigure behavior.

s — state / search: transitional, exploratory motion.

l — loop / persistence: stable cycles that maintain a pattern.

Example token string: h0{0.2s} l3{2.0s} s1{0.6s}

“send a quick hub, hold a loop for 2s, then explore for 0.6s”.

# Contents

hslcity.html – “Fractal Night City”: nested H/S/L growth patterns as a city-like visualization (click to seed).

viral_mtx.html – simple entrainment/synchrony demo driven by H/S/L rhythms.

smartflyXx.html – a small agent/organism sketch that emits MTX tokens while moving in a field.

city6.html – another minimal growth/attractor toy using the same MTX grammar.

mtx_poisson.py – Poisson/field demo that uses MTX events to steer a solver and plot residuals.

conversation about mtx with ai.txt – notes/background.

# SyntheticEEG.py: 

script generates synthetic, EEG-like multichannel data from MTX (h/s/l) token sequences:
It samples a token timeline with a simple semi-Markov process (loops, hubs, states), then converts it to MTX text.
It renders an 8-channel signal at 250 Hz by summing band-limited sinusoids (δ/θ/α/β/γ) with different channel topographies,
producing something EEG-ish (alpha stronger occipitally, beta/gamma frontally).

It writes:
synthetic_slx.wav (mono preview mixdown)
synthetic_slx_signal.npz (arrays: sig [8×N], fs, t, tokens, mtx)
Run it with python synthetic_eeg.py and you’ll see the printed MTX string and the two files saved.
(File names may evolve; each HTML file is standalone.)

How to run

# HTML demos

Quick: double-click the .html file to open in a modern browser (Chrome/Edge/Firefox).

If you hit CORS issues: start a tiny local server in the folder:

python -m http.server 8000

then open http://localhost:8000/hslcity.html (or another page).

Some pages may ask for camera permission (for interactive/pose demos). Allow it if you want the full effect.

# Python demo

Requirements: python 3.9+, numpy, matplotlib (and optionally scipy if prompted).

pip install numpy matplotlib scipy

python mtx_poisson.py

# Why this exists

To test whether a tiny, human-readable token stream (H/S/L) can:

couple different systems (agents ↔ fields ↔ visuals),

create entrainment between things,

and provide a simple “control law” for exploration (s), stabilization (l), and resets (h).

# Extend

Add a new interpreter that maps h/l/s to your domain (sound, UI, robotics).

Log your system as MTX tokens and route them between demos.

PRs with tiny, focused examples are welcome.

# License

MIT — do whatever you want; attribution appreciated.
