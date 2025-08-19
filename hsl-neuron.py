# MTX Homeostatic Cognitive Unit (HCU) demo
# - One oscillator tries to keep novelty & coherence in a sweet spot
# - It lives in a 2D wave field it can sense & perturb
# - It emits MTX tokens (h = hub/reset, l = loop/lock, s = state/transient)
#
# Output:
# 1) Animated GIF of the field with the HCU moving and MTX readout
# 2) Phase portrait image (signal vs delayed signal) with token colors
# 3) CSV of emitted tokens over time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from collections import deque
import pandas as pd
import os

# ---------------------------
# Utilities
# ---------------------------
rng = np.random.default_rng(42)

def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, 1, 0) + np.roll(Z, -1, 0)
        + np.roll(Z, 1, 1) + np.roll(Z, -1, 1)
    )

# ---------------------------
# MTX Port
# ---------------------------
class MtxPort:
    def __init__(self, win=60, delay=15):
        self.win = win
        self.delay = delay
        self.buf = deque(maxlen=win)
        self.delay_buf = deque(maxlen=delay+1)
        self.tokens = []  # (t, token, novelty, coherence)
        self.last_token = None
        self.persist_l = 0  # how long we're "locked"
    
    def update(self, val, t):
        # delayed value (for phase portrait / coherence)
        self.delay_buf.append(val)
        delayed = self.delay_buf[0] if len(self.delay_buf) == self.delay_buf.maxlen else 0.0
        
        self.buf.append(val)
        if len(self.buf) < 3:
            return None, 0.0, 0.0, delayed
        
        arr = np.array(self.buf)
        # novelty ~ magnitude of recent change
        novelty = np.abs(arr[-1] - arr[-2])
        # coherence ~ inverse of variance in window (normalized)
        v = np.var(arr)
        coherence = 1.0 / (1e-6 + 1.0 + 10.0*v)  # squashes into (0,1) ish
        
        token = None
        # simple rules:
        if novelty > 0.7:           # big surprise -> hub/reset
            token = 'h'
            self.persist_l = 0
        elif coherence > 0.8:       # very steady -> loop/lock
            self.persist_l += 1
            if self.persist_l > 8:
                token = 'l'
        else:
            self.persist_l = max(0, self.persist_l - 1)
            token = 's'
        
        if token is not None:
            self.tokens.append((t, token, float(novelty), float(coherence)))
            self.last_token = token
        return token, float(novelty), float(coherence), delayed

# ---------------------------
# HCU
# ---------------------------
class HCU:
    def __init__(self, x, y, freq=0.12):
        self.x = x
        self.y = y
        self.freq = freq
        self.phase = 0.0
        self.noise_mix = 0.05
        self.k_field = 0.4      # coupling to field
        self.k_move  = 0.8      # how strongly we follow gradients
        self.port = MtxPort(win=60, delay=15)
        self.sig_hist = []      # for portrait
    
    def step(self, F, dt, t):
        H, W = F.shape
        # sample field & gradient
        xi = int(np.clip(self.x, 1, W-2))
        yi = int(np.clip(self.y, 1, H-2))
        local = F[yi, xi]
        gx = (F[yi, xi+1] - F[yi, xi-1]) * 0.5
        gy = (F[yi+1, xi] - F[yi-1, xi]) * 0.5
        
        # internal oscillator driven by field + noise
        dtheta = 2*np.pi*self.freq*dt + self.k_field*local*dt
        dtheta += self.noise_mix * rng.normal(0, 0.4) * np.sqrt(dt)
        self.phase = (self.phase + dtheta) % (2*np.pi)
        signal = np.sin(self.phase)
        self.sig_hist.append(signal)
        
        # emit MTX and read metrics
        token, novelty, coherence, delayed = self.port.update(signal, t)
        
        # homeostat: keep novelty in [0.25, 0.5] and coherence in [0.2, 0.6]
        # raise novelty by increasing noise; increase coherence by reducing noise and nudging freq stability
        if novelty < 0.25:
            self.noise_mix += 0.002
        elif novelty > 0.5:
            self.noise_mix -= 0.003
        
        if coherence < 0.2:
            self.freq += 0.0008  # drift to new regime
        elif coherence > 0.6:
            self.freq -= 0.0005  # avoid getting stuck
        
        self.noise_mix = float(np.clip(self.noise_mix, 0.0, 0.2))
        self.freq = float(np.clip(self.freq, 0.05, 0.25))
        
        # act on world: inject pulse on hub (h) and small ripple on loop (l)
        inj = 0.0
        if token == 'h':
            inj = 1.2
        elif token == 'l':
            inj = 0.3
        
        if inj > 0:
            F[yi-1:yi+2, xi-1:xi+2] += inj
        
        # move along gradient to "interesting" spots (gradient ascent on |grad|)
        gmagx = gx
        gmagy = gy
        self.x = (self.x + self.k_move*gmagx) % W
        self.y = (self.y + self.k_move*gmagy) % H
        
        return token, novelty, coherence, signal, delayed, (xi, yi)

# ---------------------------
# Simulation
# ---------------------------
H, W = 100, 100
F = rng.normal(0, 0.02, size=(H, W))     # initial field
F_prev = F.copy()

decay = 0.995
diff  = 0.15

hcu = HCU(x=W*0.65, y=H*0.55, freq=0.11)

steps = 700
dt = 1.0
snap_every = 2  # render every N steps

frames = []
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
ax.set_axis_off()

im = ax.imshow(F, vmin=-1.0, vmax=1.0, interpolation='bilinear', animated=True)
hcu_dot, = ax.plot([], [], marker='o', markersize=6, linestyle='None')
txt = ax.text(2, 4, "", fontsize=9, color='w', bbox=dict(facecolor='k', alpha=0.4, boxstyle='round'))
token_strip = ""

# Pre-allocate list for animation frames (PillowWriter pulls from draw state)
def update_frame(step):
    global F, F_prev, token_strip
    # wave update
    F = decay*F + diff*laplacian(F)
    # small background stirring
    if step % 25 == 0:
        F += rng.normal(0, 0.03, size=F.shape)
    
    token, nov, coh, sig, dly, (xi, yi) = hcu.step(F, dt, t=step*dt)
    
    # keep strip manageable
    if token is not None:
        token_strip = (token_strip + token)[-45:]
    
    # draw
    im.set_data(F)
    hcu_dot.set_data([xi], [yi])
    txt.set_text(
        f"t={step:4d}  MTX:{token_strip}\n"
        f"novelty={nov:0.2f}  coherence={coh:0.2f}  freq={hcu.freq:0.3f} noise={hcu.noise_mix:0.2f}"
    )
    return [im, hcu_dot, txt]

# Render animation
writer = PillowWriter(fps=20)
anim_path = "hcu_field.gif"
with writer.saving(fig, anim_path, dpi=120):
    for s in range(steps):
        artists = update_frame(s)
        if s % snap_every == 0:
            writer.grab_frame()

plt.close(fig)

# ---------------------------
# Phase portrait with token colors
# ---------------------------
sig = np.array(hcu.sig_hist)
# build delayed series identical to MtxPort
delay = hcu.port.delay
sig_delayed = np.concatenate([np.zeros(delay), sig[:-delay]])

# token times
df_tokens = pd.DataFrame(hcu.port.tokens, columns=["t", "token", "novelty", "coherence"])
csv_path = "mtx_tokens.csv"
df_tokens.to_csv(csv_path, index=False)

# map tokens to colors
color_map = {'h': 'tab:red', 'l': 'tab:green', 's': 'tab:blue'}
colors = np.array(['tab:gray']*len(sig))
if not df_tokens.empty:
    idx = np.clip(df_tokens['t'].astype(int).values, 0, len(sig)-1)
    for i, tok in zip(idx, df_tokens['token'].values):
        colors[i] = color_map.get(tok, 'tab:gray')

plt.figure(figsize=(6,6))
plt.scatter(sig_delayed, sig, s=6, c=colors, alpha=0.7)
plt.title("Phase Portrait (signal vs delayed) — MTX colored")
plt.xlabel("Delayed(signal)")
plt.ylabel("Signal")
portrait_path = "phase_portrait.png"
plt.tight_layout()
plt.savefig(portrait_path, dpi=140)
plt.close()

# Also save a short summary text of counts
counts = df_tokens['token'].value_counts().to_dict() if not df_tokens.empty else {}
summary_txt = f"MTX counts: {counts}\nTotal steps: {steps}\n"
with open("summary.txt", "w") as f:
    f.write(summary_txt)

anim_path, portrait_path, csv_path, "summary.txt"
