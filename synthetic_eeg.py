import numpy as np
from scipy.io.wavfile import write as wav_write

# ----------------------------
# Config
# ----------------------------
FS = 250.0                 # Hz
EPOCH = 0.5                # seconds per token epoch
BANDS = {"delta":2, "theta":6, "alpha":10, "beta":20, "gamma":40}  # Hz centers
RNG = np.random.default_rng(42)

# Token type -> band weights (relative amplitudes)
WEIGHTS_BY_TYPE = {
    "loop":  np.array([0.3, 0.5, 1.0, 0.5, 0.2]),   # α-dominant, stable
    "hub":   np.array([0.2, 0.8, 0.4, 0.4, 0.9]),   # θ–γ boosted (gating)
    "state": np.array([0.2, 0.4, 0.5, 0.8, 0.6]),   # β/γ a bit higher
}

# 8 toy channels with simple spatial flavouring (Occipital ~ alpha, Frontal ~ beta/gamma)
CHANNEL_SHAPES = np.array([
    [0.4,0.6,1.2,0.6,0.3],  # O1
    [0.4,0.6,1.1,0.6,0.3],  # O2
    [0.3,0.6,1.0,0.7,0.3],  # Oz
    [0.3,0.5,0.9,0.8,0.4],  # POz
    [0.2,0.4,0.7,1.0,0.6],  # Pz
    [0.2,0.3,0.6,1.1,0.7],  # Fz
    [0.2,0.3,0.5,1.1,0.8],  # F3
    [0.2,0.3,0.5,1.1,0.8],  # F4
])  # shape [n_ch, n_bands]


# ----------------------------
# 1) Synthetic Language (SLX) generator
# ----------------------------
def build_symbols(n_loops=2, n_hubs=2, n_states=6):
    syms = []
    syms += [f"l{i}" for i in range(n_loops)]
    syms += [f"h{i}" for i in range(n_hubs)]
    syms += [f"s{i}" for i in range(n_states)]
    return syms

def sample_hsmm_sequence(T=120, n_loops=2, n_hubs=2, n_states=6, n_modules=2,
                         p_bridge=0.05, loop_mean_epochs=4, state_mean_epochs=1,
                         hub_mean_epochs=1):
    """
    Hidden semi-Markov-ish:
    - States partitioned into modules; transitions prefer within-module.
    - Loops have self-dwell > others.
    - Hubs connect modules; short dwell.
    Returns list of (symbol, duration_sec).
    """
    # Partition into modules
    loops  = [f"l{i}" for i in range(n_loops)]
    hubs   = [f"h{i}" for i in range(n_hubs)]
    states = [f"s{i}" for i in range(n_states)]

    modules = []
    split = np.array_split(np.arange(n_states), n_modules)
    for m, idxs in enumerate(split):
        modules.append({"loops":[loops[m % n_loops]],
                        "hubs":[hubs[m % n_hubs]],
                        "states":[f"s{i}" for i in idxs.tolist()]})

    # Simple dwell samplers (in epochs)
    def dwell_loop():  return max(1, int(np.round(RNG.exponential(loop_mean_epochs))))
    def dwell_state(): return max(1, int(np.round(RNG.exponential(state_mean_epochs))))
    def dwell_hub():   return max(1, int(np.round(RNG.exponential(hub_mean_epochs))))

    # Start in module 0, at its loop
    m = 0
    seq = []
    time_epochs = 0
    cur = modules[m]["loops"][0]

    while time_epochs < T:
        if cur.startswith("l"):
            d = dwell_loop()
            seq.append((cur, d*EPOCH))
            time_epochs += d
            # 70% stay in-module state, 25% return to loop, 5% bridge via hub
            r = RNG.random()
            if r < 0.05 and n_modules > 1:
                # bridge
                seq.append((modules[m]["hubs"][0], dwell_hub()*EPOCH))
                time_epochs += dwell_hub()
                # switch module
                m = (m + 1) % n_modules
                cur = modules[m]["states"][RNG.integers(len(modules[m]["states"]))]
            elif r < 0.30:
                cur = modules[m]["states"][RNG.integers(len(modules[m]["states"]))]
            else:
                cur = modules[m]["loops"][0]

        elif cur.startswith("s"):
            d = dwell_state()
            seq.append((cur, d*EPOCH))
            time_epochs += d
            r = RNG.random()
            if r < 0.15:
                cur = modules[m]["loops"][0]
            elif r < 0.15 + p_bridge and n_modules > 1:
                seq.append((modules[m]["hubs"][0], dwell_hub()*EPOCH))
                time_epochs += dwell_hub()
                m = (m + 1) % n_modules
                cur = modules[m]["states"][RNG.integers(len(modules[m]["states"]))]
            else:
                cur = modules[m]["states"][RNG.integers(len(modules[m]["states"]))]

        else:  # hub
            d = dwell_hub()
            seq.append((cur, d*EPOCH))
            time_epochs += d
            # after hub, go to loop of new module
            cur = modules[m]["loops"][0]

    return merge_runs(seq)

def merge_runs(seq):
    """Merge consecutive identical symbols, keep durations in seconds."""
    out = []
    for sym, dur in seq:
        if out and out[-1][0] == sym:
            out[-1] = (sym, out[-1][1] + dur)
        else:
            out.append((sym, float(dur)))
    return out

def to_mtx(tokens):
    parts = ["∂"]
    for sym, dur in tokens:
        parts.append(f"{sym}{{{dur:.2f}s}}")
    parts.append("∂")
    return " ".join(parts)

# ----------------------------
# 2) Minimal synthetic signal renderer
# ----------------------------
def render_signal(tokens, fs=FS, channel_shapes=CHANNEL_SHAPES, smooth_alpha=0.2):
    """
    Phase-continuous, band-summed signal with smoothed amplitude targets.
    Returns:
      sig_ch: [n_channels, n_samples]
      t:      [n_samples]
    """
    bands = list(BANDS.values())           # [f_delta, f_theta, ...]
    n_b = len(bands)
    n_ch = channel_shapes.shape[0]

    # Precompute total samples
    total_s = sum(d for _, d in tokens)
    n_samples = int(np.round(total_s * fs))
    t = np.arange(n_samples) / fs
    sig = np.zeros((n_ch, n_samples), dtype=np.float32)

    # Continuous phase per band
    phase = np.zeros(n_b, dtype=np.float64)

    # Current amplitude per channel x band
    A = np.zeros((n_ch, n_b), dtype=np.float64)

    # Token envelopes
    cursor = 0
    for sym, dur in tokens:
        steps = int(np.round(dur * fs))
        if steps <= 0: 
            continue

        # Target amplitude by token type
        typ = "loop" if sym.startswith("l") else ("hub" if sym.startswith("h") else "state")
        base = WEIGHTS_BY_TYPE[typ]  # shape [n_b]

        # Channel-specific scaling (simple topography)
        target = channel_shapes @ base  # [n_ch]

        # Expand per band using outer product: channel x band
        target_cb = channel_shapes * base  # [n_ch, n_b]

        # Smoothly approach target
        for i in range(steps):
            A = (1.0 - smooth_alpha) * A + smooth_alpha * target_cb  # EMA per sample

            # Oscillators sum per band with continuous phase
            frame = np.zeros(n_ch, dtype=np.float64)
            for b_idx, f0 in enumerate(bands):
                phase[b_idx] += 2.0 * np.pi * f0 / fs
                frame += A[:, b_idx] * np.sin(phase[b_idx])

            sig[:, cursor] = frame.astype(np.float32)
            cursor += 1
            if cursor >= n_samples:
                break

    # Normalize to safe audio range for listening (also keeps values bounded)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig /= (peak * 1.1)

    return sig, t

def save_wav_mono(path, sig, fs=FS):
    mono = np.mean(sig, axis=0)  # simple mixdown
    wav_write(path, int(fs), (mono * 32767).astype(np.int16))

# ----------------------------
# Demo / CLI-ish
# ----------------------------
if __name__ == "__main__":
    # 1) Make a synthetic sequence (~60s)
    tokens = sample_hsmm_sequence(
        T=120,            # epochs of 0.5s -> ~60 s
        n_loops=2,
        n_hubs=2,
        n_states=8,
        n_modules=2,
        p_bridge=0.05,    # rare bridges -> higher modularity
        loop_mean_epochs=4,
        state_mean_epochs=1,
        hub_mean_epochs=1
    )

    mtx = to_mtx(tokens)
    print("\nMind Text (SLX/MTX):\n", mtx, "\n")

    # 2) Render simple synthetic signal (8 channels)
    sig, t = render_signal(tokens, fs=FS)

    # 3) Save outputs
    np.savez("synthetic_slx_signal.npz", sig=sig, fs=FS, t=t, tokens=tokens, mtx=mtx)
    save_wav_mono("synthetic_slx.wav", sig, fs=FS)
    print("Saved: synthetic_slx.wav (mono preview) and synthetic_slx_signal.npz (8-ch signal).")
