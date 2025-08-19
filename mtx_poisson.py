# mtx_physics_poisson.py
# MTX-as-Physics: a tiny MTX controller solves ∇²u = f via SOR + annealing.
# deps: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.set_printoptions(suppress=True)

# ---------------- grid & rhs ----------------
N = 128                 # grid size
h2 = 1.0                # grid spacing^2 (take h=1)
f = np.zeros((N, N), dtype=np.float32)

def add_gaussian(cx, cy, amp=1.0, sigma=6.0):
    y, x = np.mgrid[0:N, 0:N]
    f[:] += amp * np.exp(-((x-cx)**2 + (y-cy)**2)/(2*sigma**2))

# synthetic RHS (two charges)
add_gaussian(N*0.35, N*0.55, +1.2, 5.0)
add_gaussian(N*0.70, N*0.35, -1.0, 7.0)

# ---------------- solver state ----------------
u = np.zeros_like(f)    # Dirichlet: boundaries fixed at 0
omega = 1.6             # SOR relaxation, MTX will adapt this
T = 0.0                 # annealing temperature (MTX raises/lowers)
rng = np.random.default_rng(42)

def laplacian(U):
    return (np.roll(U, +1, 0) + np.roll(U, -1, 0) +
            np.roll(U, +1, 1) + np.roll(U, -1, 1) - 4.0*U)

def residual(U):
    r = laplacian(U) - f*h2
    # zero-out boundary residuals (Dirichlet)
    r[0,:]=r[-1,:]=r[:,0]=r[:,-1]=0.0
    return r

def sor_step(U, omega):
    # in-place red-black SOR for speed & simplicity
    for parity in (0, 1):
        # update interior points with checkerboard parity
        U1 = U[1:-1, 1:-1]
        north = U[0:-2, 1:-1]
        south = U[2:  , 1:-1]
        west  = U[1:-1, 0:-2]
        east  = U[1:-1, 2:  ]
        rhs   = f[1:-1, 1:-1]*h2
        new = 0.25*(north+south+west+east - rhs)
        mask = ((np.add.outer(np.arange(1,N-1), np.arange(1,N-1)) & 1) == parity)
        U1[mask] = (1-omega)*U1[mask] + omega*new[mask]

# ---------------- MTX controller ----------------
mtx_log = []
tok_curr, tok_clock = None, 0.0

hist = []     # rolling residual history
H = 40

def emit(tok):
    global tok_curr, tok_clock
    nonlocal_vars = {}
    if tok == tok_curr:
        return
    if tok_curr is not None and tok_clock > 0.05:
        mtx_log.append(f"{tok_curr}{{{tok_clock:.1f}s}}")
        if len(mtx_log) > 60:
            del mtx_log[:len(mtx_log)-60]
    tok_curr = tok
    tok_clock = 0.0

def controller(res, dres):
    """Map residual level/slope -> MTX + parameter changes."""
    global omega, T, tok_clock
    tok_clock += 0.02

    improving_fast = dres < -0.02*res
    plateau = abs(dres) < 1e-6
    stuck   = (len(hist) > 10) and (res > np.min(hist[-10:]) * 0.999) and plateau

    if stuck:
        # h0: reset/novelty -> kick temperature & retune omega
        emit('h0')
        T = min(0.06, T*0.5 + 0.04)
        omega = np.clip(omega + rng.uniform(-0.25, +0.25), 1.2, 1.95)
    elif improving_fast and res > 1e-5:
        # l3: focus/commit -> cool down & nudge omega toward optimal ~1.9
        emit('l3')
        T = max(0.0, T*0.8 - 0.005)
        omega = np.clip(0.95*omega + 0.05*1.9, 1.2, 1.95)
    else:
        # s1: scan/explore
        emit('s1')
        T = max(0.0, T*0.98 - 0.001)

# ---------------- UI ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), gridspec_kw={'width_ratios':[1.2,1]})
im = ax1.imshow(u, cmap='coolwarm', vmin=-0.6, vmax=0.6, interpolation='bilinear')
ax1.set_title("Potential u solving ∇²u = f")
ax1.set_axis_off()
(line,) = ax2.plot([], [], lw=2)
ax2.set_xlim(0, 800)
ax2.set_ylim(1e-7, 1e-1)
ax2.set_yscale('log')
ax2.grid(alpha=0.3, which='both')
ax2.set_title("Residual ||∇²u - f||₂")
hud = ax1.text(6, 10, "", color='white',
               bbox=dict(facecolor='black', alpha=0.5, pad=4), family='monospace', fontsize=9)
bus = ax1.text(6, 26, "", color='yellow',
               bbox=dict(facecolor='black', alpha=0.45, pad=4), family='monospace', fontsize=8)

iters = [0]
res_curve = []

def step():
    # SOR sweeps + tiny Gaussian noise from temperature
    for _ in range(6):
        sor_step(u, omega)
        if T > 0:
            noise = T * rng.standard_normal(size=u.shape).astype(np.float32)
            noise[0,:]=noise[-1,:]=noise[:,0]=noise[:,-1]=0.0
            u[:] += noise

    r = residual(u)
    res = float(np.sqrt(np.mean(r*r)))
    hist.append(res)
    if len(hist) > H: del hist[:len(hist)-H]
    dres = 0.0 if len(hist) < 3 else (hist[-1] - hist[-3])

    controller(res, dres)

    iters[0] += 1
    res_curve.append(res)
    return res

def animate(_):
    res = step()
    im.set_data(u)
    x = np.arange(len(res_curve))
    line.set_data(x, res_curve)

    hud.set_text(f"iter={iters[0]}  ω={omega:.2f}  T={T:.3f}  res={res:8.2e}")
    bus.set_text("MTX: " + " ".join(mtx_log[-26:]))
    ax2.set_xlim(0, max(120, len(res_curve)))

    # stop criterion (auto-close)
    if res < 3e-5 or iters[0] > 3000:
        plt.pause(0.1)
        ani.event_source.stop()
    return im, line, hud, bus

ani = FuncAnimation(fig, animate, interval=25, blit=False)
plt.tight_layout()
plt.show()
