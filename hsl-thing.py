# hcu_life.py
# A unified "living system": wave field + homeostatic cognitive units + MTX bus
# Runs on NumPy. Optional: set USE_TORCH=True for PyTorch/CUDA acceleration.

import math, time, random, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------- Config -------------------------
GRID = 168                 # field size (NxN). 168 is a sweet spot for Ryzen 5500
DT = 0.12                  # simulation dt
STEPS_PER_FRAME = 2        # physics steps per drawn frame
C = 0.85                   # wave speed
DAMP = 0.015               # global damping
NONLIN = 0.18              # double-well-ish nonlinearity strength
NOISE_AMP = 0.0007         # background field noise (toggle with 'n')

NUM_HCU = 12               # "body" nodes; they self-organize into a soft organism
RING = True                # connect ends to make a loop
SPRING_K = 0.12            # body spring stiffness
SPRING_REST = 12.0         # body link rest length (in px)
SPACING_REPULSION = 150.0  # soft area repulsion to keep body coherent

HCU_SENSE_SIGMA = 3.0      # how wide they read/write the field
HCU_STAMP = 0.012          # how strongly they write the field when acting
HCU_MOVE_GAIN = 0.85       # how strongly they surf field gradients
HCU_NOISE = 0.35           # internal exploratory jitter
HCU_TARGET_AMP = 0.30      # internal Hopf attractor's preferred |z|
HCU_BASE_FREQ = 1.6        # nominal internal frequency (rad/s)

BUS_MAX = 60               # how many recent MTX tokens to show
GUIDANCE = False           # 'g' key toggles: scripted bursts into the MTX bus
FIELD_NOISE_ON = True      # 'n' key toggles

USE_TORCH = False          # switch to True to try PyTorch (optional)
# ----------------------------------------------------------

# ------------------------- Backend ------------------------
if USE_TORCH:
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        USE_TORCH = False
        device = None

def xp(array_like):
    if USE_TORCH:
        return torch.tensor(array_like, dtype=torch.float32, device=device)
    else:
        return np.array(array_like, dtype=np.float32)

def xzeros(shape):
    if USE_TORCH:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    else:
        return np.zeros(shape, dtype=np.float32)

def laplacian(A):
    if USE_TORCH:
        return (torch.roll(A, 1, 0) + torch.roll(A, -1, 0) +
                torch.roll(A, 1, 1) + torch.roll(A, -1, 1) - 4*A)
    else:
        return (np.roll(A, 1, 0) + np.roll(A, -1, 0) +
                np.roll(A, 1, 1) + np.roll(A, -1, 1) - 4*A)

def to_np(A):
    if USE_TORCH:
        return A.detach().cpu().numpy()
    else:
        return A

# Prebuild a small Gaussian stamp used by HCUs to read/write field
def gaussian_stamp(radius=7, sigma=HCU_SENSE_SIGMA):
    r = int(radius)
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x**2 + y**2)/(2*sigma**2))
    g /= g.sum()
    return g.astype(np.float32)

STAMP = gaussian_stamp(7, HCU_SENSE_SIGMA)

def splat(field, x, y, amp):
    """Add a Gaussian blob to the field at (x,y) with amplitude amp."""
    h, w = field.shape
    r = STAMP.shape[0]//2
    xi, yi = int(x), int(y)
    x0, x1 = max(0, xi-r), min(w, xi+r+1)
    y0, y1 = max(0, yi-r), min(h, yi+r+1)
    sx0, sx1 = r-(xi-x0), r+(x1-xi)
    sy0, sy1 = r-(yi-y0), r+(y1-yi)
    if x0 < x1 and y0 < y1:
        field[y0:y1, x0:x1] += amp * STAMP[sy0:sy1, sx0:sx1]

# ------------------------- HCU ----------------------------
class HCU:
    """Homeostatic Cognitive Unit with internal Hopf oscillator and MTX behavior."""
    def __init__(self, x, y, idx):
        self.x = float(x); self.y = float(y)
        self.vx = 0.0; self.vy = 0.0
        self.idx = idx

        # Hopf oscillator: dz/dt = (mu + i*omega - |z|^2) z + input
        self.z = complex(np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1))
        self.mu = 1.0
        self.omega = np.random.uniform(0.8, 1.2)*HCU_BASE_FREQ

        # State tracking
        self.energy = 0.0
        self.energy_smooth = 0.0
        self.last_token = None
        self.token_clock = 0.0
        self.mtx_history = []

    def hopf_step(self, u, dt):
        # u is complex input; implement Euler step for Hopf normal form
        z = self.z
        r2 = (z.real*z.real + z.imag*z.imag)
        dz = complex(self.mu - r2, self.omega) * z + u
        z = z + dz*dt
        self.z = z

    def sense(self, field):
        # Read local field and gradient using Gaussian window
        h, w = field.shape
        xi, yi = int(self.x), int(self.y)
        r = STAMP.shape[0]//2
        x0, x1 = max(0, xi-r), min(w, xi+r+1)
        y0, y1 = max(0, yi-r), min(h, yi+r+1)
        sx0, sx1 = r-(xi-x0), r+(x1-xi)
        sy0, sy1 = r-(yi-y0), r+(y1-yi)
        patch = field[y0:y1, x0:x1]
        mask = STAMP[sy0:sy1, sx0:sx1]
        val = float((patch * mask).sum())
        # Approximate gradient from central diffs
        gx = float((field[yi, (xi+1)%w] - field[yi, (xi-1)%w]) * 0.5)
        gy = float((field[(yi+1)%h, xi] - field[(yi-1)%h, xi]) * 0.5)
        return val, gx, gy

    def act(self, field, dt, bus):
        # Sense world
        val, gx, gy = self.sense(field)

        # Internal drive: try to keep |z| ~ HCU_TARGET_AMP, phase ~ free
        r = abs(self.z)
        amp_err = (HCU_TARGET_AMP - r)
        # Map sensed field into complex input aligned with gradient direction
        u = complex(val*0.8, amp_err*0.6)
        self.hopf_step(u, dt)

        # Homeostatic "energy" = amplitude error + novelty (field curvature)
        energy = abs(amp_err) + 0.3*math.sqrt(gx*gx + gy*gy)
        self.energy = energy
        self.energy_smooth = 0.92*self.energy_smooth + 0.08*energy

        # Motion: surf down the gradient to reduce curvature; plus exploratory noise
        self.vx += (-gx * HCU_MOVE_GAIN + np.random.randn()*HCU_NOISE) * dt
        self.vy += (-gy * HCU_MOVE_GAIN + np.random.randn()*HCU_NOISE) * dt

        # Mild velocity damping
        self.vx *= 0.96; self.vy *= 0.96

        self.x = (self.x + self.vx) % field.shape[1]
        self.y = (self.y + self.vy) % field.shape[0]

        # Write to world: if stable -> reinforce (focus), if unstable -> reset (novelty)
        token = None
        if self.energy_smooth < 0.12:
            splat(field, self.x, self.y, +HCU_STAMP)   # focus = add energy well
            token = 'l3'
        elif self.energy_smooth > 0.28:
            splat(field, self.x, self.y, -HCU_STAMP)   # novelty = kick/reset
            token = 'h0'
        else:
            # transition/scan
            token = 's1'

        # Simple persistence to form MTX durations
        if token == self.last_token:
            self.token_clock += dt
        else:
            if self.last_token is not None and self.token_clock > 0.12:
                self.mtx_history.append(f"{self.last_token}{{{self.token_clock:.1f}s}}")
                bus.append((self.idx, self.last_token, self.token_clock))
            self.last_token = token
            self.token_clock = 0.0

        # Bound history
        if len(self.mtx_history) > 32:
            self.mtx_history = self.mtx_history[-32:]

        return token

# ------------------------- World --------------------------
class World:
    def __init__(self):
        self.phi = xzeros((GRID, GRID))
        self.phi_prev = xzeros((GRID, GRID))
        self.field_noise_on = FIELD_NOISE_ON
        self.bus = []  # (idx, token, duration)

        # Drops: (x,y,amp,decay)
        self.sources = []  # positive = attractor, negative = repulsor

        # ---------- build organism ----------
        self.agents = []
        cx, cy = GRID // 2, GRID // 2
        if RING:
            radius = max(10.0, SPRING_REST * NUM_HCU / (2.0 * math.pi))
            for i in range(NUM_HCU):
                angle = 2.0 * math.pi * i / NUM_HCU
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                self.agents.append(HCU(x, y, i))
        else:
            for i in range(NUM_HCU):
                x = cx + (i - NUM_HCU / 2) * SPRING_REST
                y = cy
                self.agents.append(HCU(x, y, i))

        # springs (links) between consecutive agents
        self.links = []
        for i in range(NUM_HCU - 1):
            self.links.append((i, i + 1))
        if RING and NUM_HCU > 2:
            self.links.append((NUM_HCU - 1, 0))

    # ---------------- physics ----------------
    def step_field(self, dt):
        # wave + soft double-well nonlinearity + damping
        L = laplacian(self.phi)
        phi_new = (2.0 - DAMP) * self.phi - (1.0 - DAMP) * self.phi_prev + (C ** 2) * (dt ** 2) * L
        phi_new -= NONLIN * (dt ** 2) * (self.phi ** 3)

        # inject sources (attractors/repulsors) with decay
        if len(self.sources) > 0:
            for s in list(self.sources):
                x, y, amp, decay = s
                splat(phi_new, x, y, amp)
                s[2] *= (1.0 - decay)
                if abs(s[2]) < 1e-4:
                    self.sources.remove(s)

        # ambient noise
        if self.field_noise_on and NOISE_AMP > 0:
            if USE_TORCH:
                phi_new += NOISE_AMP * (2.0 * torch.rand_like(phi_new) - 1.0)
            else:
                phi_new += NOISE_AMP * (2.0 * np.random.rand(*phi_new.shape) - 1.0)

        self.phi_prev, self.phi = self.phi, phi_new

    def apply_body_forces(self, dt):
        # springs along links
        for i, j in self.links:
            a, b = self.agents[i], self.agents[j]
            dx = b.x - a.x
            dy = b.y - a.y
            # shortest path on torus
            if abs(dx) > GRID / 2: dx -= math.copysign(GRID, dx)
            if abs(dy) > GRID / 2: dy -= math.copysign(GRID, dy)

            dist = math.hypot(dx, dy) + 1e-6
            force = SPRING_K * (dist - SPRING_REST)
            fx = force * dx / dist
            fy = force * dy / dist
            a.vx += fx * dt
            a.vy += fy * dt
            b.vx -= fx * dt
            b.vy -= fy * dt

        # soft short-range repulsion to avoid collapse
        rep_k = SPRING_K * 0.25
        thresh = SPRING_REST * 0.75
        for i in range(NUM_HCU):
            ai = self.agents[i]
            for j in range(i + 1, NUM_HCU):
                aj = self.agents[j]
                dx = aj.x - ai.x
                dy = aj.y - ai.y
                if abs(dx) > GRID / 2: dx -= math.copysign(GRID, dx)
                if abs(dy) > GRID / 2: dy -= math.copysign(GRID, dy)
                dist = math.hypot(dx, dy) + 1e-6
                if dist < thresh:
                    f = rep_k * (thresh - dist)
                    fx = f * dx / dist
                    fy = f * dy / dist
                    ai.vx -= fx * dt
                    ai.vy -= fy * dt
                    aj.vx += fx * dt
                    aj.vy += fy * dt

    def step_agents(self, dt):
        tokens = []
        for a in self.agents:
            tok = a.act(self.phi, dt, self.bus)
            tokens.append(tok)
        self.apply_body_forces(dt)

        # keep MTX bus bounded
        while len(self.bus) > BUS_MAX:
            self.bus.pop(0)

        # simple hive metrics
        novelty = sum(1 for t in tokens if t == 'h0') / max(1, len(tokens))
        coherence = sum(1 for t in tokens if t == 'l3') / max(1, len(tokens))
        return novelty, coherence

    def step(self, dt):
        self.step_field(dt)
        return self.step_agents(dt)

# ---------------- visualization ----------------
def run():
    world = World()

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_axis_off()

    im = ax.imshow(to_np(world.phi), cmap='viridis', vmin=-1, vmax=1, animated=True, interpolation='bilinear')
    scat = ax.scatter([a.x for a in world.agents], [a.y for a in world.agents],
                      c='deepskyblue', s=18, edgecolors='k', linewidths=0.5)

    # lines for links
    link_lines = []
    for i, j in world.links:
        ln, = ax.plot([world.agents[i].x, world.agents[j].x],
                      [world.agents[i].y, world.agents[j].y],
                      'w-', alpha=0.25, linewidth=1.0)
        link_lines.append(ln)

    # HUD
    info = ax.text(6, 8, "", color='white', fontsize=9,
                   bbox=dict(facecolor='black', alpha=0.45, pad=4), family='monospace')
    bus_txt = ax.text(6, 22, "", color='yellow', fontsize=8,
                      bbox=dict(facecolor='black', alpha=0.35, pad=4), family='monospace')

    paused = {'v': False}

    def on_click(event):
        if event.inaxes != ax: return
        x, y = event.xdata, event.ydata
        if event.button == 1:
            # attractor
            world.sources.append([x, y, +0.6, 0.012])
        elif event.button == 3:
            # repulsor
            world.sources.append([x, y, -0.6, 0.012])

    def on_key(event):
        nonlocal paused
        if event.key == ' ':
            paused['v'] = not paused['v']
        elif event.key == 'n':
            world.field_noise_on = not world.field_noise_on
        elif event.key == 'g':
            global GUIDANCE
            GUIDANCE = not GUIDANCE
        elif event.key == 'c':
            world.bus.clear()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Optional top-down “guidance” that injects periodic MTX-like pulses
    guide_clock = [0.0]

    def animate(_):
        if paused['v']:
            return im,

        # physics substeps
        for _ in range(STEPS_PER_FRAME):
            novelty, coherence = world.step(DT)

            # global guidance (acts like a cortex nudging the colony)
            if GUIDANCE:
                guide_clock[0] += DT
                if guide_clock[0] > 3.5:
                    guide_clock[0] = 0.0
                    # pick a random quadrant and drop a strong attractor & a synthetic MTX burst
                    qx = np.random.uniform(GRID * 0.2, GRID * 0.8)
                    qy = np.random.uniform(GRID * 0.2, GRID * 0.8)
                    world.sources.append([qx, qy, +1.2, 0.02])
                    # append "scripted" bus hint
                    world.bus.append((-1, 'h0', 0.4))
                    world.bus.append((-1, 'l3', 1.8))
                    if len(world.bus) > BUS_MAX:
                        world.bus = world.bus[-BUS_MAX:]

        # draw field
        im.set_data(to_np(world.phi))

        # update scatter
        xs = [a.x for a in world.agents]
        ys = [a.y for a in world.agents]
        scat.set_offsets(np.c_[xs, ys])

        # update link lines (wrap-aware)
        for ln, (i, j) in zip(link_lines, world.links):
            ax_i, ay_i = world.agents[i].x, world.agents[i].y
            ax_j, ay_j = world.agents[j].x, world.agents[j].y
            # draw shortest segment on torus
            dx = ax_j - ax_i
            dy = ay_j - ay_i
            if abs(dx) > GRID / 2: ax_j -= math.copysign(GRID, dx)
            if abs(dy) > GRID / 2: ay_j -= math.copysign(GRID, dy)
            ln.set_data([ax_i, ax_j], [ay_i, ay_j])

        # HUD text
        nov = sum(1 for a in world.agents if (a.last_token or '') == 'h0') / max(1, len(world.agents))
        coh = sum(1 for a in world.agents if (a.last_token or '') == 'l3') / max(1, len(world.agents))
        info.set_text(
            f"n={len(world.agents)}  novelty={nov:.2f}  coherence={coh:.2f}  "
            f"noise={'on' if world.field_noise_on else 'off'}  guidance={'on' if GUIDANCE else 'off'}\n"
            f"controls: L-click attractor  R-click repulsor  [n]oise  [g]uidance  [space] pause"
        )

        # MTX bus view
        tail = world.bus[-BUS_MAX:]
        def fmt(t):
            idx, tok, dur = t
            who = 'Σ' if idx == -1 else str(idx)
            return f"{tok}{{{dur:.1f}s}}@{who}"
        bus_txt.set_text("MTX: " + " ".join(fmt(t) for t in tail[-18:]))

        return im, scat, *link_lines, info, bus_txt

    ax.set_xlim(0, GRID)
    ax.set_ylim(0, GRID)
    ax.set_aspect('equal')

    ani = FuncAnimation(fig, animate, interval=30, blit=False)
    plt.show()


if __name__ == "__main__":
    run()
