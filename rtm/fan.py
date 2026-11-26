"""FE/CV Flow-Analysis-Network (FAN) front tracking (Okabe Sec. 2.1, Eqs. 6-8).

Per node i (P1 dof, median-dual control volume):
    pore volume   Vp_i = phi * |CV_i|          (per unit thickness, m^2)  <- A7 porosity fix
    net inflow    Q_i  = -(A p)_i              (m^2/s, conservative CV-face flux)
    fill update   f_i <- f_i + Q_i dt / Vp_i
    time step     dt = min_{0<=f_i<1, Q_i>0} (1 - f_i) Vp_i / Q_i   -> exactly one CV fills/step
After each step the pressure is re-solved on the filled region only:
    unknowns  : filled nodes (f = 1) that are not gate/vent nodes
    Dirichlet : gate nodes p = p_inlet; vent nodes and every node with f < 1 (front + empty,
                assumption A5) p = p_vent.
Time convention: t = 0 when resin enters at the gate; seconds.
"""
import time
import numpy as np
import scipy.sparse.linalg as spla


def run_fan(A, area, gate, vent, cfg, snapshot_fracs=(), log=print):
    n = A.shape[0]
    Vp = cfg.phi * area                       # pore volume per unit thickness [m^2]
    dp = cfg.dp                               # [Pa]; we solve for p' = p - p_vent
    f = np.zeros(n)
    t_fill = np.full(n, np.inf)
    filled = np.zeros(n, bool)
    f[gate] = 1.0; filled[gate] = True; t_fill[gate] = 0.0
    is_gate = np.zeros(n, bool); is_gate[gate] = True
    is_vent = np.zeros(n, bool); is_vent[vent] = True
    total_area = area.sum()

    A = A.tocsr()
    t, step = 0.0, 0
    snaps, snap_iter = [], iter(sorted(snapshot_fracs))
    next_snap = next(snap_iter, None)
    q_gate_hist = []
    pprime = np.zeros(n)
    tic = time.time()
    while not filled.all():
        # ---- pressure on the filled region (p' = p - p_vent; p'_gate = dp, p'_front = 0) ----
        unk = np.flatnonzero(filled & ~is_gate & ~is_vent)
        pprime[:] = 0.0
        pprime[is_gate] = dp
        if unk.size:
            rhs = -A[unk][:, is_gate] @ np.full(is_gate.sum(), dp)
            pprime[unk] = spla.spsolve(A[unk][:, unk].tocsc(), rhs)
        Q = -(A @ pprime)                     # net inflow per CV [m^2/s]
        if step == 0:
            p_first = pprime + cfg.p_vent     # earliest pressure field (Re check)

        front = (~filled) & (Q > 1e-30 * dp)
        if not front.any():
            log(f"[FAN] stalled at t={t:.1f}s with {(~filled).sum()} empty nodes (no inflow)")
            break
        dt_i = (1.0 - f[front]) * Vp[front] / Q[front]
        dt = dt_i.min()
        idx = np.flatnonzero(front)
        f[idx] += Q[idx] * dt / Vp[idx]
        t += dt
        newly = idx[f[idx] >= 1.0 - 1e-10]    # >=1 node (ties fill together)
        f[newly] = 1.0; filled[newly] = True; t_fill[newly] = t
        q_gate_hist.append((t, -Q[is_gate].sum()))   # inflow at gate = -outflow sum
        step += 1

        frac = area[filled].sum() / total_area
        while next_snap is not None and frac >= next_snap:
            snaps.append(dict(t=t, frac=frac, p=pprime + cfg.p_vent, f=f.copy()))
            next_snap = next(snap_iter, None)
        if step % 2000 == 0:
            log(f"[FAN] step {step:6d}  t = {t:9.1f} s  filled area = {100*frac:5.1f} %  "
                f"({time.time()-tic:.0f} s wall)")
    log(f"[FAN] done: {step} steps, fill time {t:.1f} s, wall {time.time()-tic:.0f} s")
    return dict(t_fill=t_fill, t_total=t, steps=step, snapshots=snaps, p_first=p_first,
                gate_flow=np.array(q_gate_hist))
