"""
bergman_impulsive.py
====================
Modèle de Bergman à quatre compartiments (G, X, I, m) avec injections impulsionnelles
d'insuline, coût, gradient adjoint exact, orbite périodique et multiplicateurs de Floquet.

Unités : G [mg/dL], X [1/min], I [mU/L], m [mg/dL/min], temps [min].
Équations (entre deux injections) :
    G' = -p1 (G - Gb) - X G + m
    X' = -p2 X + p3 (I - Ib)
    I' = -n (I - Ib)
    m' = -m/tau + h(t),    h = somme de repas  A_q/tau^2 exp(-(t-theta_q)/tau) 1_[theta_q, theta_q+Tm)
Saut à l'instant t_j :  I(t_j+) = I(t_j-) + kappa_j.
Coût :  J = int_0^T [ wh (G-Gu)_+^2 + wl (Gl-G)_+^2 + wI I^2 ] dt + (rho/2) sum kappa_j^2.
"""
import warnings
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, Bounds, LinearConstraint

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------
# 1. Paramètres (Tableau 1 de l'article)
# ----------------------------------------------------------------------------------------
PHYS = dict(p1=0.03082, p2=0.02093, p3=1.593e-4, n=0.05, Gb=92.0, Ib=7.3,
            tau=20.0, Tm=180.0, kG=6.0)
WEIGHTS = dict(wh=2.0, wl=8.0, wI=0.002, rho=0.02, Gu=180.0, Gl=70.0)
KMAX, KTOT = 30.0, 90.0          # dose maximale par injection, dose totale journalière
T_DAY = 1440.0


@dataclass(frozen=True)
class Scenario:
    """Un patient et un jour type : paramètres, repas (grammes de glucides) et injections."""
    phys: dict = field(default_factory=lambda: dict(PHYS))
    carbs: tuple = (60.0, 70.0, 55.0)        # g
    meal_t: tuple = (420.0, 720.0, 1140.0)   # début des repas [min]
    inj_t: tuple = (410.0, 710.0, 1130.0, 1380.0)
    w: dict = field(default_factory=lambda: dict(WEIGHTS))
    period: float = 1440.0                    # longueur d'un « jour » [min]

    @property
    def amp(self):                            # A_q = kG * carbs  [mg/dL]
        return tuple(self.phys["kG"] * c for c in self.carbs)

    def with_phys(self, **kw):
        return replace(self, phys=dict(self.phys, **kw))


NOMINAL = Scenario()


# ----------------------------------------------------------------------------------------
# 2. Intégration par segments (les discontinuités de h et les sauts sont des points de coupure)
# ----------------------------------------------------------------------------------------
def _breakpoints(sc, days):
    Tm = sc.phys["Tm"]
    pts = {0.0, days * sc.period}
    for d in range(days):
        for t in sc.inj_t:
            pts.add(t + d * sc.period)
        for t in sc.meal_t:
            pts.add(t + d * sc.period); pts.add(min(t + d * sc.period + Tm, days * sc.period))
    return sorted(p for p in pts if 0.0 <= p <= days * sc.period)


def _active_meals(sc, a, b, days):
    """Repas actifs sur le segment [a,b] (test au milieu : pas de problème de bord)."""
    mid, Tm = 0.5 * (a + b), sc.phys["Tm"]
    out = []
    for d in range(days):
        for A, t0 in zip(sc.amp, sc.meal_t):
            if A > 0 and t0 + d * sc.period <= mid < t0 + d * sc.period + Tm:
                out.append((A, t0 + d * sc.period))
    return out


def _rhs_factory(sc, meals):
    P, W = sc.phys, sc.w
    p1, p2, p3, n, Gb, Ib, tau = (P[k] for k in ("p1", "p2", "p3", "n", "Gb", "Ib", "tau"))
    wh, wl, wI, Gu, Gl = (W[k] for k in ("wh", "wl", "wI", "Gu", "Gl"))

    def f(t, y):
        G, X, I, m = y[0], y[1], y[2], y[3]
        h = 0.0
        for A, t0 in meals:
            h += A / tau ** 2 * np.exp(-(t - t0) / tau)
        l = wh * max(G - Gu, 0.0) ** 2 + wl * max(Gl - G, 0.0) ** 2 + wI * I * I
        return [-p1 * (G - Gb) - X * G + m, -p2 * X + p3 * (I - Ib), -n * (I - Ib),
                -m / tau + h, l]
    return f


def integrate(kap, sc=NOMINAL, days=1, y0=None, dt=None, rtol=1e-10, atol=1e-11, keep=False):
    """Intègre le système impulsionnel.
    Retourne l'état final augmenté (G,X,I,m,coût) ; si dt est donné, la trajectoire sur une grille ;
    si keep=True, la liste des segments avec sortie dense (pour l'adjoint)."""
    P = sc.phys
    y = np.array([P["Gb"], 0.0, P["Ib"], 0.0, 0.0] if y0 is None else list(y0)[:4] + [0.0], float)
    kd = {}
    for d in range(days):
        for t, k in zip(sc.inj_t, kap):
            kd[t + d * sc.period] = k
    bps = _breakpoints(sc, days)
    ts, Ys, segs = [], [], []
    for a, b in zip(bps[:-1], bps[1:]):
        if a in kd:
            y = y.copy(); y[2] += kd[a]
        f = _rhs_factory(sc, _active_meals(sc, a, b, days))
        if dt is not None:
            grid = np.arange(np.ceil(a / dt - 1e-9) * dt, b + 1e-9, dt)
            grid = np.clip(grid[(grid >= a - 1e-9) & (grid <= b + 1e-9)], a, b)
            sol = solve_ivp(f, (a, b), y, method="DOP853", rtol=rtol, atol=atol, t_eval=grid)
            ts.append(sol.t); Ys.append(sol.y.T)
        else:
            sol = solve_ivp(f, (a, b), y, method="DOP853", rtol=rtol, atol=atol, dense_output=keep)
        if keep:
            segs.append((a, b, sol))
        y = sol.y[:, -1]
    if dt is not None:
        t_all = np.concatenate(ts); Y_all = np.vstack(Ys)
        _, idx = np.unique(np.round(t_all, 9), return_index=True)   # supprime les doublons aux jonctions
        return y, t_all[idx], Y_all[idx]
    return (y, segs) if keep else y


def trajectory(kap, sc=NOMINAL, days=1, y0=None, dt=0.1):
    _, ts, Y = integrate(kap, sc, days, y0, dt=dt)
    return ts, Y[:, :4]


# ----------------------------------------------------------------------------------------
# 3. Coût et gradient
# ----------------------------------------------------------------------------------------
def cost(kap, sc=NOMINAL, y0=None):
    y = integrate(kap, sc, y0=y0)
    return y[4] + 0.5 * sc.w["rho"] * float(np.sum(np.square(kap)))


def dl_dG(G, w):
    return 2 * w["wh"] * max(G - w["Gu"], 0.0) - 2 * w["wl"] * max(w["Gl"] - G, 0.0)


def gradient_adjoint(kap, sc=NOMINAL, y0=None, lam_T=None, rtol=1e-10, atol=1e-11):
    """dJ/dkappa_j = rho kappa_j + lambda_3(t_j),  lambda' = -grad_Y H,  lambda(T) = lam_T (0 par défaut).
    Le pas maximal de 0,1 min sert à ne pas manquer les courts intervalles où G > Gu."""
    P, W = sc.phys, sc.w
    p1, p2, p3, n = P["p1"], P["p2"], P["p3"], P["n"]; tau = P["tau"]
    _, segs = integrate(kap, sc, y0=y0, keep=True, rtol=rtol, atol=atol)
    lam = np.zeros(4) if lam_T is None else np.array(lam_T, float)
    pos = {t: i for i, t in enumerate(sc.inj_t)}
    g = np.zeros(len(kap))
    for a, b, sol in reversed(segs):
        def fa(t, l, sol=sol):
            G, X, I, m, _ = sol.sol(t)
            return [-dl_dG(G, W) + l[0] * (p1 + X), l[0] * G + l[1] * p2,
                    -2 * W["wI"] * I - l[1] * p3 + l[2] * n, -l[0] + l[3] / tau]
        lam = solve_ivp(fa, (b, a), lam, method="DOP853", rtol=rtol, atol=atol, max_step=0.1).y[:, -1]
        if a in pos:
            g[pos[a]] = W["rho"] * kap[pos[a]] + lam[2]
    return g, lam


def gradient_forward(kap, sc=NOMINAL, y0=None, rtol=1e-10, atol=1e-11):
    """Même gradient par sensibilités directes (un seul passage avant) : plus rapide, sert pour
    l'optimisation. Chaque dérivée dY/dkappa_j obéit à l'équation variationnelle avec saut e_3 en t_j."""
    P, W = sc.phys, sc.w
    p1, p2, p3, n = P["p1"], P["p2"], P["p3"], P["n"]; tau = P["tau"]
    Jn = len(kap)
    Pn = sc.phys
    y = np.array([Pn["Gb"], 0, Pn["Ib"], 0, 0] if y0 is None else list(y0)[:4] + [0], float)
    S = np.zeros((Jn, 4)); gint = np.zeros(Jn)            # sensibilités et intégrales de dl/dY . s
    kd = {t: (i, k) for i, (t, k) in enumerate(zip(sc.inj_t, kap))}
    bps = _breakpoints(sc, 1)
    for a, b in zip(bps[:-1], bps[1:]):
        if a in kd:
            i, k = kd[a]; y = y.copy(); y[2] += k; S[i] = [0, 0, 1.0, 0]
        meals = _active_meals(sc, a, b, 1); f0 = _rhs_factory(sc, meals)

        def f(t, z):
            Y = z[:5]; dY = f0(t, Y[:5])
            G, X = Y[0], Y[1]; dG = dl_dG(G, W); dI = 2 * W["wI"] * Y[2]
            out = list(dY)
            for j in range(Jn):
                s = z[5 + 4 * j: 9 + 4 * j]
                ds = [-(p1 + X) * s[0] - G * s[1] + s[3], -p2 * s[1] + p3 * s[2], -n * s[2], -s[3] / tau]
                out += ds
            for j in range(Jn):
                s = z[5 + 4 * j: 9 + 4 * j]
                out.append(dG * s[0] + dI * s[2])
            return out
        z0 = np.concatenate([y, S.ravel(), gint])
        z = solve_ivp(f, (a, b), z0, method="DOP853", rtol=rtol, atol=atol).y[:, -1]
        y = z[:5]; S = z[5:5 + 4 * Jn].reshape(Jn, 4); gint = z[5 + 4 * Jn:]
    return sc.w["rho"] * np.asarray(kap) + gint


def kkt_residual(kap, grad, kmax=KMAX):
    """Résidu du gradient projeté sur la boîte [0,kmax] (la contrainte de somme est inactive)."""
    return np.abs(np.asarray(kap) - np.clip(np.asarray(kap) - grad, 0.0, kmax)).max()


def optimise(x0, sc=NOMINAL, method="L-BFGS-B", maxiter=100):
    x0 = np.asarray(x0, float)
    fun = lambda k: cost(k, sc)
    jac = lambda k: gradient_forward(k, sc)
    if method == "L-BFGS-B":
        r = minimize(fun, x0, jac=jac, method="L-BFGS-B", bounds=Bounds(0, KMAX),
                     options=dict(ftol=1e-15, gtol=1e-9, maxiter=maxiter))
    else:  # SLSQP avec la contrainte de somme explicite
        r = minimize(fun, x0, jac=jac, method="SLSQP", bounds=Bounds(0, KMAX),
                     constraints=[LinearConstraint(np.ones((1, len(x0))), -np.inf, KTOT)],
                     options=dict(ftol=1e-13, maxiter=maxiter))
    return r


# ----------------------------------------------------------------------------------------
# 4. Régime périodique
# ----------------------------------------------------------------------------------------
def periodic_orbit(kap, sc=NOMINAL, n_iter=4):
    """Point fixe de l'application de Poincaré (itération : la contraction est de l'ordre de 1e-13/jour)."""
    P = sc.phys
    y0 = np.array([P["Gb"], 0.0, P["Ib"], 0.0])
    for _ in range(n_iter):
        y0 = integrate(kap, sc, y0=y0)[:4]
    return y0


def cost_periodic(kap, sc=NOMINAL):
    return cost(kap, sc, y0=periodic_orbit(kap, sc))


def gradient_periodic(kap, sc=NOMINAL, n_sweeps=3):
    """Gradient exact du coût périodique : adjoint périodique lambda(T) = lambda(0),
    obtenu par balayages successifs (Psi^T est numériquement nul sur une journée)."""
    y0 = periodic_orbit(kap, sc); lam_T = np.zeros(4)
    for _ in range(n_sweeps):
        g, lam0 = gradient_adjoint(kap, sc, y0=y0, lam_T=lam_T)
        lam_T = lam0
    return g


def monodromy(kap, sc, Tp, y0, rtol=1e-11, atol=1e-13):
    """Matrice de monodromie Psi = Phi(T,0) par intégration de l'équation variationnelle le long de l'orbite
    (utilisé pour un schéma réduit : une injection et un repas par période)."""
    P = sc.phys
    p1, p2, p3, n, tau = P["p1"], P["p2"], P["p3"], P["n"], P["tau"]
    kd = dict(zip(sc.inj_t, kap)); bps = sorted({0.0, Tp, *sc.inj_t, *sc.meal_t, *(t + P["Tm"] for t in sc.meal_t if t + P["Tm"] < Tp)})
    y = np.array(list(y0) + [0.0]); Phi = np.eye(4); intX = 0.0
    for a, b in zip(bps[:-1], bps[1:]):
        if a in kd:
            y = y.copy(); y[2] += kd[a]
        meals = _active_meals(sc, a, b, 1); f0 = _rhs_factory(sc, meals)

        def f(t, z):
            Y = z[:5]; dY = f0(t, Y); G, X = Y[0], Y[1]
            B = np.array([[-(p1 + X), -G, 0, 1], [0, -p2, p3, 0], [0, 0, -n, 0], [0, 0, 0, -1 / tau]])
            return np.concatenate([dY, (B @ z[5:21].reshape(4, 4)).ravel(), [X]])
        z = solve_ivp(f, (a, b), np.concatenate([y, Phi.ravel(), [intX]]), method="DOP853", rtol=rtol, atol=atol).y[:, -1]
        y, Phi, intX = z[:5], z[5:21].reshape(4, 4), z[21]
    return Phi, intX


# ----------------------------------------------------------------------------------------
# 5. Indices cliniques
# ----------------------------------------------------------------------------------------
def indices(G):
    return dict(TIR=100 * np.mean((G >= 70) & (G <= 180)), TAR=100 * np.mean(G > 180),
                TBR=100 * np.mean(G < 70), peak=G.max(), nadir=G.min())


def day_indices(kap, sc=NOMINAL, dt=0.1):
    ts, Y = trajectory(kap, sc, dt=dt)
    return indices(Y[:, 0])


# ----------------------------------------------------------------------------------------
# 6. Population virtuelle (Runge-Kutta 4 vectorisé, pas fixe 0,25 min)
# ----------------------------------------------------------------------------------------
def population(kap, N=300, seed=0, sig=0.08, a_unc=0.25, jit=20.0, days=3, dt=0.25, sc=NOMINAL):
    """Applique les doses fixes kap à N patients virtuels (sans ré-optimisation).
    Paramètres p1,p2,p3,n : facteurs log-normaux d'écart-type `sig`; repas : charge +-a_unc (uniforme) ;
    heures de repas : +-jit minutes (uniforme). Indices calculés sur le dernier jour."""
    rng = np.random.default_rng(seed); P = sc.phys
    lognorm = lambda: np.exp(rng.normal(0.0, sig, N))
    p1, p2, p3, n = P["p1"] * lognorm(), P["p2"] * lognorm(), P["p3"] * lognorm(), P["n"] * lognorm()
    A = np.array(sc.amp)[None, :] * (1 + rng.uniform(-a_unc, a_unc, (N, 3)))
    th = np.array(sc.meal_t)[None, :] + rng.uniform(-jit, jit, (N, 3))
    Gb, Ib, tau, Tm = P["Gb"], P["Ib"], P["tau"], P["Tm"]
    y = np.tile([Gb, 0.0, Ib, 0.0], (N, 1))
    inj = {int(round((t + d * T_DAY) / dt)): k for d in range(days) for t, k in zip(sc.inj_t, kap)}

    def h(t):
        s = np.zeros(N)
        for d in range(days):
            for q in range(3):
                t0 = th[:, q] + d * T_DAY
                s += np.where((t >= t0) & (t < t0 + Tm), A[:, q] / tau ** 2 * np.exp(-(t - t0) / tau), 0.0)
        return s

    def F(t, y):
        G, X, I, m = y.T
        return np.stack([-p1 * (G - Gb) - X * G + m, -p2 * X + p3 * (I - Ib), -n * (I - Ib), -m / tau + h(t)], 1)
    G_last, t_last = [], []
    for i in range(int(days * T_DAY / dt)):
        t = i * dt
        if i in inj:
            y[:, 2] += inj[i]
        k1 = F(t, y); k2 = F(t + dt / 2, y + dt / 2 * k1); k3 = F(t + dt / 2, y + dt / 2 * k2); k4 = F(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if t >= (days - 1) * T_DAY:
            G_last.append(y[:, 0].copy()); t_last.append(t + dt - (days - 1) * T_DAY)
    G = np.array(G_last)                              # (temps, patients)
    return dict(t=np.array(t_last), G=G,
                TIR=100 * np.mean((G >= 70) & (G <= 180), 0),
                TAR=100 * np.mean(G > 180, 0), TBR=100 * np.mean(G < 70, 0))
