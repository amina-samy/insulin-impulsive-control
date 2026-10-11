"""
run_all.py : reproduit tous les nombres, tableaux et figures de l'article.

Usage
-----
    python run_all.py                  calcul complet (environ 4 à 5 min), figures AFFICHÉES à la fin
                                       puis enregistrées dans ./figures (EPS + PDF) ; journal : results.txt
    python run_all.py --figures-only   uniquement les 4 figures de l'article (environ 1 à 2 min), affichées
    python run_all.py --no-show        n'affiche rien (serveur sans écran) ; enregistre seulement
    python run_all.py --outdir DOSSIER dossier de sortie des figures (défaut : ./figures)

Les quatre figures affichées sont exactement celles de l'article (Fig. 1 à 4).
Fermer les fenêtres de figures pour terminer le programme.
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(description="Reproduction des résultats de l'article")
ap.add_argument("--figures-only", action="store_true", help="ne calcule que les figures")
ap.add_argument("--no-show", action="store_true", help="n'ouvre pas de fenêtre")
ap.add_argument("--outdir", default=None, help="dossier de sortie des figures")
ARGS = ap.parse_args()
FULL, SHOW = not ARGS.figures_only, not ARGS.no_show
import numpy as np
import matplotlib
if not SHOW:
    matplotlib.use("Agg")
else:                                   # backend interactif si disponible, sinon repli sans écran
    try:
        import matplotlib.pyplot as _p; _p.figure(); _p.close("all")
        if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "cairo", "template"):
            raise RuntimeError("backend non interactif")
    except Exception:
        matplotlib.use("Agg"); SHOW = False
        print("Aucun affichage graphique disponible : les figures sont seulement enregistrées.")
import matplotlib.pyplot as plt
from bergman_impulsive import *

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = ARGS.outdir or os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(HERE, "results.txt" if FULL else "results_figures_only.txt"), "w", encoding="utf-8")


def out(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# EPS ne gère pas la transparence : on utilise des couleurs pleines et claires, jamais d'alpha.
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.5, "legend.frameon": False,
                     "ps.fonttype": 42, "pdf.fonttype": 42})
BLUE, RED, GREEN, BROWN = "#1f5fa8", "#c0392b", "#e6f2ea", "#8b4513"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".eps")); fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.canvas.manager.set_window_title(name) if SHOW and fig.canvas.manager else None
    if not SHOW:
        plt.close(fig)


sc = NOMINAL
if FULL:
    # =========================================================================================
    out("=" * 90, "\n1. DOSES OPTIMALES (SLSQP + gradient exact ; contrôle : L-BFGS-B)", "\n" + "=" * 90)
    t0 = time.time()
    r = optimise([7.11, 10.72, 5.29, 0.05], sc, method="SLSQP", maxiter=300)
    kstar = r.x
    g1 = gradient_forward(kstar); ga, _ = gradient_adjoint(kstar)
    out("kappa* =", np.round(kstar, 4), " somme =", round(kstar.sum(), 3), " (%.0f %% de KTOT)" % (100 * kstar.sum() / KTOT))
    out("J(kappa*) = %.4f   J(0) = %.4e   plancher wI*Ib^2*T = %.3f" %
        (cost(kstar), cost(np.zeros(4)), sc.w["wI"] * sc.phys["Ib"] ** 2 * T_DAY))
    out("gradient (sensibilités)", np.round(g1, 4), "| gradient (adjoint)", np.round(ga, 4))
    out("résidu KKT (gradient projeté sur [0,30]) = %.1e" % kkt_residual(kstar, g1))
    # vérification du gradient adjoint par différences finies centrées (point exact kappa*)
    for hs in (1e-3, 1e-4):
        fd = np.array([(cost(kstar + hs * e) - cost(kstar - hs * e)) / (2 * hs) for e in np.eye(4)])
        out("  FD pas %.0e : erreur relative (2-norme) = %.1e" % (hs, np.linalg.norm(fd - ga) / np.linalg.norm(fd)))
    out("Multi-départ (6 points initiaux x 2 solveurs) :")
    for x0 in ([7.11, 10.72, 5.29, 0.05], [0, 0, 0, 0], [30, 30, 30, 0], [20, 20, 20, 10], [5, 5, 5, 5], [1, 25, 1, 25]):
        row = []
        for m in ("L-BFGS-B", "SLSQP"):
            rr = optimise(x0, sc, method=m, maxiter=300); row.append(np.round(rr.x, 3))
        out("  départ", x0, "->", row[0], row[1], " J=%.4f" % rr.fun)
    out("temps :", round(time.time() - t0), "s")

    # =========================================================================================
    out("\n" + "=" * 90, "\n2. INDICES CLINIQUES SUR LE JOUR NOMINAL (grille de 0,1 min)", "\n" + "=" * 90)
    fmt = lambda d: {a: round(float(b), 1) for a, b in d.items()}
    base0 = day_indices(np.zeros(4)); carb = day_indices([6, 7, 5.5, 0]); opt = day_indices(kstar)
    out("sans bolus     ", fmt(base0)); out("règle glucides  ", fmt(carb)); out("kappa*          ", fmt(opt))
    ts, Y = trajectory(kstar, sc); ts0, Y0 = trajectory(np.zeros(4), sc)
    G = Y[:, 0]; X = Y[:, 1]
    out("temps total G>180 sous kappa* : %.1f min ; pic max %.2f ; min %.2f" % ((G > 180).sum() * 0.1, G.max(), G.min()))
    out("pics de G sous kappa*:", [round(G[(ts > t) & (ts < t + 200)].max(), 1) for t in sc.meal_t],
        " sans bolus:", [round(Y0[:, 0][(ts0 > t) & (ts0 < t + 200)].max(), 1) for t in sc.meal_t])
    out("pics de X :", [round(X[(ts > t) & (ts < t + 250)].max(), 4) for t in sc.inj_t[:3]],
        " retard théorique ln(n/p2)/(n-p2) = %.1f min" % (np.log(sc.phys["n"] / sc.phys["p2"]) / (sc.phys["n"] - sc.phys["p2"])))
    out("pics de m :", [round(Y[:, 3][(ts > t) & (ts < t + 200)].max(), 2) for t in sc.meal_t])
    out("doses par gramme :", [round(k / c, 3) for k, c in zip(kstar[:3], sc.carbs)])
    out("doses en unités (D = 15 kappa V_I / 1000, V_I = 8,4 L) :", [round(15 * k * 8.4 / 1000, 2) for k in kstar[:3]])

    # =========================================================================================
    out("\n" + "=" * 90, "\n3. SENSIBILITÉ : erreurs de dose, repas sautés ou retardés, p3 multiplié", "\n" + "=" * 90)
    for f in (1, 2, 3, 4, 5, 6, 8):
        k = kstar.copy(); k[:3] *= f; out("facteur", f, fmt(day_indices(k)))
    scen = {"Petit-déj. sauté": dict(carbs=(0, 70, 55)), "Déjeuner sauté": dict(carbs=(60, 0, 55)),
            "Dîner sauté": dict(carbs=(60, 70, 0)), "Tous sautés": dict(carbs=(0, 0, 0)),
            "Retard 90 min": dict(meal_t=tuple(t + 90 for t in sc.meal_t))}
    for name, kw in scen.items():
        out(name.ljust(18), fmt(day_indices(kstar, Scenario(**kw))))
    for mu in (1, 2, 3, 4):
        out("p3 x", mu, fmt(day_indices(kstar, sc.with_phys(p3=sc.phys["p3"] * mu))))
    w0 = Scenario(w=dict(WEIGHTS, wl=0.0))
    rw = optimise([7, 10, 5, 0], w0, method="SLSQP", maxiter=300)
    out("optimum avec w_hypo = 0 :", np.round(rw.x, 3), "(identique : le terme hypo est inactif)")

    # =========================================================================================
    out("\n" + "=" * 90, "\n4. RÉGIME PÉRIODIQUE : orbite, coût périodique, gradient périodique", "\n" + "=" * 90)
    y0 = periodic_orbit(kstar, sc); basal = np.array([sc.phys["Gb"], 0, sc.phys["Ib"], 0])
    out("Y0 sur l'orbite :", np.round(y0, 4), "| distance l1 à l'état basal : %.3f" % np.abs(y0 - basal).sum(),
        "| G0-Gb = %.3f" % (y0[0] - basal[0]))
    Jd, Jp = cost(kstar, sc), cost_periodic(kstar, sc)
    out("J un jour = %.6f ; J périodique = %.6f ; écart relatif = %.1e" % (Jd, Jp, abs(Jp - Jd) / Jd))
    out("gradient un jour   :", np.round(gradient_adjoint(kstar, sc)[0], 4))
    gper = gradient_periodic(kstar, sc); out("gradient périodique:", np.round(gper, 4))
    hs = 1e-4
    fdp = np.array([(cost_periodic(kstar + hs * e, sc) - cost_periodic(kstar - hs * e, sc)) / (2 * hs) if i < 3
                    else (cost_periodic(kstar + hs * e, sc) - cost_periodic(kstar, sc)) / hs for i, e in enumerate(np.eye(4))])
    out("gradient périodique par différences finies :", np.round(fdp, 4))
    out("écart relatif du composant 4 (coucher) un-jour vs périodique : %.1f %%" %
        (100 * abs(gper[3] - gradient_adjoint(kstar, sc)[0][3]) / abs(gper[3])))
    # multiplicateurs de Floquet pour l'horizon réel de 1440 min (valeurs analytiques)
    P = sc.phys; nu = min(P["p1"], P["p2"], P["n"], 1 / P["tau"])
    Ts = sc.period
    _, ts_f, Yf = integrate(kstar, sc, y0=y0, dt=0.1)
    intX = np.trapezoid(Yf[:, 1], ts_f)
    out("Floquet (T=1440) : mu_X=%.2e mu_I=%.2e mu_m=%.2e mu_G=%.2e (int X = %.3f) ; nu = %.5f" %
        (np.exp(-P["p2"] * Ts), np.exp(-P["n"] * Ts), np.exp(-Ts / P["tau"]), np.exp(-(P["p1"] * Ts + intX)), intX, nu))
    # vérification numérique de l'équation variationnelle sur une période courte (Tp = 240 min)
    Tp = 240.0
    scf = Scenario(carbs=(60.0,), meal_t=(30.0,), inj_t=(20.0,), period=Tp)
    kf = [12.0]
    yf = periodic_orbit(kf, scf, n_iter=8)
    Psi, intXf = monodromy(kf, scf, Tp, yf)
    ev = np.sort(np.linalg.eigvals(Psi).real)
    cf = {"mu_G": np.exp(-(P["p1"] * Tp + intXf)), "mu_X": np.exp(-P["p2"] * Tp), "mu_I": np.exp(-P["n"] * Tp), "mu_m": np.exp(-Tp / P["tau"])}
    out("Floquet, période courte Tp = 240, kappa = 12 :")
    for key, val in cf.items():
        num = ev[np.argmin(np.abs(ev - val))]
        out("  %s  forme close %.6e  numérique %.6e  erreur relative %.1e" % (key, val, num, abs(num - val) / val))
    _, tt, YY = integrate(kf, scf, days=8, dt=0.1)
    npp = int(round(Tp / 0.1))
    dev = np.abs(YY[-npp:, 0] - YY[-2 * npp:-npp, 0]).max()
    out("  écart max |G(t)-G(t-Tp)| entre les deux dernières périodes : %.2e mg/dL" % dev)

    # =========================================================================================
    out("\n" + "=" * 90, "\n5. POPULATION VIRTUELLE (N = 300, jour 3, doses fixes)", "\n" + "=" * 90)
    q = lambda v: np.percentile(v, [5, 25, 50, 75, 95]).round(1)
    pops = {}
    for seed in range(6):
        pp = population(kstar, N=300, seed=seed); pops[seed] = pp
        out("graine", seed, "TIR", q(pp["TIR"]), "| TAR", q(pp["TAR"]), "| TBR", q(pp["TBR"]),
            "| min TIR %.1f | max TBR %.2f | patients TBR>0 : %d" % (pp["TIR"].min(), pp["TBR"].max(), (pp["TBR"] > 0).sum()))
    pp = pops[0]
    band = np.percentile(pp["G"], [5, 25, 50, 75, 95], axis=1)
    out("bande 5-95 : max %.1f ; min %.1f mg/dL" % (band[4].max(), band[0].min()))

else:
    # mode --figures-only : on ne calcule que ce dont les figures ont besoin
    kstar = optimise([7.11, 10.72, 5.29, 0.05], sc, method="SLSQP", maxiter=300).x
    ts, Y = trajectory(kstar, sc); ts0, Y0 = trajectory(np.zeros(4), sc)
    y0 = periodic_orbit(kstar, sc); P = sc.phys; nu = min(P["p1"], P["p2"], P["n"], 1 / P["tau"])
    pp = population(kstar, N=300, seed=0); band = np.percentile(pp["G"], [5, 25, 50, 75, 95], axis=1)
    out("kappa* =", np.round(kstar, 4), " (mode figures seules)")

# =========================================================================================
out("\n" + "=" * 90, "\n6. FIGURES (EPS + PDF)", "\n" + "=" * 90)
th_h = np.array(sc.meal_t) / 60.0
# ---- Fig. 1 : jour nominal -------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
h = ts / 60.0


def panel(a, lab):
    a.text(0.015, 0.93, lab, transform=a.transAxes, fontweight="bold", va="top"); a.set_xlim(0, 24)
    a.set_xticks(range(0, 25, 4))
    for t in th_h: a.axvline(t, color="0.6", ls=":", lw=0.7)


a = ax[0, 0]; panel(a, "(a)")
a.axhspan(70, 180, color=GREEN, lw=0, zorder=0)
a.plot(ts0 / 60, Y0[:, 0], "--", color=RED, lw=1.3, label="no bolus")
a.plot(h, Y[:, 0], color=BLUE, lw=1.5, label=r"optimal doses $\boldsymbol{\kappa}^*$")
a.set_ylabel(r"$G(t)$ [mg/dL]"); a.set_ylim(70, 280)
a.legend(loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2, fontsize=8, columnspacing=1.0, handlelength=1.8)
a = ax[0, 1]; panel(a, "(b)")
a.plot(ts0 / 60, Y0[:, 2], "--", color=RED, lw=1.3); a.plot(h, Y[:, 2], color=BLUE, lw=1.5)
for t, k in zip(sc.inj_t, kstar):
    if k > 0: a.annotate(f"${k:.2f}$", (t / 60, sc.phys["Ib"] + k), xytext=(t / 60 + 0.25, sc.phys["Ib"] + k - 0.4), fontsize=8, color="#2e6e2e")
a.set_ylabel(r"$I(t)$ [mU/L]"); a.set_ylim(6.5, 19)
a = ax[1, 0]; panel(a, "(c)")
a.plot(ts0 / 60, Y0[:, 1], "--", color=RED, lw=1.3); a.plot(h, Y[:, 1], color=BLUE, lw=1.5)
a.set_ylabel(r"$X(t)$ [min$^{-1}$]"); a.set_xlabel("Time [h]")
a = ax[1, 1]; panel(a, "(d)")
a.plot(h, Y[:, 3], color=BROWN, lw=1.5)
for t, c in zip(th_h, sc.carbs): a.text(t + 0.25, Y[:, 3].max() * 1.04, f"{c:.0f} g", fontsize=8)
a.set_ylabel(r"$m(t)$ [mg/dL/min]"); a.set_ylim(0, Y[:, 3].max() * 1.2); a.set_xlabel("Time [h]")
fig.tight_layout(w_pad=1.6, h_pad=1.0); save(fig, "fig_nominal_day")

# ---- Fig. 2 : convergence vers l'orbite ------------------------------------------------
days = 3
ts_r, Yr = trajectory(kstar, sc, days=days, y0=y0, dt=1.0)            # orbite de référence
inits = {"high glucose": [250, 0, sc.phys["Ib"], 0], "low glucose, high insulin": [60, 0.02, 25, 0],
         "meal in the gut": [sc.phys["Gb"], 0, sc.phys["Ib"], 15]}
cols = [BLUE, RED, "#2e8b57"]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
ax[0].plot(ts_r / 60, Yr[:, 0], "k--", lw=1.1, label=r"orbit $Y^*$")
ax[1].plot(Yr[:, 0], Yr[:, 1], "k--", lw=1.1)
Cemp = 0
for (nm, y_init), c in zip(inits.items(), cols):
    _, Ya = trajectory(kstar, sc, days=days, y0=y_init, dt=1.0)
    ax[0].plot(ts_r / 60, Ya[:, 0], color=c, lw=1.0, label=nm)
    ax[1].plot(Ya[:, 0], Ya[:, 1], color=c, lw=0.9)
    d = np.abs(Ya - Yr).sum(1); d0 = d[0]; rel = np.maximum(d / d0, 1e-17)
    ax[2].semilogy(ts_r / 60, rel, color=c, lw=1.0)
    i_floor = int(np.argmax(rel < 1e-9))   # au-delà de ce point on est dans le bruit numérique du solveur
    Cemp = max(Cemp, np.max((d / d0 * np.exp(nu * ts_r))[:i_floor]))
Cemp = float(np.ceil(Cemp))
ax[2].semilogy(ts_r / 60, np.exp(-nu * ts_r), "k:", lw=1.1, label=r"$e^{-\nu t}$")
ax[2].semilogy(ts_r / 60, Cemp * np.exp(-nu * ts_r), "k-.", lw=1.0, label=r"$Ce^{-\nu t}$, $C=%.0f$" % Cemp)
ax[2].set_ylim(1e-13, 1e3)
ax[0].set_xlabel("Time [h]"); ax[0].set_ylabel(r"$G(t)$ [mg/dL]")
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.0))
ax[1].set_xlabel(r"$G$ [mg/dL]"); ax[1].set_ylabel(r"$X$ [min$^{-1}$]")
ax[2].set_xlabel("Time [h]"); ax[2].set_ylabel(r"$\|Y-Y^*\|_1/\|Y_0-Y^*_0\|_1$"); ax[2].legend(fontsize=6, loc="upper right")
for a, l in zip(ax, ("(a)", "(b)", "(c)")): a.text(0.02, 0.95, l, transform=a.transAxes, fontweight="bold", va="top")
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.92)); save(fig, "fig_convergence")
out("convergence : constante empirique C = %.1f (avant le plancher numérique 1e-9)" % Cemp)

# ---- Fig. 3 : population ---------------------------------------------------------------
fig, a = plt.subplots(figsize=(6.2, 3.3))
tp = pp["t"] / 60
a.axhspan(70, 180, color=GREEN, lw=0, zorder=0)
a.fill_between(tp, band[0], band[4], color="#c6dbef", lw=0, label="5th–95th pct")
a.fill_between(tp, band[1], band[3], color="#6baed6", lw=0, label="25th–75th pct")
a.plot(tp, band[2], color=BLUE, lw=1.5, label="median")
a.axhline(180, color="0.4", ls="--", lw=0.8); a.axhline(70, color="0.4", ls="--", lw=0.8)
a.set_xlim(0, 24); a.set_xticks(range(0, 25, 4)); a.set_xlabel("Time [h]"); a.set_ylabel(r"$G(t)$ [mg/dL]")
a.legend(loc="upper right", fontsize=8); fig.tight_layout(); save(fig, "fig_population")

# ---- Fig. 4 : tests de stress ----------------------------------------------------------
names = ["Nominal", "B", "L", "D", "All", "Delay\n90 min"]
sc_kw = {"Nominal": {}, "B": dict(carbs=(0, 70, 55)), "L": dict(carbs=(60, 0, 55)), "D": dict(carbs=(60, 70, 0)),
         "All": dict(carbs=(0, 0, 0)), "Delay\n90 min": dict(meal_t=tuple(t + 90 for t in sc.meal_t))}
R = {n: trajectory(kstar, Scenario(**kw), dt=0.1) for n, kw in sc_kw.items()}
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.6))


def traj(a, key, lab):
    a.axhspan(28, 54, color="#e9bdb6", lw=0, zorder=0); a.axhspan(54, 70, color="#f4dad5", lw=0, zorder=0)
    a.axhline(70, color="0.4", ls=":", lw=0.8); a.axhline(54, color="0.4", ls="-.", lw=0.8)
    a.plot(R["Nominal"][0] / 60, R["Nominal"][1][:, 0], "--", color="0.35", lw=1.1, label="nominal day")
    a.plot(R[key][0] / 60, R[key][1][:, 0], color=BLUE, lw=1.5, label=lab)
    a.plot(np.array(sc.inj_t[:3]) / 60, [32] * 3, "^", color="k", ms=4)
    a.set_xlim(0, 24); a.set_xticks(range(0, 25, 4)); a.set_ylim(28, 290)
    a.set_xlabel("Time [h]"); a.set_ylabel(r"$G(t)$ [mg/dL]"); a.legend(loc="upper left", bbox_to_anchor=(0.09, 1.0), fontsize=8)


traj(ax[0, 0], "L", "lunch skipped"); traj(ax[0, 1], "Delay\n90 min", "meals delayed by 90 min")
for a, l in zip(ax[0], ("(a)", "(b)")): a.text(0.015, 0.95, l, transform=a.transAxes, fontweight="bold", va="top")
mu = np.linspace(0.5, 4.0, 15); sz = np.linspace(0.0, 1.4, 15); Z = np.zeros((len(sz), len(mu)))
for i, s in enumerate(sz):
    for j, m_ in enumerate(mu):
        _, Yz = trajectory(kstar, Scenario(carbs=tuple(s * c for c in sc.carbs)).with_phys(p3=sc.phys["p3"] * m_), dt=0.5)
        Z[i, j] = Yz[:, 0].min()
a = ax[1, 0]
cs = a.contourf(mu, sz, Z, levels=np.arange(30, 100.1, 5), cmap="YlGnBu", extend="both"); a.grid(False)
c1 = a.contour(mu, sz, Z, [70], colors="k", linewidths=1.3, linestyles=":"); c2 = a.contour(mu, sz, Z, [54], colors="k", linewidths=1.3, linestyles="-.")
a.clabel(c1, fmt="70", fontsize=8); a.clabel(c2, fmt="54", fontsize=8); a.plot(1, 1, marker="*", color="#f1c40f", mec="k", ms=11)
a.set_xlabel(r"insulin-action multiplier $\mu$"); a.set_ylabel("meal size / nominal")
fig.colorbar(cs, ax=a, pad=0.02).set_label(r"$\min_t G$ [mg/dL]")
a.text(0.03, 0.95, "(c)", transform=a.transAxes, fontweight="bold", va="top")
a = ax[1, 1]; vals = [R[n][1][:, 0].min() for n in names]
a.bar(range(6), vals, color=[BLUE] + ["#5b8fc9"] * 5, width=0.65)
a.axhline(70, color="0.4", ls=":", lw=1); a.axhline(54, color="0.4", ls="-.", lw=1)
for i, v in enumerate(vals): a.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
a.set_xticks(range(6)); a.set_xticklabels(["Nominal", "B", "L", "D", "All", "Delay\n90 min"]); a.set_ylim(0, 105)
a.set_ylabel(r"$\min_t G$ [mg/dL]"); a.text(0.03, 0.95, "(d)", transform=a.transAxes, fontweight="bold", va="top"); a.grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.5, h_pad=1.2); save(fig, "fig_stress")
out("minima par scénario (Fig. 4d) :", [round(v, 1) for v in vals])
out("figures écrites dans", os.path.abspath(FIG))
if SHOW:
    print("Affichage des figures : fermez les fenêtres pour terminer.")
    plt.show()
