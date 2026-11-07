# =============================================================================
# Rhizopus delemar R144 – complete statistical re-analysis
# Reproduces every number, table and figure used in the revised manuscript.
# Input : Tur_compiled_analysis_ready.xlsx  (sheets: Temperature_Growth,
#         pH_Growth, Enzyme_Activity, Lesion_Size)
# Output: ./results/  (tables .csv/.xlsx, figures .png/.tif 600 dpi, results.json)
# =============================================================================
#%% ---- 0. Setup --------------------------------------------------------------
import json, os, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
warnings.filterwarnings("ignore")

DATA = "data/Tur_compiled_analysis_ready.xlsx"
OUT = "results"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(2026)          # fixed seed -> identical bootstrap CIs
N_BOOT = 2000
ALPHA = 0.05

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": FONT, "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "savefig.dpi": 600, "figure.dpi": 150, "pdf.fonttype": 42,
})
BAR = "#8DA9C4"      # single neutral fill: one series per panel, so no colour coding
INK = "#1F2A36"
DOT = "#2E4057"

temp = pd.read_excel(DATA, "Temperature_Growth")
ph = pd.read_excel(DATA, "pH_Growth")
enz = pd.read_excel(DATA, "Enzyme_Activity")
les = pd.read_excel(DATA, "Lesion_Size")
enz["Enzyme"] = enz["Enzyme"].str.strip()
les["LW_ratio"] = les["Length_cm"] / les["Width_cm"]
les["LxW_cm2"] = les["Length_cm"] * les["Width_cm"]
ENZ_ORDER = list(dict.fromkeys(enz["Enzyme"]))
RES = {}

#%% ---- 1. Helper functions ---------------------------------------------------
def describe(df, g, y):
    d = df.groupby(g, sort=False)[y].agg(n="count", mean="mean", sd="std")
    d["se"] = d["sd"] / np.sqrt(d["n"])
    t = stats.t.ppf(0.975, d["n"] - 1)
    d["ci_low"], d["ci_high"] = d["mean"] - t * d["se"], d["mean"] + t * d["se"]
    d["cv_pct"] = 100 * d["sd"] / d["mean"]
    return d


def cld(groups, means, sig_pairs):
    """Compact letter display (insert-and-absorb, Piepho 2004).
    Groups sharing a letter are not significantly different. 'a' = highest mean."""
    order = [g for _, g in sorted(zip(means, groups), key=lambda z: -z[0])]
    cols = [set(order)]
    for a, b in sig_pairs:
        new = []
        for c in cols:
            if a in c and b in c:
                new += [c - {a}, c - {b}]
            else:
                new.append(c)
        new = [c for c in new if c]
        cols = [c for c in new if not any(c < o for o in new)]
        uniq = []
        for c in cols:
            if c not in uniq:
                uniq.append(c)
        cols = uniq
    cols.sort(key=lambda c: min(order.index(g) for g in c))
    letters = {g: "" for g in groups}
    for i, c in enumerate(cols):
        for g in c:
            letters[g] += "abcdefghijklmnopqrstuvwxyz"[i]
    return {g: "".join(sorted(l)) for g, l in letters.items()}


def welch_anova(groups):
    n = np.array([len(g) for g in groups], float); m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups]); k = len(groups)
    w = n / v; mw = np.sum(w * m) / np.sum(w)
    A = np.sum(w * (m - mw) ** 2) / (k - 1)
    tmp = np.sum((1 - w / np.sum(w)) ** 2 / (n - 1))
    F = A / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp)
    df1, df2 = k - 1, (k ** 2 - 1) / (3 * tmp)
    return F, df1, df2, stats.f.sf(F, df1, df2)


def games_howell(names, groups):
    out = []; k = len(groups)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = np.asarray(groups[i]), np.asarray(groups[j])
            va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
            d = a.mean() - b.mean()
            q = abs(d) / np.sqrt((va + vb) / 2)
            dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
            out.append((names[i], names[j], a.mean(), b.mean(), d, float(stats.studentized_range.sf(q, k, dfw))))
    return out


def tukey(names, groups):
    P = stats.tukey_hsd(*groups).pvalue; out = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = np.mean(groups[i]), np.mean(groups[j])
            out.append((names[i], names[j], a, b, a - b, float(P[i, j])))
    return out


def oneway(df, g, y, label):
    """Assumption checks -> classic ANOVA + Tukey HSD (homogeneous variances, Levene p >= 0.05)
    or Welch ANOVA + Games-Howell (heterogeneous variances)."""
    d = df[[g, y]].dropna().copy(); d[g] = d[g].astype(str)
    names = list(dict.fromkeys(d[g])); groups = [d.loc[d[g] == n, y].values for n in names]
    resid = d[y] - d.groupby(g)[y].transform("mean")
    sw = stats.shapiro(resid)
    lev = stats.levene(*groups, center="median")
    grand = d[y].mean()
    ss_b = sum(len(x) * (x.mean() - grand) ** 2 for x in groups)
    ss_w = sum(((x - x.mean()) ** 2).sum() for x in groups)
    df_b, df_w = len(groups) - 1, len(d) - len(groups); ms_w = ss_w / df_w
    eta2 = ss_b / (ss_b + ss_w); omega2 = (ss_b - df_b * ms_w) / (ss_b + ss_w + ms_w)
    if lev.pvalue >= ALPHA:
        F, p = stats.f_oneway(*groups); df1, df2 = df_b, df_w
        test, post, pairs = "One-way ANOVA", "Tukey HSD", tukey(names, groups)
    else:
        F, df1, df2, p = welch_anova(groups)
        test, post, pairs = "Welch ANOVA", "Games-Howell", games_howell(names, groups)
    ph_ = pd.DataFrame(pairs, columns=["A", "B", "mean(A)", "mean(B)", "diff", "p_adj"])
    sig = [(r.A, r.B) for r in ph_.itertuples() if r.p_adj < ALPHA]
    letters = cld(names, [x.mean() for x in groups], sig)
    ph_.insert(0, "variable", label); ph_.insert(1, "test", post)
    kw = stats.kruskal(*groups)
    summary = dict(variable=label, test=test, posthoc=post, F=float(F), df1=float(df1),
                   kruskal_H=float(kw.statistic), kruskal_p=float(kw.pvalue),
                   df2=float(df2), p=float(p), eta2=float(eta2), omega2=float(omega2),
                   shapiro_W=float(sw.statistic), shapiro_p=float(sw.pvalue),
                   levene_F=float(lev.statistic), levene_p=float(lev.pvalue))
    return summary, letters, ph_


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"

#%% ---- 2. Cardinal models ----------------------------------------------------
def ctmi(T, mu_opt, Tmin, Topt, Tmax):
    """Cardinal Temperature Model with Inflection (Rosso et al. 1993)."""
    T = np.asarray(T, float)
    num = (T - Tmax) * (T - Tmin) ** 2
    den = (Topt - Tmin) * ((Topt - Tmin) * (T - Topt) - (Topt - Tmax) * (Topt + Tmin - 2 * T))
    out = mu_opt * num / den
    return np.where((T > Tmin) & (T < Tmax), out, 0.0)


def betaYH(T, mu_opt, Tmin, Topt, Tmax):
    """Beta function of Yan & Hunt (1999), generalised with a base temperature."""
    T = np.asarray(T, float)
    a = (Topt - Tmin) / (Tmax - Topt)
    x1 = np.clip((Tmax - T) / (Tmax - Topt), 0, None)
    x2 = np.clip((T - Tmin) / (Topt - Tmin), 0, None)
    return np.where((T > Tmin) & (T < Tmax), mu_opt * x1 * x2 ** a, 0.0)


def cpm(pH, mu_opt, pHmin, pHopt, pHmax):
    """Cardinal pH Model (Rosso et al. 1995)."""
    x = np.asarray(pH, float)
    num = (x - pHmin) * (x - pHmax)
    den = num - (x - pHopt) ** 2
    return np.where((x > pHmin) & (x < pHmax), mu_opt * num / den, 0.0)


def gauss(pH, mu_opt, pHopt, sigma):
    """Gaussian pH-response curve; pH range giving >= 50 % of maximum = pHopt ± 1.1774·sigma."""
    return mu_opt * np.exp(-(np.asarray(pH, float) - pHopt) ** 2 / (2 * sigma ** 2))


def fit_model(f, x, y, p0, bounds, names):
    popt, pcov = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=200000)
    yhat = f(x, *popt)
    rss = float(np.sum((y - yhat) ** 2)); n, k = len(y), len(popt)
    r2 = 1 - rss / float(np.sum((y - y.mean()) ** 2))
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    aic = n * np.log(rss / n) + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    # nonparametric bootstrap: resample replicates within each level
    levels = np.unique(x); idx = {l: np.where(x == l)[0] for l in levels}
    boots = []
    for _ in range(N_BOOT):
        s = np.concatenate([RNG.choice(idx[l], size=len(idx[l]), replace=True) for l in levels])
        try:
            pb, _ = curve_fit(f, x[s], y[s], p0=popt, bounds=bounds, maxfev=20000)
            boots.append(pb)
        except Exception:
            pass
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    tq = stats.t.ppf(0.975, n - k); se = np.sqrt(np.diag(pcov))
    params = {nm: dict(estimate=float(popt[i]), se=float(se[i]),
                       ci_low=float(popt[i] - tq * se[i]), ci_high=float(popt[i] + tq * se[i]),
                       boot_low=float(lo[i]), boot_high=float(hi[i])) for i, nm in enumerate(names)}
    return dict(params=params, R2=r2, adjR2=adj, RMSE=float(np.sqrt(rss / n)), AIC=float(aic),
                AICc=float(aicc), n=n, n_boot=int(len(boots)), popt=popt.tolist())

#%% ---- 3. Temperature and pH -------------------------------------------------
x_t, y_t = temp["Temperature_C"].values.astype(float), temp["Radial_Growth_mm"].values.astype(float)
x_p, y_p = ph["pH"].values.astype(float), ph["Radial_Growth_mm"].values.astype(float)
TN = ["mu_opt (mm)", "Tmin (°C)", "Topt (°C)", "Tmax (°C)"]
PN = ["mu_opt (mm)", "pHmin", "pHopt", "pHmax"]
RES["ctmi"] = fit_model(ctmi, x_t, y_t, [90, 5, 29, 42], ([10, -10, 20, 40.5], [150, 9.9, 39, 60]), TN)
RES["beta"] = fit_model(betaYH, x_t, y_t, [90, 5, 29, 42], ([10, -10, 20, 40.5], [150, 9.9, 39, 60]), TN)
RES["cpm"] = fit_model(cpm, x_p, y_p, [88, 2.5, 6, 9], ([10, 0, 4.1, 8.05], [150, 2.99, 7.9, 14]), PN)
RES["gauss"] = fit_model(gauss, x_p, y_p, [88, 5.7, 1.3], ([10, 4, 0.2], [150, 8, 5]), ["mu_opt (mm)", "pHopt", "sigma (pH units)"])
_g = RES["gauss"]["params"]
RES["gauss"]["pH50_range"] = [_g["pHopt"]["estimate"] - 1.1774 * _g["sigma (pH units)"]["estimate"],
                              _g["pHopt"]["estimate"] + 1.1774 * _g["sigma (pH units)"]["estimate"]]

# performance breadth: range over which predicted growth >= 50 % / 80 % of the fitted maximum
from scipy.optimize import brentq, minimize_scalar
_p = RES["ctmi"]["popt"]; _f = lambda T: ctmi(T, *_p)
_opt = minimize_scalar(lambda T: -_f(T), bounds=(_p[1] + 1, _p[3] - 0.1), method="bounded")
RES["ctmi"]["breadth"] = {str(fr): [brentq(lambda T: _f(T) - fr * -_opt.fun, _p[1] + 1e-3, _opt.x),
                                    brentq(lambda T: _f(T) - fr * -_opt.fun, _opt.x, _p[3] - 1e-3)] for fr in (0.5, 0.8)}
_g = RES["gauss"]["popt"]
RES["gauss"]["breadth"] = {str(fr): [_g[1] - _g[2] * np.sqrt(2 * np.log(1 / fr)), _g[1] + _g[2] * np.sqrt(2 * np.log(1 / fr))] for fr in (0.5, 0.8)}
# breadth of the beta model (best temperature model by AICc)
_p = RES["beta"]["popt"]; _f = lambda T: betaYH(T, *_p)
_opt = minimize_scalar(lambda T: -_f(T), bounds=(_p[1] + 1, _p[3] - 0.1), method="bounded")
RES["beta"]["breadth"] = {str(fr): [brentq(lambda T: _f(T) - fr * -_opt.fun, _p[1] + 1e-3, _opt.x),
                                    brentq(lambda T: _f(T) - fr * -_opt.fun, _opt.x, _p[3] - 1e-3)] for fr in (0.5, 0.8)}

# model selection: candidate models compared by AICc; only the lowest-AICc model is reported in the text
def briere2(T, a, Tmin, Tmax, m):
    """Briere-2 model (Briere et al. 1999)."""
    T = np.asarray(T, float)
    return np.where((T > Tmin) & (T < Tmax), a * T * (T - Tmin) * np.clip(Tmax - T, 0, None) ** (1 / m), 0.0)
def quadratic(x, a, b, c):
    x = np.asarray(x, float); return a + b * x + c * x ** 2
_cands = {"temperature": [("CTMI", ctmi, [90, 5, 29, 42], ([10, -10, 20, 40.5], [150, 9.9, 39, 60])),
                          ("Beta", betaYH, [90, 5, 29, 42], ([10, -10, 20, 40.5], [150, 9.9, 39, 60])),
                          ("Briere-2", briere2, [0.01, 3, 42, 2], ([1e-6, -10, 40.5, 0.5], [1, 9.9, 60, 10])),
                          ("Quadratic", quadratic, [0, 1, -0.01], (-np.inf, np.inf))],
          "pH": [("Gaussian", gauss, [88, 5.7, 1.3], ([10, 4, 0.2], [150, 8, 5])),
                 ("CPM", cpm, [88, 2.5, 6, 9], ([10, 0, 4.1, 8.05], [150, 2.99, 7.9, 14])),
                 ("Beta", betaYH, [88, 2.5, 6, 9], ([10, 0, 4.1, 8.05], [150, 2.99, 7.9, 14])),
                 ("Quadratic", quadratic, [0, 30, -3], (-np.inf, np.inf))]}
RES["model_selection"] = {}
for _fac, _x, _y, _grid in [("temperature", x_t, y_t, np.linspace(10, 40, 3001)), ("pH", x_p, y_p, np.linspace(3, 8, 3001))]:
    _rows = []
    for _nm, _fn, _p0, _bd in _cands[_fac]:
        _pp, _ = curve_fit(_fn, _x, _y, p0=_p0, bounds=_bd, maxfev=200000)
        _res = _y - _fn(_x, *_pp); _n, _k = len(_y), len(_pp); _rss = float((_res ** 2).sum())
        _rows.append(dict(model=_nm, k=_k, R2=1 - _rss / float(((_y - _y.mean()) ** 2).sum()), RMSE=float(np.sqrt(_rss / _n)),
                          AICc=_n * np.log(_rss / _n) + 2 * _k + 2 * _k * (_k + 1) / (_n - _k - 1),
                          optimum=float(_grid[np.argmax(_fn(_grid, *_pp))]), shapiro_p=float(stats.shapiro(_res).pvalue)))
    _m = min(r["AICc"] for r in _rows); _w = [np.exp(-(r["AICc"] - _m) / 2) for r in _rows]
    for r, w in zip(_rows, _w):
        r["dAICc"] = r["AICc"] - _m; r["weight"] = w / sum(_w)
    RES["model_selection"][_fac] = sorted(_rows, key=lambda r: r["AICc"])
desc_t, desc_p = describe(temp, "Temperature_C", "Radial_Growth_mm"), describe(ph, "pH", "Radial_Growth_mm")
s_t, L_t, ph_t = oneway(temp, "Temperature_C", "Radial_Growth_mm", "Radial growth – temperature")
s_p, L_p, ph_p = oneway(ph, "pH", "Radial_Growth_mm", "Radial growth – pH")
desc_t["letter"] = [L_t[str(i)] for i in desc_t.index]
desc_p["letter"] = [L_p[str(i)] for i in desc_p.index]

#%% ---- 4. Enzymes ------------------------------------------------------------
desc_e = describe(enz, "Enzyme", "Activity_U_per_ml").loc[ENZ_ORDER]
s_e, L_e, ph_e = oneway(enz, "Enzyme", "Activity_U_per_ml", "Enzyme activity")
desc_e["letter"] = [L_e[i] for i in desc_e.index]
desc_abs = describe(enz, "Enzyme", "Absorbance").loc[ENZ_ORDER]
desc_e["fold_vs_pectinase"] = desc_e["mean"] / desc_e.loc["Pectinase", "mean"]

#%% ---- 5. Lesion (discoloration-zone) development ----------------------------
LES_VARS = [("Length_cm", "Length (cm)"), ("Width_cm", "Width (cm)"),
            ("Area_cm2", "Area (cm²)"), ("LW_ratio", "Length:width ratio")]
les_desc, les_sum, les_post = {}, [], []
for v, lab in LES_VARS:
    d = describe(les, "DAI", v)
    s, L, p_ = oneway(les, "DAI", v, lab)
    d["letter"] = [L[str(i)] for i in d.index]
    les_desc[v] = d; les_sum.append(s); les_post.append(p_)
# exponential growth of discoloured area: ln(A) = ln(A0) + r*t
lr = stats.linregress(les["DAI"], np.log(les["Area_cm2"]))
r = lr.slope; ci_r = stats.t.ppf(0.975, len(les) - 2) * lr.stderr
RES["exp_area"] = dict(r_per_day=r, r_ci=[r - ci_r, r + ci_r], A0=float(np.exp(lr.intercept)),
                       R2=lr.rvalue ** 2, p=lr.pvalue, doubling_days=np.log(2) / r,
                       doubling_ci=[np.log(2) / (r + ci_r), np.log(2) / (r - ci_r)])
lin = stats.linregress(les["DAI"], les["Area_cm2"])
RES["lin_area"] = dict(R2=lin.rvalue ** 2)
# relative expansion rates of length and width (slopes of ln(dimension) on DAI) and their difference
_rL, _rW = stats.linregress(les["DAI"], np.log(les["Length_cm"])), stats.linregress(les["DAI"], np.log(les["Width_cm"]))
_t = (_rL.slope - _rW.slope) / np.sqrt(_rL.stderr ** 2 + _rW.stderr ** 2)
_tq = stats.t.ppf(0.975, len(les) - 2)
RES["elongation"] = dict(r_length=_rL.slope, r_length_ci=[_rL.slope - _tq * _rL.stderr, _rL.slope + _tq * _rL.stderr],
                         r_width=_rW.slope, r_width_ci=[_rW.slope - _tq * _rW.stderr, _rW.slope + _tq * _rW.stderr],
                         ratio=_rL.slope / _rW.slope, t=_t, df=2 * len(les) - 4, p=2 * stats.t.sf(abs(_t), 2 * len(les) - 4))
# standardized major axis (SMA) of ln(length) on ln(width): slope > 1 = length grows disproportionately
_x, _y = np.log(les["Width_cm"]), np.log(les["Length_cm"]); _r = np.corrcoef(_x, _y)[0, 1]
_b = np.sign(_r) * _y.std() / _x.std(); _B = stats.t.ppf(0.975, len(les) - 2) ** 2 * (1 - _r ** 2) / (len(les) - 2)
RES["sma"] = dict(slope=_b, ci=[_b * (np.sqrt(_B + 1) - np.sqrt(_B)), _b * (np.sqrt(_B + 1) + np.sqrt(_B))], r=_r)
# competing two-parameter models for area on the raw scale (AICc, Akaike weights)
_tt, _A = les["DAI"].to_numpy(float), les["Area_cm2"].to_numpy(float)
_mods = {"linear": (lambda t, a, b: a + b * t, [0, 0.1]), "exponential": (lambda t, a, r: a * np.exp(r * t), [0.5, 0.017]),
         "power": (lambda t, a, b: a * t ** b, [1e-3, 1.5])}
_aicc = {}
for _k, (_fn, _p0) in _mods.items():
    _pp, _ = curve_fit(_fn, _tt, _A, p0=_p0, maxfev=100000)
    _rss = float(((_A - _fn(_tt, *_pp)) ** 2).sum()); _n, _kk = len(_A), 3
    _aicc[_k] = dict(params=_pp.tolist(), AICc=_n * np.log(_rss / _n) + 2 * _kk + 2 * _kk * (_kk + 1) / (_n - _kk - 1))
_m = min(v["AICc"] for v in _aicc.values()); _w = {k: np.exp(-(v["AICc"] - _m) / 2) for k, v in _aicc.items()}
for _k in _aicc:
    _aicc[_k]["dAICc"] = _aicc[_k]["AICc"] - _m; _aicc[_k]["weight"] = _w[_k] / sum(_w.values())
RES["area_models"] = _aicc
# mean expansion rates between harvests (cm² or cm per day)
m = les.groupby("DAI")[["Length_cm", "Width_cm", "Area_cm2"]].mean()
# (no 0–60 d rate: the wound itself has non-zero size at day 0 and no day-0 measurement exists)
rates = pd.DataFrame({"interval": ["60–120", "120–180"]})
for v in ["Length_cm", "Width_cm", "Area_cm2"]:
    vals = [(m.loc[120, v] - m.loc[60, v]) / 60, (m.loc[180, v] - m.loc[120, v]) / 60]
    rates[v + "_per_day"] = vals
fold = (m.loc[180] / m.loc[60]).to_dict()
RES["lesion_fold_60_180"] = fold
# agreement between recorded area and L x W
diff = les["Area_cm2"] - les["LxW_cm2"]
RES["area_vs_LxW"] = dict(mean_diff=float(diff.mean()), sd_diff=float(diff.std()),
                          max_abs=float(diff.abs().max()),
                          r=float(stats.pearsonr(les["Area_cm2"], les["LxW_cm2"])[0]),
                          n_equal_2dp=int((diff.abs() < 0.005).sum()))
RES["L_W_corr"] = {int(k): float(stats.spearmanr(g["Length_cm"], g["Width_cm"])[0]) for k, g in les.groupby("DAI")}

#%% ---- 6. Data-integrity screening (run before submission!) -----------------
def symmetric_triplets(df, g, y):
    out = []
    for k, x in df.groupby(g, sort=False):
        v = np.sort(x[y].values)
        if len(v) == 3:
            out.append(dict(group=str(k), values=v.tolist(),
                            mid_minus_midrange=round(float(v[1] - (v[0] + v[2]) / 2), 4)))
    return out
integ = {}
for nm, df_, g, y in [("temperature", temp, "Temperature_C", "Radial_Growth_mm"),
                      ("pH", ph, "pH", "Radial_Growth_mm"),
                      ("enzyme_activity", enz, "Enzyme", "Activity_U_per_ml"),
                      ("absorbance", enz, "Enzyme", "Absorbance")]:
    t = symmetric_triplets(df_, g, y)
    integ[nm] = dict(n_triplets=len(t), n_exactly_symmetric=sum(abs(z["mid_minus_midrange"]) < 0.0051 for z in t), detail=t)
integ["absorbance_vs_activity_CV"] = {e: dict(cv_abs=float(desc_abs.loc[e, "cv_pct"]), cv_act=float(desc_e.loc[e, "cv_pct"]))
                                      for e in ENZ_ORDER}
integ["growth_values_above_90mm"] = temp.loc[temp["Radial_Growth_mm"] > 90, ["Temperature_C", "Radial_Growth_mm"]].values.tolist()
RES["integrity"] = integ

#%% ---- 7. Tables -------------------------------------------------------------
def mse(d, nd=2):
    return [f"{a:.{nd}f} ± {b:.{nd}f} {l}" for a, b, l in zip(d["mean"], d["se"], d["letter"])]
T1 = pd.DataFrame({"Temperature (°C)": desc_t.index, "Radial growth (mm)": mse(desc_t),
                   "CV (%)": desc_t["cv_pct"].round(1).values,
                   "pH": desc_p.index, "Radial growth at pH (mm)": mse(desc_p),
                   "CV (%) ": desc_p["cv_pct"].round(1).values})
rows = []
for key, lab in [("beta", "Beta (temperature)"), ("gauss", "Gaussian (pH)")]:   # best models only
    f = RES[key]
    for nm, pr in f["params"].items():
        rows.append([lab, nm, f'{pr["estimate"]:.2f}', f'{pr["ci_low"]:.2f}–{pr["ci_high"]:.2f}',
                     f'{pr["boot_low"]:.2f}–{pr["boot_high"]:.2f}'])
    rows.append([lab, "R² / adj. R²", f'{f["R2"]:.3f} / {f["adjR2"]:.3f}', "", ""])
    rows.append([lab, "RMSE (mm) / AICc", f'{f["RMSE"]:.2f} / {f["AICc"]:.1f}', "", ""])
T2 = pd.DataFrame(rows, columns=["Model", "Parameter", "Estimate", "95% CI (asymptotic)", "95% CI (bootstrap)"])
T3 = pd.DataFrame({"Enzyme": desc_e.index, "Activity (U/mL)": mse(desc_e),
                   "SD": desc_e["sd"].round(2).values, "CV (%)": desc_e["cv_pct"].round(1).values,
                   "Fold vs pectinase": desc_e["fold_vs_pectinase"].round(1).values})
T4 = pd.DataFrame({"DAI": les_desc["Length_cm"].index})
for v, lab in LES_VARS:
    T4[lab] = mse(les_desc[v])
TS1 = pd.DataFrame([s_t, s_p, s_e] + les_sum)
TS2 = pd.concat([ph_t, ph_p, ph_e] + les_post, ignore_index=True)
# main-text Table 1 = cardinal models; Supplementary Tables S1-S6
with pd.ExcelWriter(f"{OUT}/Tables_R144.xlsx") as xw:
    for nm, t in [("Table1_cardinal_models", T2), ("TableS1_growth", T1), ("TableS2_enzymes", T3),
                  ("TableS3_lesions", T4), ("TableS4_ANOVA", TS1), ("TableS5_posthoc", TS2),
                  ("TableS6_model_CIs", T2)]:
        t.to_excel(xw, nm, index=False); t.to_csv(f"{OUT}/{nm}.csv", index=False)

#%% ---- 8. Figures -------------------------------------------------------------
def jitter(n, w=0.18):
    return RNG.uniform(-w, w, n)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.tif", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

# Figure 2 – temperature and pH response with fitted cardinal models
fig, axs = plt.subplots(1, 2, figsize=(7.09, 2.9))
for ax, df_, xcol, d, mod, fn, lab, xlab, tag in [
        (axs[0], temp, "Temperature_C", desc_t, "beta", betaYH, "Topt (°C)", "Temperature (°C)", "A"),
        (axs[1], ph, "pH", desc_p, "gauss", gauss, "pHopt", "pH", "B")]:
    xs = np.linspace(df_[xcol].min() - 1.5 if xcol == "Temperature_C" else 2.6,
                     df_[xcol].max() + 1.5 if xcol == "Temperature_C" else 8.4, 400)
    p = RES[mod]["popt"]
    ax.plot(xs, fn(xs, *p), color=INK, lw=1.2, zorder=2,
            label="Beta fit" if mod == "beta" else "Gaussian fit")
    jw = 0.5 if xcol == "Temperature_C" else 0.06
    ax.scatter(df_[xcol] + RNG.uniform(-jw, jw, len(df_)), df_["Radial_Growth_mm"], s=9,
               color=BAR, edgecolor=DOT, lw=0.4, zorder=3, label="Replicate")
    ax.errorbar(d.index, d["mean"], yerr=d["se"], fmt="o", ms=4, color=INK, capsize=2.5,
                lw=0.9, zorder=4, label="Mean ± SE")
    for xi, (mu, se, l) in zip(d.index, zip(d["mean"], d["se"], d["letter"])):
        ax.text(xi, mu + se + 3.5, l, ha="center", va="bottom", fontsize=8)
    opt = RES[mod]["params"][lab]
    ax.axvline(opt["estimate"], color="#7A7A7A", ls=(0, (3, 2)), lw=0.8, zorder=1)
    ax.text(opt["estimate"], 2, f' {lab.split(" ")[0]} = {opt["estimate"]:.1f}', fontsize=7,
            color="#4A4A4A", ha="left" if mod == "gauss" else "right")
    ax.set_xlabel(xlab); ax.set_ylim(0, 108); ax.set_ylabel("Colony diameter after 3 d (mm)")
    ax.text(-0.14, 1.02, tag, transform=ax.transAxes, fontsize=11, fontweight="bold")
axs[0].set_xticks(desc_t.index); axs[1].set_xticks([3, 4, 5, 5.5, 6, 7, 8])
axs[1].set_xticklabels(["3", "4", "5", "5.5", "6", "7", "8"])
axs[0].legend(frameon=False, fontsize=7, loc="upper left"); axs[1].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout(); save(fig, "Figure3_temperature_pH")

# Figure 4 – enzyme profile
de = desc_e.sort_values("mean", ascending=False)
fig, ax = plt.subplots(figsize=(3.46, 3.0))
xpos = np.arange(len(de))
ax.bar(xpos, de["mean"], width=0.62, color=BAR, edgecolor=INK, lw=0.6, zorder=2)
ax.errorbar(xpos, de["mean"], yerr=de["se"], fmt="none", ecolor=INK, capsize=2.5, lw=0.9, zorder=3)
for i, e in enumerate(de.index):
    v = enz.loc[enz.Enzyme == e, "Activity_U_per_ml"].values
    ax.scatter(i + jitter(len(v), 0.12), v, s=8, color="white", edgecolor=DOT, lw=0.6, zorder=4)
    ax.text(i, max(v.max(), de.loc[e, "mean"] + de.loc[e, "se"]) + 0.4, de.loc[e, "letter"],
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(xpos)
ax.set_xticklabels(de.index, fontsize=7.5, rotation=35, ha="right", rotation_mode="anchor")
ax.set_ylabel("Activity (U mL$^{-1}$)"); ax.set_ylim(0, 18)
fig.tight_layout(); save(fig, "Figure5_enzymes")

# Figure 6A – discoloration-zone dimensions
fig, axs = plt.subplots(1, 4, figsize=(7.09, 2.5))
for ax, (v, lab), tag in zip(axs, LES_VARS, "ABCD"):
    d = les_desc[v]; xp = np.arange(3)
    ax.bar(xp, d["mean"], width=0.6, color=BAR, edgecolor=INK, lw=0.6, zorder=2)
    ax.errorbar(xp, d["mean"], yerr=d["se"], fmt="none", ecolor=INK, capsize=2.5, lw=0.9, zorder=3)
    for i, k in enumerate(d.index):
        vals = les.loc[les.DAI == k, v].values
        ax.scatter(i + jitter(len(vals), 0.14), vals, s=6, color="white", edgecolor=DOT, lw=0.5, zorder=4)
        top = max(vals.max(), d.loc[k, "mean"] + d.loc[k, "se"])
        ax.text(i, top * 1.03 + 0.02 * d["mean"].max(), d.loc[k, "letter"], ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xp); ax.set_xticklabels(["60", "120", "180"]); ax.set_xlabel("Days after inoculation")
    ax.set_ylabel(lab); ax.set_ylim(0, les[v].max() * 1.18)
    ax.text(-0.3, 1.02, tag, transform=ax.transAxes, fontsize=10, fontweight="bold")
fig.tight_layout(w_pad=1.2); save(fig, "Figure6A_discoloration")

# Supplementary Figure S3 – residual diagnostics
fig, axs = plt.subplots(2, 4, figsize=(7.09, 3.6))
sets = [(temp, "Temperature_C", "Radial_Growth_mm", "Temperature"), (ph, "pH", "Radial_Growth_mm", "pH"),
        (enz, "Enzyme", "Activity_U_per_ml", "Enzymes"), (les, "DAI", "Area_cm2", "Lesion area")]
for j, (df_, g, y, t) in enumerate(sets):
    fitted = df_.groupby(g)[y].transform("mean"); res_ = df_[y] - fitted
    stats.probplot(res_, dist="norm", plot=axs[0, j])
    axs[0, j].get_lines()[0].set(markersize=3, markerfacecolor=BAR, markeredgecolor=DOT)
    axs[0, j].get_lines()[1].set(color=INK, lw=0.8)
    axs[0, j].set_title(t, fontsize=8); axs[0, j].set_xlabel("Theoretical quantiles", fontsize=7)
    axs[0, j].set_ylabel("Residual", fontsize=7)
    axs[1, j].scatter(fitted, res_, s=6, color=BAR, edgecolor=DOT, lw=0.4)
    axs[1, j].axhline(0, color=INK, lw=0.7); axs[1, j].set_xlabel("Fitted (group mean)", fontsize=7)
    axs[1, j].set_ylabel("Residual", fontsize=7)
for a in axs.flat:
    a.tick_params(labelsize=6.5)
fig.tight_layout(); save(fig, "FigureS3_residual_diagnostics")

#%% ---- 9. Save everything ----------------------------------------------------
RES.update(dict(anova=TS1.to_dict("records"),
                desc_temp=desc_t.reset_index().to_dict("records"),
                desc_ph=desc_p.reset_index().to_dict("records"),
                desc_enz=desc_e.reset_index().to_dict("records"),
                desc_abs=desc_abs.reset_index().to_dict("records"),
                desc_les={k: v.reset_index().to_dict("records") for k, v in les_desc.items()},
                rates=rates.to_dict("records")))
with open(f"{OUT}/results.json", "w", encoding="utf8") as fh:
    json.dump(RES, fh, indent=1, default=float, ensure_ascii=False)

#%% ---- 10. Print a readable summary ------------------------------------------
print(f"Font used: {FONT}\n")
for s in RES["anova"]:
    print(f"{s['variable']:<30} {s['test']:<14} F({s['df1']:.0f}, {s['df2']:.2f}) = {s['F']:.2f}, "
          f"p {fmt_p(s['p'])}, eta2 = {s['eta2']:.3f} | Shapiro p = {s['shapiro_p']:.3f}, "
          f"Levene p = {s['levene_p']:.3f} -> {s['posthoc']}")
print()
for k in ["ctmi", "beta", "gauss", "cpm"]:
    f = RES[k]; print(k.upper(), {n: f"{p['estimate']:.2f} [{p['ci_low']:.2f}, {p['ci_high']:.2f}]" for n, p in f["params"].items()},
                      f"R2={f['R2']:.3f} RMSE={f['RMSE']:.2f} AICc={f['AICc']:.1f} boot={f['n_boot']}")
print("Gaussian pH range >=50% of max:", np.round(RES["gauss"]["pH50_range"], 2))
for s_ in RES["anova"]:
    print(f"  Kruskal-Wallis sensitivity {s_['variable']}: H = {s_['kruskal_H']:.2f}, p = {s_['kruskal_p']:.2g}")
print("\nExponential area model:", {k: (np.round(v, 4) if not isinstance(v, list) else np.round(v, 4).tolist()) for k, v in RES["exp_area"].items()})
print("Linear area model R2:", round(RES["lin_area"]["R2"], 3))
print("Temperature breadth, beta model (>=50%, >=80% of max):", {k: np.round(v, 1).tolist() for k, v in RES["beta"]["breadth"].items()})
for _fac, _rows in RES["model_selection"].items():
    print(f"Model selection ({_fac}):", [(r["model"], round(r["dAICc"], 1), round(r["weight"], 3), round(r["optimum"], 2), round(r["shapiro_p"], 3)) for r in _rows])
print("pH breadth (>=50%, >=80% of max):", {k: np.round(v, 2).tolist() for k, v in RES["gauss"]["breadth"].items()})
print("Relative elongation:", {k: (np.round(v, 4).tolist() if isinstance(v, list) else round(float(v), 4)) for k, v in RES["elongation"].items()})
print("SMA ln(length) ~ ln(width):", {k: (np.round(v, 2).tolist() if isinstance(v, list) else round(float(v), 3)) for k, v in RES["sma"].items()})
print("Area models:", {k: (round(v["dAICc"], 1), round(v["weight"], 3)) for k, v in RES["area_models"].items()})
print("Mean expansion rates between harvests:\n", rates.round(4).to_string(index=False))
print("Fold change 60->180 DAI:", {k: round(v, 2) for k, v in fold.items()})
print("Area vs LxW:", RES["area_vs_LxW"])
print("\nDATA-INTEGRITY SCREEN")
for k in ["temperature", "pH", "enzyme_activity", "absorbance"]:
    print(f"  {k}: {integ[k]['n_exactly_symmetric']}/{integ[k]['n_triplets']} replicate triplets have middle value = midpoint of the other two")
print("  CV absorbance vs CV activity:", {e: (round(v['cv_abs'], 2), round(v['cv_act'], 1)) for e, v in integ["absorbance_vs_activity_CV"].items()})
print("  Growth values > 90 mm (plate diameter?):", integ["growth_values_above_90mm"])
for nm, t in [("Table 1", T1), ("Table 2", T2), ("Table 3", T3), ("Table 4", T4)]:
    print(f"\n{nm}\n", t.to_string(index=False))
