"""Independent replication: ACS 1-year PUMS 2019 and 2024, vision difficulty and employment.
Run: python -I replicate.py
"""
import zipfile, json, math, sys
import numpy as np
import pandas as pd

DATA = {2019: r"data\pus_2019.zip", 2024: r"data\pus_2024.zip"}
OUT = r"independent.json"
STATEVAR = {2019: "ST", 2024: "STATE"}
REPS = ["PWGTP%d" % i for i in range(1, 81)]
W = ["PWGTP"] + REPS
KEYS = ["vis", "esr", "atwork", "wfh", "ba", "other", "ageg", "indiana", "selfemp"]


def load(year):
    z = zipfile.ZipFile(DATA[year])
    st = STATEVAR[year]
    base = ["AGEP", "ESR", "RELSHIPP", "DEYE", "DEAR", "DREM", "DPHY", "DDRS", "DOUT",
            "JWTRNS", "SCHL", "COW", "PERNP", "ADJINC", st]
    aggs, micro = [], []
    nrec = 0
    for name in sorted(n for n in z.namelist() if n.lower().endswith(".csv")):
        with z.open(name) as f:
            hdr = f.readline().decode().strip().split(",")
        up = {c.upper(): c for c in hdr}
        missing = [c for c in base + W if c not in up]
        if missing:
            raise SystemExit("%d %s missing columns %s" % (year, name, missing))
        use = [up[c] for c in base + W]
        with z.open(name) as f:
            for ch in pd.read_csv(f, usecols=use, chunksize=250000, dtype="float64"):
                ch.columns = [c.upper() for c in ch.columns]
                nrec += len(ch)
                u = ch[(ch.AGEP >= 18) & (ch.AGEP <= 64)
                       & ~ch.ESR.isin([4, 5]) & (ch.RELSHIPP != 37)].copy()
                # sanity: no missing values in universe-defining fields
                assert u.ESR.notna().all() and u.DEYE.isin([1, 2]).all()
                for d in ["DEAR", "DREM", "DPHY", "DDRS", "DOUT"]:
                    assert u[d].isin([1, 2]).all(), d
                u["vis"] = (u.DEYE == 1).astype(int)
                u["esr"] = u.ESR.astype(int)
                u["atwork"] = u.JWTRNS.notna().astype(int)
                u["wfh"] = (u.JWTRNS == 11).astype(int)
                u["ba"] = (u.SCHL >= 21).astype(int)
                u["other"] = ((u.DEAR == 1) | (u.DREM == 1) | (u.DPHY == 1)
                              | (u.DDRS == 1) | (u.DOUT == 1)).astype(int)
                u["ageg"] = np.select([u.AGEP <= 34, u.AGEP <= 49], [0, 1], 2)
                u["indiana"] = (u[st] == 18).astype(int)
                u["selfemp"] = u.COW.isin([6, 7]).astype(int)
                aggs.append(u.groupby(KEYS)[W].sum())
                micro.append(u.loc[(u.vis == 1) & (u.atwork == 1), ["wfh", "PERNP", "ADJINC", "PWGTP"]])
        print(year, name, "done", nrec, flush=True)
    agg = pd.concat(aggs).groupby(level=KEYS).sum().reset_index()
    return agg, pd.concat(micro), nrec


def tot(agg, mask):
    return agg.loc[mask, W].sum().values  # length 81


def se_from(vals):
    return math.sqrt(4.0 / 80.0 * float(((vals[1:] - vals[0]) ** 2).sum()))


def pct(agg, num, den):
    n, d = tot(agg, num & den), tot(agg, den)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 * n / d
    se = se_from(r)
    return {"pct": float(r[0]), "se": se, "moe90": 1.645 * se,
            "numerator_weighted": int(round(n[0])), "denominator_weighted": int(round(d[0]))}


def count(agg, mask):
    t = tot(agg, mask)
    se = se_from(t)
    return {"count": int(round(t[0])), "se": se, "moe90": 1.645 * se}


def wmedian(v, w):
    o = np.argsort(v, kind="mergesort")
    v, w = v[o], w[o]
    c = np.cumsum(w)
    half = c[-1] / 2.0
    i = int(np.searchsorted(c, half, side="left"))
    if c[i] == half and i + 1 < len(v):
        return float((v[i] + v[i + 1]) / 2.0)
    return float(v[i])


res = {"_notes": {
    "universe": "AGEP 18-64, ESR not in (4,5), RELSHIPP != 37; weight PWGTP",
    "moe": "SE = sqrt(4/80 * sum_r (est_r - est)^2) over PWGTP1..PWGTP80; MOE90 = 1.645*SE",
}}
for year in (2019, 2024):
    agg, micro, nrec = load(year)
    a = agg
    T = pd.Series(True, index=a.index)
    emp = a.esr.isin([1, 2]); unemp = a.esr == 3; lf = a.esr.isin([1, 2, 3]); nilf = a.esr == 6
    atw = a.atwork == 1; wfh = a.wfh == 1
    R = {"records_read_all_ages": nrec}
    # consistency check: at work (JWTRNS reported) vs ESR==1
    R["check_atwork_by_esr_weighted"] = {str(k): int(v) for k, v in
                                         a[atw].groupby("esr").PWGTP.sum().items()}
    R["check_esr1_without_jwtrns_weighted"] = int(a[(a.esr == 1) & ~atw].PWGTP.sum())
    for label, v in (("with_vision_difficulty", 1), ("without_vision_difficulty", 0)):
        g = a.vis == v
        d = {"a_population": count(a, g),
             "b_pct_employed": pct(a, emp, g),
             "b_unemployment_rate": pct(a, unemp, g & lf),
             "b_pct_not_in_labor_force": pct(a, nilf, g),
             "c_pct_worked_from_home_among_at_work": pct(a, wfh, g & atw)}
        if year == 2024:
            for bl, bv in (("bachelors_or_higher", 1), ("less_than_bachelors", 0)):
                gb = g & (a.ba == bv)
                d["d_" + bl] = {"pct_employed": pct(a, emp, gb),
                                "pct_worked_from_home_among_at_work": pct(a, wfh, gb & atw)}
        R[label] = d
    g = a.vis == 1
    R["e_vision_only_no_other_difficulty"] = {"population": count(a, g & (a.other == 0)),
                                              "pct_employed": pct(a, emp, g & (a.other == 0))}
    R["e_vision_plus_at_least_one_other_difficulty"] = {"population": count(a, g & (a.other == 1)),
                                                        "pct_employed": pct(a, emp, g & (a.other == 1))}
    R["i_vision_wfh_pct_self_employed"] = pct(a, a.selfemp == 1, g & wfh)
    if year == 2024:
        R["f_vision_difficulty_indiana_population"] = count(a, g & (a.indiana == 1))
        R["g_vision_bachelors_pct_employed_by_age"] = {
            n: pct(a, emp, g & (a.ba == 1) & (a.ageg == k))
            for k, n in ((0, "age_18_34"), (1, "age_35_49"), (2, "age_50_64"))}
        h = {}
        adj = micro.ADJINC.iloc[0] / 1e6
        assert (micro.ADJINC == micro.ADJINC.iloc[0]).all()
        for n, k in (("worked_from_home", 1), ("did_not_work_from_home", 0)):
            m = micro[micro.wfh == k]
            assert m.PERNP.notna().all()
            med = wmedian(m.PERNP.values, m.PWGTP.values)
            pos = m[m.PERNP > 0]
            h[n] = {"weighted_median_PERNP_as_reported": med,
                    "weighted_median_PERNP_times_ADJINC": med * adj,
                    "weighted_median_PERNP_excluding_zero_or_negative_earnings":
                        wmedian(pos.PERNP.values, pos.PWGTP.values),
                    "unweighted_n": int(len(m)), "weighted_n": int(m.PWGTP.sum())}
        h["ADJINC_factor"] = adj
        R["h_vision_at_work_median_earnings"] = h
    res[str(year)] = R

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=1))
