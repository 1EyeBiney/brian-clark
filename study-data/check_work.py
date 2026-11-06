"""Checks on the study's own numbers. Writes results/checks.json.

    python check_work.py

1. Compares stage-one results with the Census Bureau's own published table
   B18120 (employment status by disability status and type, ages 18 to 64),
   read from the files in the "official" folder.
2. Puts a 90 percent margin of error on every difference and derived figure the
   page relies on, using the 80 replicate weights. A difference is called
   "clear" when it is larger than its margin of error.
3. Recomputes the "what if" figures a second way.
4. Counts people who are not working now but have worked in the past five years.
5. Recomputes median earnings with margins of error.

For a figure that uses more than one year, the years are independent samples:
the replicate estimates are formed by varying one year at a time and the
variances are added.
"""
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
REPS = [f'PWGTP{i}' for i in range(1, 81)]
W = ['PWGTP'] + REPS


class Cube:
    def __init__(self, pattern):
        frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(here, 'results', pattern)))]
        self.by_year = {int(y): d for y, d in pd.concat(frames, ignore_index=True).groupby('year')}
        self.years = sorted(self.by_year)
        self.cache = {}

    def V(self, year, **kw):
        """Weighted total (main weight then 80 replicates) for one year and a set of filters."""
        key = (year, tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in kw.items())))
        if key not in self.cache:
            d = self.by_year[year]
            for k, v in kw.items():
                d = d[d[k].isin(v)] if isinstance(v, (list, tuple)) else d[d[k] == v]
            self.cache[key] = d[W].sum().to_numpy(dtype='float64')
        return self.cache[key]

    def evaluate(self, stat, digits=1):
        """stat(T) -> number, where T(year, **filters) gives a weighted total."""
        est = float(stat(lambda y, **kw: self.V(y, **kw)[0]))
        var = 0.0
        for yv in self.years:
            reps = stat(lambda y, **kw: self.V(y, **kw)[1:] if y == yv else self.V(y, **kw)[0])
            reps = np.broadcast_to(np.asarray(reps, dtype='float64'), (80,))
            var += 4.0 / 80.0 * float(((reps - est) ** 2).sum())
        moe = 1.645 * var ** 0.5
        return {'est': round(est, digits), 'moe': round(moe, digits), 'clear': bool(abs(est) > moe)}


cube = Cube('cube_*_pus_*.csv')
years = cube.years
Y0, Y1 = years[0], years[-1]
AT = ['home', 'onsite']
out = {'years': years}


def pct(T, y, num, den):
    return 100.0 * T(y, **num) / T(y, **den)


def emp(T, y, **g):
    return pct(T, y, dict(g, status='employed'), g)


def wfh(T, y, **g):
    return pct(T, y, dict(g, place='home'), dict(g, place=AT))


# ---------------------------------------------------------------- 1. official
official = {}
for f in sorted(glob.glob(os.path.join(here, 'official', 'b18120_*.dat'))):
    y = int(os.path.basename(f).split('_')[1].split('.')[0])
    for row in csv.DictReader(open(f, encoding='utf-8'), delimiter='|'):
        if row['GEO_ID'] in ('0100000US', '0400000US18'):
            e, u, n = (int(row[f'B18120_E{k:03d}']) for k in (6, 15, 24))
            official[(y, row['GEO_ID'])] = {'employed': e, 'unemployed': u, 'nilf': n, 'population': e + u + n,
                                            'employment_rate_pct': 100.0 * e / (e + u + n),
                                            'unemployment_rate_pct': 100.0 * u / (e + u),
                                            'not_in_labor_force_pct': 100.0 * n / (e + u + n)}
mine = {}
for f in glob.glob(os.path.join(here, 'results', 'acs_*.json')):
    d = json.load(open(f, encoding='utf-8'))
    mine[d['year']] = d['vision']
out['official'] = {}
print('=== 1. Census Bureau table B18120 against this study (people with vision difficulty, ages 18-64)')
for y in years:
    o = official.get((y, '0100000US'))
    if not o:
        continue
    m = mine[y]
    rec = {'published_employment_rate': round(o['employment_rate_pct'], 2), 'mine_employment_rate': round(m['employment_rate_pct']['estimate'], 2),
           'mine_employment_rate_moe': m['employment_rate_pct']['moe90'],
           'published_unemployment_rate': round(o['unemployment_rate_pct'], 2), 'mine_unemployment_rate': round(m['unemployment_rate_pct']['estimate'], 2),
           'published_nilf': round(o['not_in_labor_force_pct'], 2), 'mine_nilf': round(m['not_in_labor_force_pct']['estimate'], 2),
           'published_population': o['population'], 'mine_population': round(m['population']['estimate']),
           'published_employed': o['employed'], 'mine_employed': round(m['employed']['estimate'])}
    rec['employment_rate_difference'] = round(rec['mine_employment_rate'] - rec['published_employment_rate'], 2)
    rec['population_difference_pct'] = round(100.0 * (rec['mine_population'] - o['population']) / o['population'], 2)
    out['official'][str(y)] = rec
    print(' ', y, rec)
ind = [official[(y, '0400000US18')] for y in years if y >= 2022 and (y, '0400000US18') in official]
if ind:
    out['official_indiana_pooled_2022_on'] = round(100.0 * sum(i['employed'] for i in ind) / sum(i['population'] for i in ind), 1)
    out['official_indiana_population_last'] = official[(Y1, '0400000US18')]['population']
    print('  Indiana published, pooled 2022 on:', out['official_indiana_pooled_2022_on'], 'population', out['official_indiana_population_last'])

# ---------------------------------------------------------------- 2. differences
print('=== 2. Differences and derived figures, with 90% margins of error')
res = {}


def add(key, stat, digits=1):
    res[key] = cube.evaluate(stat, digits)
    r = res[key]
    print(f"  {key}: {r['est']} +/- {r['moe']}{'' if r['clear'] else '   (NOT clear of its margin of error)'}")


for y in years:
    add(f'emp_vision_{y}', lambda T, y=y: emp(T, y, vision=1))
    add(f'emp_others_{y}', lambda T, y=y: emp(T, y, vision=0))
    add(f'wfh_vision_{y}', lambda T, y=y: wfh(T, y, vision=1))
    add(f'wfh_others_{y}', lambda T, y=y: wfh(T, y, vision=0))
    add(f'wfh_gap_{y}', lambda T, y=y: wfh(T, y, vision=0) - wfh(T, y, vision=1))
    add(f'not_employed_vision_{y}', lambda T, y=y: 100 - emp(T, y, vision=1))
add('emp_change_vision', lambda T: emp(T, Y1, vision=1) - emp(T, Y0, vision=1))
add('emp_change_others', lambda T: emp(T, Y1, vision=0) - emp(T, Y0, vision=0))
add('emp_change_vision_minus_others', lambda T: (emp(T, Y1, vision=1) - emp(T, Y0, vision=1)) - (emp(T, Y1, vision=0) - emp(T, Y0, vision=0)))
for col in (1, 0):
    add(f'wfh_gap_{Y1}_college{col}', lambda T, col=col: wfh(T, Y1, vision=0, college=col) - wfh(T, Y1, vision=1, college=col))
    add(f'emp_gap_{Y1}_college{col}', lambda T, col=col: emp(T, Y1, vision=0, college=col) - emp(T, Y1, vision=1, college=col))
add(f'emp_vision_degree_minus_others_nodegree_{Y1}', lambda T: emp(T, Y1, vision=1, college=1) - emp(T, Y1, vision=0, college=0))
add(f'wfh_gap_{Y1}_office_admin', lambda T: wfh(T, Y1, vision=0, occ='office_admin') - wfh(T, Y1, vision=1, occ='office_admin'))
for a in ('18-34', '35-49', '50-64'):
    add(f'emp_gap_{Y1}_degree_age_{a}', lambda T, a=a: emp(T, Y1, vision=0, college=1, age=a) - emp(T, Y1, vision=1, college=1, age=a))
add(f'emp_gap_{Y1}_degree_vision_only', lambda T: emp(T, Y1, vision=0, college=1) - emp(T, Y1, vision=1, college=1, other_dis=0))
add(f'degree_holders_vision_{Y1}', lambda T: T(Y1, vision=1, college=1), 0)
add(f'degree_holders_vision_not_employed_{Y1}', lambda T: T(Y1, vision=1, college=1, status=['unemployed', 'nilf']), 0)
add(f'degree_holders_vision_nilf_{Y1}', lambda T: T(Y1, vision=1, college=1, status='nilf'), 0)

# who is not employed: the groups nearest the old "70 percent" figure
for label, g in (('vision_other_dis', dict(vision=1, other_dis=1)), ('vision_only', dict(vision=1, other_dis=0)),
                 ('vision_no_degree', dict(vision=1, college=0)), ('vision_no_degree_other_dis', dict(vision=1, college=0, other_dis=1)),
                 ('vision_degree_only', dict(vision=1, college=1, other_dis=0)), ('others', dict(vision=0)),
                 ('vision_50_64', dict(vision=1, age='50-64'))):
    add(f'not_employed_{label}_{Y1}', lambda T, g=g: 100 - emp(T, Y1, **g))
    add(f'population_{label}_{Y1}', lambda T, g=g: T(Y1, **g), 0)
add(f'unemployment_rate_vision_{Y1}', lambda T: 100.0 * T(Y1, vision=1, status='unemployed') / T(Y1, vision=1, status=['employed', 'unemployed']))
add(f'nilf_vision_{Y1}', lambda T: 100.0 * T(Y1, vision=1, status='nilf') / T(Y1, vision=1))

# Indiana
P = [y for y in years if y >= 2022]


def pooled(T, num, den):
    return 100.0 * sum(T(y, **num) for y in P) / sum(T(y, **den) for y in P)


add('indiana_emp_change', lambda T: pooled(T, dict(indiana=1, vision=1, status='employed'), dict(indiana=1, vision=1)) - emp(T, Y0, indiana=1, vision=1))
add('indiana_wfh_gap', lambda T: pooled(T, dict(indiana=1, vision=0, place='home'), dict(indiana=1, vision=0, place=AT))
    - pooled(T, dict(indiana=1, vision=1, place='home'), dict(indiana=1, vision=1, place=AT)))

# ---------------------------------------------------------------- 3. what-if figures, two ways
print('=== 3. The "what if" figures, each computed two ways')


def home_pop(T, y, **g):
    return pct(T, y, dict(g, place='home'), g)


def atwork_pop(T, y, **g):
    return pct(T, y, dict(g, place=AT), g)


# share of the gain that is work from home
add('share_home_rate_method', lambda T: 100.0 * (home_pop(T, Y1, vision=1) - home_pop(T, Y0, vision=1)) / (atwork_pop(T, Y1, vision=1) - atwork_pop(T, Y0, vision=1)))
add('share_home_count_method', lambda T: 100.0 * (T(Y1, vision=1, place='home') - T(Y0, vision=1, place='home')) / (T(Y1, vision=1, place=AT) - T(Y0, vision=1, place=AT)))
for od in (0, 1):
    add(f'share_home_rate_method_otherdis{od}', lambda T, od=od: 100.0 * (home_pop(T, Y1, vision=1, other_dis=od) - home_pop(T, Y0, vision=1, other_dis=od))
        / (atwork_pop(T, Y1, vision=1, other_dis=od) - atwork_pop(T, Y0, vision=1, other_dis=od)))


def share_home_fixed_mix(T, cells):
    """Share of the within-group rise in the at-work rate that is work from home, holding the mix of groups at its two-year average."""
    tot0, tot1 = T(Y0, vision=1), T(Y1, vision=1)
    dh = da = 0.0
    for g in cells:
        s = (T(Y0, vision=1, **g) / tot0 + T(Y1, vision=1, **g) / tot1) / 2.0
        dh = dh + s * (home_pop(T, Y1, vision=1, **g) - home_pop(T, Y0, vision=1, **g))
        da = da + s * (atwork_pop(T, Y1, vision=1, **g) - atwork_pop(T, Y0, vision=1, **g))
    return 100.0 * dh / da


add('share_home_fixed_mix_two_groups', lambda T: share_home_fixed_mix(T, [dict(other_dis=o) for o in (0, 1)]))
add('share_home_fixed_mix_twelve_groups', lambda T: share_home_fixed_mix(T, [dict(other_dis=o, age=a, college=c) for o in (0, 1) for a in ('18-34', '35-49', '50-64') for c in (0, 1)]))
add('home_rate_change', lambda T: home_pop(T, Y1, vision=1) - home_pop(T, Y0, vision=1), 2)
add('onsite_rate_change', lambda T: (atwork_pop(T, Y1, vision=1) - home_pop(T, Y1, vision=1)) - (atwork_pop(T, Y0, vision=1) - home_pop(T, Y0, vision=1)), 2)

# composition: how much of the employment gain is a change in who is counted
def decomp(T, cells, weights):
    """Gain within groups. weights: 'first' (mix of first year), 'last', or 'average'."""
    tot0, tot1 = T(Y0, vision=1), T(Y1, vision=1)
    within = 0.0
    for g in cells:
        s0, s1 = T(Y0, vision=1, **g) / tot0, T(Y1, vision=1, **g) / tot1
        r0, r1 = emp(T, Y0, vision=1, **g), emp(T, Y1, vision=1, **g)
        s = {'first': s0, 'last': s1, 'average': (s0 + s1) / 2.0}[weights]
        within = within + s * (r1 - r0)
    return within


two = [dict(other_dis=o) for o in (0, 1)]
fine = [dict(other_dis=o, age=a, college=c) for o in (0, 1) for a in ('18-34', '35-49', '50-64') for c in (0, 1)]
add('gain_total', lambda T: emp(T, Y1, vision=1) - emp(T, Y0, vision=1))
for name, cells in (('two_groups', two), ('twelve_groups', fine)):
    for wt in ('first', 'last', 'average'):
        add(f'gain_within_{name}_{wt}_mix', lambda T, cells=cells, wt=wt: decomp(T, cells, wt))
        add(f'gain_mix_{name}_{wt}_mix', lambda T, cells=cells, wt=wt: (emp(T, Y1, vision=1) - emp(T, Y0, vision=1)) - decomp(T, cells, wt))

# standardised work-from-home rates
OCC = ['management_professional', 'office_admin', 'sales', 'service', 'construction_production_transport', 'none']


def std(T, cells, rates_from, mix_from):
    tot = T(Y1, vision=mix_from, place=AT)
    s = 0.0
    for g in cells:
        den = T(Y1, vision=rates_from, place=AT, **g)
        rate = T(Y1, vision=rates_from, place='home', **g) / np.where(den == 0, 1, den)
        s = s + (T(Y1, vision=mix_from, place=AT, **g) / tot) * rate
    return 100.0 * s


edu = [dict(college=c) for c in (0, 1)]
occ = [dict(occ=o) for o in OCC]
both = [dict(college=c, occ=o) for c in (0, 1) for o in OCC]
for name, cells in (('education', edu), ('occupation', occ), ('education_and_occupation', both)):
    add(f'wfh_vision_rates_others_mix_{name}', lambda T, cells=cells: std(T, cells, 1, 0))
    add(f'wfh_others_rates_vision_mix_{name}', lambda T, cells=cells: std(T, cells, 0, 1))
    # share of the gap explained, both directions
    add(f'gap_explained_pct_{name}_method1', lambda T, cells=cells: 100.0 * (std(T, cells, 1, 0) - wfh(T, Y1, vision=1)) / (wfh(T, Y1, vision=0) - wfh(T, Y1, vision=1)))
    add(f'gap_explained_pct_{name}_method2', lambda T, cells=cells: 100.0 * (wfh(T, Y1, vision=0) - std(T, cells, 0, 1)) / (wfh(T, Y1, vision=0) - wfh(T, Y1, vision=1)))
out['checks'] = res

# ---------------------------------------------------------------- 4. work experience
if glob.glob(os.path.join(here, 'results', 'wkl_*.csv')):
    print('=== 4. People not working now who have worked before')
    wk = Cube('wkl_*.csv')
    wres = {}
    NE = ['unemployed', 'nilf']

    def wadd(key, stat, digits=1):
        wres[key] = wk.evaluate(stat, digits)
        print(f"  {key}: {wres[key]['est']} +/- {wres[key]['moe']}")

    for y in wk.years:
        for vis in (1, 0):
            for label, g in (('all', {}), ('degree', dict(college=1)), ('no_degree', dict(college=0))):
                base = dict(g, vision=vis, status=NE)
                wadd(f'recent_work_pct_{y}_vision{vis}_{label}', lambda T, y=y, base=base: 100.0 * T(y, wkl=[1, 2], **base) / T(y, **base))
                wadd(f'last_year_work_pct_{y}_vision{vis}_{label}', lambda T, y=y, base=base: 100.0 * T(y, wkl=1, **base) / T(y, **base))
                if vis == 1:
                    wadd(f'recent_work_count_{y}_vision1_{label}', lambda T, y=y, base=base: T(y, wkl=[1, 2], **base), 0)
                    wadd(f'not_employed_count_{y}_vision1_{label}', lambda T, y=y, base=base: T(y, **base), 0)
        for a in ('18-34', '35-49', '50-64'):
            base = dict(vision=1, status=NE, age=a)
            wadd(f'recent_work_pct_{y}_vision1_age_{a}', lambda T, y=y, base=base: 100.0 * T(y, wkl=[1, 2], **base) / T(y, **base))
        wadd(f'check_emp_vision_{y}', lambda T, y=y: 100.0 * T(y, vision=1, status='employed') / T(y, vision=1))
    out['work_experience'] = wres

# ---------------------------------------------------------------- 5. median earnings with margins of error
med = {}
for f in sorted(glob.glob(os.path.join(here, 'results', 'earnhist_*.csv'))):
    e = pd.read_csv(f)
    y = int(e['year'].iloc[0])
    e = e[e['PERNP'].notna()].sort_values('PERNP')
    print(f'=== 5. Median earnings {y}, people at work')
    for vis in (1, 0):
        for place in AT:
            sub = e[(e['vision'] == vis) & (e['place'] == place)]
            vals = sub['PERNP'].to_numpy(dtype='float64')
            m = []
            for w in W:
                cum = np.cumsum(sub[w].to_numpy(dtype='float64'))
                m.append(float(vals[np.searchsorted(cum, cum[-1] / 2.0)]))
            moe = 1.645 * (4.0 / 80.0 * sum((x - m[0]) ** 2 for x in m[1:])) ** 0.5
            key = f"median_{y}_{'vision' if vis else 'others'}_{place}"
            med[key] = {'est': m[0], 'moe': round(moe), 'records': int(sub['records'].sum())}
            print(f'  {key}: {m[0]:,.0f} +/- {moe:,.0f} ({int(sub["records"].sum()):,} records)')
out['median_earnings'] = med

# ---------------------------------------------------------------- 6. independent replication
ipath = os.path.join(here, 'verify', 'independent.json')
if os.path.exists(ipath):
    print('=== 6. Independent replication (a second program, written without sight of this one)')
    ind = json.load(open(ipath, encoding='utf-8'))

    def exact(stat):
        return float(stat(lambda y, **kw: cube.V(y, **kw)[0]))

    pairs = []
    for y in (Y0, Y1):
        i = ind[str(y)]
        for vis, name in ((1, 'with_vision_difficulty'), (0, 'without_vision_difficulty')):
            g = i[name]
            pairs += [(f'{y} {name} population', g['a_population']['count'], exact(lambda T: T(y, vision=vis)), True),
                      (f'{y} {name} percent employed', g['b_pct_employed']['pct'], exact(lambda T: emp(T, y, vision=vis)), False),
                      (f'{y} {name} percent not in labor force', g['b_pct_not_in_labor_force']['pct'], exact(lambda T: 100.0 * T(y, vision=vis, status='nilf') / T(y, vision=vis)), False),
                      (f'{y} {name} work from home', g['c_pct_worked_from_home_among_at_work']['pct'], exact(lambda T: wfh(T, y, vision=vis)), False)]
        for od, name in ((0, 'e_vision_only_no_other_difficulty'), (1, 'e_vision_plus_at_least_one_other_difficulty')):
            pairs += [(f'{y} {name} population', i[name]['population']['count'], exact(lambda T: T(y, vision=1, other_dis=od)), True),
                      (f'{y} {name} percent employed', i[name]['pct_employed']['pct'], exact(lambda T: emp(T, y, vision=1, other_dis=od)), False)]
        pairs.append((f'{y} self-employed share of vision work from home', i['i_vision_wfh_pct_self_employed']['pct'],
                      exact(lambda T: 100.0 * T(y, vision=1, place='home', selfemp=1) / T(y, vision=1, place='home')), False))
    i = ind[str(Y1)]
    for vis, name in ((1, 'with_vision_difficulty'), (0, 'without_vision_difficulty')):
        for col, cname in ((1, 'd_bachelors_or_higher'), (0, 'd_less_than_bachelors')):
            pairs += [(f'{Y1} {name} {cname} employed', i[name][cname]['pct_employed']['pct'], exact(lambda T: emp(T, Y1, vision=vis, college=col)), False),
                      (f'{Y1} {name} {cname} work from home', i[name][cname]['pct_worked_from_home_among_at_work']['pct'], exact(lambda T: wfh(T, Y1, vision=vis, college=col)), False)]
    for a in ('18-34', '35-49', '50-64'):
        pairs.append((f'{Y1} degree holders with vision difficulty employed, age {a}', i['g_vision_bachelors_pct_employed_by_age']['age_' + a.replace('-', '_')]['pct'],
                      exact(lambda T: emp(T, Y1, vision=1, college=1, age=a)), False))
    pairs.append((f'{Y1} Indiana population with vision difficulty', i['f_vision_difficulty_indiana_population']['count'], exact(lambda T: T(Y1, vision=1, indiana=1)), True))
    for place, name in (('home', 'worked_from_home'), ('onsite', 'did_not_work_from_home')):
        pairs.append((f'{Y1} median earnings, vision difficulty, {name}', i['h_vision_at_work_median_earnings'][name]['weighted_median_PERNP_as_reported'],
                      med[f'median_{Y1}_vision_{place}']['est'], True))
    worst_pct = max(abs(a - b) for _, a, b, is_count in pairs if not is_count)
    worst_count = max(abs(a - b) for _, a, b, is_count in pairs if is_count)
    for label, a, b, is_count in pairs:
        if abs(a - b) > (0.5 if is_count else 0.0005):
            print('  DIFFERS:', label, a, b)
    out['independent'] = {'figures_compared': len(pairs), 'largest_difference_in_a_percent': round(worst_pct, 6),
                          'largest_difference_in_a_count': round(worst_count, 3)}
    print(' ', out['independent'])

with open(os.path.join(here, 'results', 'checks.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('saved results/checks.json')
